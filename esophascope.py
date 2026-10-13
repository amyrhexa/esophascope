"""EsophaScope: Quantitative Whole-Slide Image Analysis & Hotspot Localization Suite.

Clinical pathology workstation for esophageal biopsy screening, automated
eosinophil detection, diagnostic High-Power Field (HPF) quantification, and
bidirectional QuPath GeoJSON synchronization.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime
import json
import logging
import math
import os
import sys
import threading
from collections import OrderedDict, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import openslide
from PIL import Image
from PySide6.QtCore import (
    QByteArray,
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    QStandardPaths,
    Qt,
    QThread,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QShortcut,
    QTransform,
    QWheelEvent,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QStyleOptionGraphicsItem,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial import KDTree

logger = logging.getLogger("esophascope")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ==============================================================================
# CLINICAL BENCHMARKS & PIPELINE CONSTANTS
# ==============================================================================
APP_ID = "esophascope"
APP_ICON_FILE = "app.ico"
DEFAULT_MODEL_WEIGHTS = "best.onnx"
TARGET_MODEL_MPP = 0.173
PATCH_PIXEL_DIMENSION = 640
# Fill for the unused area of edge patches (Ultralytics letterbox grey).
PATCH_PAD_VALUE = 114
PATCH_MICRON_SPAN = PATCH_PIXEL_DIMENSION * TARGET_MODEL_MPP

# Standard clinical HPF area: 0.238 mm²
CLINICAL_HPF_AREA_MM2 = 0.238
CLINICAL_HPF_AREA_UM2 = CLINICAL_HPF_AREA_MM2 * 1_000_000.0
CLINICAL_HPF_RADIUS_UM = math.sqrt(CLINICAL_HPF_AREA_UM2 / math.pi)
EOE_DIAGNOSTIC_HOTSPOT_THRESHOLD = 15

CENTROID_MATCH_TOLERANCE_UM = 7.0
BASE_TENSOR_EXTRACTION_CONFIDENCE = 0.10
DEFAULT_OPERATIONAL_CONFIDENCE = 0.25
SLIDE_NMS_IOU_THRESHOLD = 0.45
# Per-patch NMS settings matching the Ultralytics predict() defaults that
# produced the original results (iou=0.7, max_det=300).
PATCH_NMS_IOU_THRESHOLD = 0.7
PATCH_MAX_DETECTIONS = 300
PATCH_STRIDE_OVERLAP_RATIO = 0.20
TISSUE_COVERAGE_MINIMUM_FRACTION = 0.05
OVERVIEW_THUMBNAIL_MAX_DIMENSION = 2048
INFERENCE_BATCH_CAPACITY = 8
# Tile edge in *pyramid-level* pixels (matches common 256/512 px scanner tiles).
PYRAMIDAL_TILE_EDGE_PX = 512
TILE_FETCH_MAX_RETRIES = 2
TILE_CACHE_MIN_BYTES = 256 * 1024**2
TILE_CACHE_MAX_BYTES = 1024 * 1024**2
TILE_CACHE_RAM_FRACTION = 0.04
PLAUSIBLE_MPP_RANGE_UM = (0.05, 2.0)
MODEL_LOAD_POLL_SECONDS = 0.1
SLIDE_EXTENSIONS = frozenset({".svs", ".ndpi", ".mrxs", ".tif", ".tiff"})
ANNOTATION_EXTENSIONS = frozenset({".geojson", ".json"})

# Viewport HUD Margin Defaults
DEFAULT_HPF_BUTTON_MARGIN_LEFT = 70
DEFAULT_HPF_BUTTON_MARGIN_BOTTOM = 15

# Visual Palette
COLOR_CANVAS_BACKGROUND = "#0F172A"
COLOR_SHELL_BACKGROUND = "#F1F5F9"
COLOR_SURFACE_PRIMARY = "#FFFFFF"
COLOR_SURFACE_DISABLED = "#E2E8F0"
COLOR_STROKE_LIGHT = "#CBD5E1"
COLOR_STROKE_BOLD = "#94A3B8"
COLOR_TEXT_EMPHASIZED = "#0F172A"
COLOR_TEXT_SECONDARY = "#475569"
COLOR_TEXT_DISABLED = "#94A3B8"
COLOR_TEXT_LIGHT = "#FFFFFF"
COLOR_ACCENT_BASE = "#2563EB"
COLOR_ACCENT_HOVER = "#1D4ED8"
COLOR_ACCENT_PRESSED = "#1E40AF"
COLOR_ACCENT_HIGHLIGHT = "#60A5FA"
COLOR_ALERT_CRITICAL = "#DC2626"
COLOR_ALERT_HOVER = "#B91C1C"
COLOR_STATUS_NEGATIVE = "#0D9488"
COLOR_TISSUE_OVERLAY = "#2563EB"
COLOR_GROUND_TRUTH = "#16A34A"
COLOR_PREDICTION_BOX = "#DC2626"
COLOR_HOTSPOT_BOUNDARY = "#D97706"


def resource_path(relative: str) -> Path:
    """Resolves a bundled asset for both source runs and PyInstaller builds."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


class _WindowsMemoryStatus(ctypes.Structure):
    """MEMORYSTATUSEX, used only on Windows to read total physical memory."""

    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def total_physical_memory_bytes() -> int:
    """Total installed RAM; falls back to 4 GiB when the platform cannot say."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        pass
    if sys.platform == "win32":
        status = _WindowsMemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    return 4 * 1024**3


def tile_cache_budget_bytes() -> int:
    """Viewport tile cache budget: a small fraction of RAM, clamped to sane bounds."""
    budget = int(total_physical_memory_bytes() * TILE_CACHE_RAM_FRACTION)
    return max(TILE_CACHE_MIN_BYTES, min(TILE_CACHE_MAX_BYTES, budget))


def tile_worker_count() -> int:
    """Concurrent viewport tile decoders: scales with CPU count, bounded to 2-6."""
    return max(2, min(6, os.cpu_count() or 2))


def patch_reader_count() -> int:
    """Read-ahead threads for inference; leaves cores for the ONNX Runtime session."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


# 24x24 stroke glyphs (Lucide-style). Rendered from SVG at several pixel sizes so
# they stay sharp at any DPI scale and look identical on Windows and Linux,
# unlike QStyle.standardIcon() which depends on the platform theme.
_ICON_SHAPES = {
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "file-text": (
        '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
        '<path d="M14 3v5h5M9 13h6M9 17h6"/>'
    ),
    "play": '<path d="M7 4.5v15l12-7.5z" fill="currentColor"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>',
    "download": '<path d="M12 3v12M7 10l5 5 5-5M4 21h16"/>',
    "chevron-left": '<path d="M15 5l-7 7 7 7"/>',
    "chevron-right": '<path d="M9 5l7 7-7 7"/>',
    "crosshair": (
        '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/>'
        '<path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>'
    ),
}
_ICON_PIXEL_SIZES = (16, 20, 24, 32, 40, 48, 64)


def build_svg_icon(
    name: str,
    color: str = COLOR_TEXT_EMPHASIZED,
    disabled_color: str = COLOR_TEXT_DISABLED,
    active_color: str | None = None,
) -> QIcon:
    """Renders a named glyph into a multi-resolution QIcon with state colors."""
    icon = QIcon()
    tints = {
        QIcon.Mode.Normal: color,
        QIcon.Mode.Disabled: disabled_color,
        QIcon.Mode.Active: active_color or color,
    }
    for mode, tint in tints.items():
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="{tint}" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round">{_ICON_SHAPES[name].replace("currentColor", tint)}</svg>'
        )
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        for size in _ICON_PIXEL_SIZES:
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHints(
                QPainter.RenderHint.Antialiasing
                | QPainter.RenderHint.SmoothPixmapTransform
            )
            renderer.render(painter)
            painter.end()
            icon.addPixmap(pixmap, mode)
    return icon


def load_app_icon() -> QIcon:
    """Loads the multi-size application icon (empty icon if the file is missing)."""
    icon_file = resource_path(APP_ICON_FILE)
    if not icon_file.is_file():
        logger.warning("Application icon not found: %s", icon_file)
        return QIcon()
    return QIcon(str(icon_file))


def build_workbench_stylesheet() -> str:
    """Builds desktop UI stylesheet including status HUD decorations."""
    return f"""
    QMainWindow {{ background: {COLOR_SHELL_BACKGROUND}; }}
    QWidget {{
        font-family: "Segoe UI", "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 10pt;
        color: {COLOR_TEXT_EMPHASIZED};
    }}
    QFrame#esophaToolbar, QFrame#esophaSidebar {{
        background: {COLOR_SURFACE_PRIMARY};
        border: 1px solid {COLOR_STROKE_LIGHT};
        border-radius: 8px;
    }}
    QFrame#panelSeparator {{
        background: {COLOR_STROKE_LIGHT};
        max-width: 1px;
        min-width: 1px;
    }}
    QGraphicsView#esophaViewport {{
        background: {COLOR_CANVAS_BACKGROUND};
        border: 1px solid {COLOR_STROKE_LIGHT};
        border-radius: 8px;
    }}
    QLabel[numeric="true"] {{
        font-family: "Consolas", "SF Mono", "Fira Code", monospace;
    }}
    QPushButton {{
        background: {COLOR_SURFACE_PRIMARY};
        color: {COLOR_TEXT_EMPHASIZED};
        border: 1px solid {COLOR_STROKE_BOLD};
        border-radius: 6px;
        padding: 5px 12px;
        min-height: 22px;
    }}
    QPushButton:hover {{ background: {COLOR_SURFACE_DISABLED}; }}
    QPushButton:pressed {{ background: {COLOR_STROKE_LIGHT}; }}
    QPushButton:disabled {{
        color: {COLOR_TEXT_DISABLED};
        background: {COLOR_SURFACE_DISABLED};
        border-color: {COLOR_STROKE_LIGHT};
    }}
    QPushButton[role="primary"] {{
        background: {COLOR_ACCENT_BASE};
        color: {COLOR_TEXT_LIGHT};
        border: 1px solid {COLOR_ACCENT_PRESSED};
        font-weight: 600;
    }}
    QPushButton[role="primary"]:hover {{ background: {COLOR_ACCENT_HOVER}; }}
    QPushButton[role="primary"]:pressed {{ background: {COLOR_ACCENT_PRESSED}; }}

    QPushButton[role="critical"] {{
        background: {COLOR_ALERT_CRITICAL};
        color: {COLOR_TEXT_LIGHT};
        border: 1px solid #991B1B;
        font-weight: 600;
    }}
    QPushButton[role="critical"]:hover {{ background: {COLOR_ALERT_HOVER}; }}

    QPushButton#btnSidebarToggle {{
        background: #FFFFFF;
        border: 1px solid {COLOR_STROKE_LIGHT};
        border-radius: 12px;
        color: #64748B;
        font-size: 13pt;
        font-weight: bold;
        min-width: 22px;
        max-width: 22px;
        min-height: 54px;
        max-height: 54px;
        padding: 0px;
    }}
    QPushButton#btnSidebarToggle:hover {{
        background: #F8FAFC;
        color: {COLOR_ACCENT_BASE};
        border-color: {COLOR_ACCENT_HIGHLIGHT};
    }}
    QFrame#hudLayerFrame {{
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(203, 213, 225, 0.90);
        border-radius: 8px;
        padding: 6px 10px;
    }}
    QFrame#hudLayerFrame QLabel {{
        font-size: 8pt;
        font-weight: 700;
        color: {COLOR_TEXT_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    QFrame#hudLayerFrame QCheckBox {{
        font-size: 9pt;
        spacing: 5px;
    }}
    QCheckBox[overlayLayer="tissue"] {{ color: {COLOR_TISSUE_OVERLAY}; font-weight: 600; }}
    QCheckBox[overlayLayer="groundTruth"] {{ color: {COLOR_GROUND_TRUTH}; font-weight: 600; }}
    QCheckBox[overlayLayer="detection"] {{ color: {COLOR_PREDICTION_BOX}; font-weight: 600; }}
    QCheckBox[overlayLayer="hotspot"] {{ color: {COLOR_HOTSPOT_BOUNDARY}; font-weight: 600; }}

    QFrame#hudConfidenceFrame {{
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(203, 213, 225, 0.85);
        border-radius: 6px;
        padding: 4px 2px;
    }}
    QSlider#verticalConfidenceSlider::groove:vertical {{
        border: 1px solid {COLOR_STROKE_BOLD};
        width: 4px;
        background: {COLOR_SURFACE_DISABLED};
        border-radius: 2px;
    }}
    QSlider#verticalConfidenceSlider::sub-page:vertical {{
        background: {COLOR_SURFACE_DISABLED};
        border-radius: 2px;
    }}
    QSlider#verticalConfidenceSlider::add-page:vertical {{
        background: qlineargradient(x1:0, y1:1, x2:0, y2:0, stop:0 {COLOR_ACCENT_BASE}, stop:1 {COLOR_ACCENT_HIGHLIGHT});
        border-radius: 2px;
    }}
    QSlider#verticalConfidenceSlider::handle:vertical {{
        background: {COLOR_SURFACE_PRIMARY};
        border: 1.5px solid {COLOR_ACCENT_BASE};
        height: 10px;
        margin-left: -3px;
        margin-right: -3px;
        border-radius: 5px;
    }}
    QSlider#verticalConfidenceSlider::handle:vertical:hover {{
        background: {COLOR_ACCENT_HIGHLIGHT};
    }}
    QPushButton#btnCalibrateF1 {{
        background: #F1F5F9;
        color: {COLOR_ACCENT_BASE};
        border: 1px solid {COLOR_STROKE_BOLD};
        border-radius: 11px;
        min-width: 22px;
        max-width: 22px;
        min-height: 22px;
        max-height: 22px;
        font-size: 6.5pt;
        font-weight: 700;
        padding: 0px;
    }}
    QPushButton#btnCalibrateF1:hover {{
        background: {COLOR_ACCENT_BASE};
        color: #FFFFFF;
        border-color: {COLOR_ACCENT_PRESSED};
    }}

    QPushButton#btnHotspotFAB {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F59E0B, stop:1 #D97706);
        color: #FFFFFF;
        border: 1.5px solid #B45309;
        border-radius: 18px;
        min-width: 36px;
        max-width: 36px;
        min-height: 36px;
        max-height: 36px;
        font-weight: 700;
        font-size: 13pt;
        padding: 0px;
    }}
    QPushButton#btnHotspotFAB:hover {{
        background: #B45309;
        border-color: #92400E;
    }}
    QPushButton#btnHotspotFAB:pressed {{
        background: #92400E;
    }}
    QPushButton#btnHotspotFAB:disabled {{
        background: rgba(203, 213, 225, 0.7);
        color: {COLOR_TEXT_DISABLED};
        border-color: transparent;
    }}

    QFrame#esophaStatusPanel {{
        background: #FFFFFF;
        border: 1px solid {COLOR_STROKE_LIGHT};
        border-radius: 8px;
        padding: 2px;
    }}
    QLabel#hudStatePill {{
        font-size: 8pt;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 3px 9px;
        border-radius: 4px;
    }}
    QLabel#hudStatePill[state="ready"] {{
        background: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    QLabel#hudStatePill[state="executing"] {{
        background: rgba(37, 99, 235, 0.12);
        color: #2563EB;
        border: 1px solid rgba(37, 99, 235, 0.3);
    }}
    QLabel#hudStatePill[state="cancelling"] {{
        background: rgba(245, 158, 11, 0.12);
        color: #D97706;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }}
    QLabel#hudStatePill[state="completed"] {{
        background: rgba(16, 185, 129, 0.16);
        color: #047857;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }}
    QLabel#hudStatePill[state="cancelled"] {{
        background: rgba(100, 116, 139, 0.12);
        color: #475569;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }}
    QLabel#hudStatePill[state="error"] {{
        background: rgba(220, 38, 38, 0.12);
        color: #DC2626;
        border: 1px solid rgba(220, 38, 38, 0.3);
    }}

    QLabel#hudInfoPill {{
        background: #F8FAFC;
        color: #334155;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 8.5pt;
        font-weight: 600;
    }}
    QLabel#hudHardwarePill {{
        background: rgba(99, 102, 241, 0.08);
        color: #4F46E5;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 8.5pt;
        font-weight: 700;
    }}
    QLabel#hudHardwarePill[modelState="loading"] {{
        background: rgba(100, 116, 139, 0.10);
        color: #475569;
        border: 1px solid rgba(100, 116, 139, 0.30);
    }}
    QLabel#hudHardwarePill[modelState="warning"] {{
        background: rgba(245, 158, 11, 0.12);
        color: #B45309;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }}
    QLabel#hudHardwarePill[modelState="error"] {{
        background: rgba(220, 38, 38, 0.10);
        color: #DC2626;
        border: 1px solid rgba(220, 38, 38, 0.30);
    }}
    QLabel#hudStatePill[state="loading"] {{
        background: rgba(99, 102, 241, 0.10);
        color: #4F46E5;
        border: 1px solid rgba(99, 102, 241, 0.30);
    }}
    QLabel[calibration="fallback"] {{
        color: #B45309;
        font-weight: 700;
    }}
    QLabel#hudProgressDetail {{
        font-size: 8.5pt;
        font-weight: 600;
        color: #64748B;
        min-width: 140px;
        text-align: right;
    }}
    QProgressBar#hudProgressBar {{
        border: 1px solid #CBD5E1;
        border-radius: 4px;
        background: #F1F5F9;
        min-height: 8px;
        max-height: 8px;
    }}
    QProgressBar#hudProgressBar::chunk {{
        border-radius: 3px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563EB, stop:1 #38BDF8);
    }}
    QFrame#hudLogBox {{
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 5px;
    }}
    QLabel#hudLogPrefix {{
        color: #38BDF8;
        font-family: "Consolas", monospace;
        font-size: 7.5pt;
        font-weight: 700;
        padding: 1px 5px;
        background: rgba(56, 189, 248, 0.15);
        border-radius: 3px;
    }}
    QLabel#hudLogMessage {{
        color: #E2E8F0;
        font-family: "Segoe UI", sans-serif;
        font-size: 8.5pt;
    }}

    QFrame[sidebarCard="true"] {{
        background: #F8FAFC;
        border: 1px solid {COLOR_STROKE_LIGHT};
        border-radius: 6px;
        padding: 8px;
    }}
    QLabel[sidebarHeading="true"] {{
        font-size: 8.5pt;
        font-weight: 700;
        color: {COLOR_ACCENT_BASE};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    QLabel#diagnosticStatusBadge {{
        border-radius: 4px;
        padding: 2px 6px;
        font-weight: 700;
        font-size: 9pt;
        letter-spacing: 0.5px;
    }}
    QLabel#diagnosticStatusBadge[clinicalEvaluation="positive"] {{
        background: rgba(220, 38, 38, 0.12);
        color: {COLOR_ALERT_CRITICAL};
        border: 1px solid rgba(220, 38, 38, 0.3);
    }}
    QLabel#diagnosticStatusBadge[clinicalEvaluation="negative"] {{
        background: rgba(13, 148, 136, 0.12);
        color: {COLOR_STATUS_NEGATIVE};
        border: 1px solid rgba(13, 148, 136, 0.3);
    }}
    QLabel#diagnosticStatusBadge[clinicalEvaluation="none"] {{
        background: {COLOR_SURFACE_DISABLED};
        color: {COLOR_TEXT_SECONDARY};
        border: 1px solid {COLOR_STROKE_LIGHT};
    }}
    """


# ==============================================================================
# SECTION 1: DOMAIN ENTITIES & VALUE OBJECTS
# ==============================================================================
@dataclass(frozen=True)
class EsophaScopeConfig:
    model_weights_path: str = DEFAULT_MODEL_WEIGHTS
    confidence_threshold: float = DEFAULT_OPERATIONAL_CONFIDENCE
    nms_iou_threshold: float = SLIDE_NMS_IOU_THRESHOLD
    patch_overlap_ratio: float = PATCH_STRIDE_OVERLAP_RATIO
    batch_size: int = INFERENCE_BATCH_CAPACITY
    # Blue-chromaticity display mapping; None = stretch to the slide's own range.
    blue_chromaticity_min: float | None = None
    blue_chromaticity_max: float | None = None
    blue_chromaticity_gamma: float = 1.0
    tissue_min_coverage: float = TISSUE_COVERAGE_MINIMUM_FRACTION
    centroid_tolerance_um: float = CENTROID_MATCH_TOLERANCE_UM
    thumbnail_max_dimension: int = OVERVIEW_THUMBNAIL_MAX_DIMENSION


@dataclass(frozen=True)
class EosinophilCell:
    cell_id: int
    confidence: float
    bbox_coordinates: tuple[float, float, float, float]
    cell_class: str = "eosinophil"

    @property
    def centroid_px(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_coordinates
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass(frozen=True)
class GroundTruthCell:
    feature_id: str
    bbox_coordinates: tuple[float, float, float, float]
    centroid_px: tuple[float, float]
    boundary_points: list[tuple[float, float]] = field(default_factory=list)
    cell_class: str = "eosinophil"


@dataclass(frozen=True)
class DiagnosticFieldSummary:
    peak_count: int
    is_eoe_positive: bool
    center_px: tuple[float, float]
    radius_px: float
    bounding_box_px: tuple[float, float, float, float]
    density_per_mm2: float


@dataclass(frozen=True)
class SlideSpecification:
    slide_path: str
    file_name: str
    width_px: int
    height_px: int
    mpp_x: float
    mpp_y: float
    scanner_model: str
    magnification: str
    level_count: int
    level_dimensions: list[tuple[int, int]]
    level_downsamples: list[float]
    tissue_surface_area_mm2: float = 0.0
    mpp_is_fallback: bool = False
    calibration_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationSummary:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    f2_score: float
    mean_centroid_error_um: float


# ==============================================================================
# SECTION 2: PATHOLOGY PROCESSING & SPATIAL ALGORITHMS
# ==============================================================================
def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_thresh: float
) -> list[int]:
    """Greedy NMS returning kept indices, highest score first (pure NumPy).

    Boxes are visited in descending score order and only compared against
    boxes whose x-range can overlap (found via a sorted x1 index), which keeps
    slide-wide merges near-linear instead of quadratic.
    """
    n = len(boxes)
    if n == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    by_x1 = np.argsort(x1)
    sorted_x1 = x1[by_x1]
    max_width = float(np.max(x2 - x1))
    suppressed = np.zeros(n, dtype=bool)
    kept: list[int] = []

    for i in np.argsort(-scores, kind="stable"):
        if suppressed[i]:
            continue
        kept.append(int(i))

        lo = np.searchsorted(sorted_x1, x1[i] - max_width, side="left")
        hi = np.searchsorted(sorted_x1, x2[i], side="right")
        cand = by_x1[lo:hi]
        cand = cand[(cand != i) & ~suppressed[cand]]
        if cand.size == 0:
            continue

        inter_w = np.maximum(
            0.0, np.minimum(x2[i], x2[cand]) - np.maximum(x1[i], x1[cand])
        )
        inter_h = np.maximum(
            0.0, np.minimum(y2[i], y2[cand]) - np.maximum(y1[i], y1[cand])
        )
        inter = inter_w * inter_h
        iou = inter / (areas[i] + areas[cand] - inter + 1e-7)
        suppressed[cand[iou > iou_thresh]] = True

    return kept


def blue_chromaticity(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blue chromaticity B / (R + G + B) per pixel, as in QuPath.

    Returns (chromaticity float32, valid bool). Pixels with R + G + B == 0 have no
    defined chromaticity; they are marked invalid and given 0.
    """
    channels = rgb[:, :, :3].astype(np.float32)
    total = channels.sum(axis=2)
    valid = total > 0.0
    chroma = np.zeros(total.shape, dtype=np.float32)
    np.divide(channels[:, :, 2], total, out=chroma, where=valid)
    return chroma, valid


def apply_display_adjustment(
    values: np.ndarray, channel_min: float, channel_max: float, gamma: float
) -> np.ndarray:
    """QuPath display mapping: clip((I - min) / (max - min), 0, 1) ** (1 / gamma)."""
    if not channel_max > channel_min:
        raise ValueError("Channel max must be greater than channel min.")
    if not gamma > 0.0:
        raise ValueError("Gamma must be positive.")
    normalised = np.clip((values - channel_min) / (channel_max - channel_min), 0.0, 1.0)
    return normalised ** (1.0 / gamma)


def otsu_threshold(values_u8: np.ndarray) -> int:
    """Otsu's threshold on 8-bit values: maximises between-class variance."""
    histogram = np.bincount(values_u8.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return 0
    levels = np.arange(256, dtype=np.float64)
    weight_low = np.cumsum(histogram)
    weight_high = total - weight_low
    mean_cumulative = np.cumsum(histogram * levels)
    mean_total = mean_cumulative[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_low = mean_cumulative / weight_low
        mean_high = (mean_total - mean_cumulative) / weight_high
        variance = weight_low * weight_high * (mean_low - mean_high) ** 2
    variance[~np.isfinite(variance)] = -1.0
    return int(np.argmax(variance))


def generate_tissue_mask_blue_chromaticity(
    thumbnail_rgb: np.ndarray,
    channel_min: float | None = None,
    channel_max: float | None = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Binary tissue mask (0/255): blue chromaticity -> display mapping -> Otsu.

    Channel min/max default to the 0.5th/99.5th percentiles of the slide's own
    chromaticity (contrast stretch). Otsu splits the adjusted image into two
    classes; tissue is the class with the lower mean luminance (darker than the
    glass background). Returns an empty mask for a flat image.
    """
    chroma, valid = blue_chromaticity(thumbnail_rgb)
    empty = np.zeros(chroma.shape, dtype=np.uint8)
    if not valid.any():
        return empty
    observed = chroma[valid]
    low = channel_min
    high = channel_max
    if low is None or high is None:
        auto_low, auto_high = np.percentile(observed, (0.5, 99.5))
        low = float(auto_low) if low is None else low
        high = float(auto_high) if high is None else high
    if not high > low:
        return empty

    adjusted = apply_display_adjustment(chroma, low, high, gamma)
    levels = np.rint(adjusted * 255.0).astype(np.uint8)
    threshold = otsu_threshold(levels[valid])
    upper = valid & (levels > threshold)
    lower = valid & ~upper
    if not upper.any() or not lower.any():
        return empty

    luminance = thumbnail_rgb[:, :, :3].astype(np.float32) @ np.array(
        [0.299, 0.587, 0.114], np.float32
    )
    tissue = upper if luminance[upper].mean() < luminance[lower].mean() else lower
    binary_tissue = tissue.astype(np.uint8) * 255
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(
        binary_tissue, cv2.MORPH_CLOSE, structuring_element, iterations=2
    )


def build_tissue_overlay_rgba(mask_binary: np.ndarray) -> np.ndarray:
    """Creates a transparent RGBA overlay array for the tissue layer."""
    h, w = mask_binary.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_binary > 0] = (37, 99, 235, 75)
    return rgba


def identify_peak_diagnostic_hpf(
    cells: list[EosinophilCell], mpp_x: float, mpp_y: float
) -> DiagnosticFieldSummary | None:
    """Locates the peak High-Power Field (0.238 mm²) via physical-space KDTree query."""
    if not cells:
        return None

    physical_um_pts = np.array(
        [[cell.centroid_px[0] * mpp_x, cell.centroid_px[1] * mpp_y] for cell in cells],
        dtype=np.float32,
    )
    spatial_kdtree = KDTree(physical_um_pts)
    neighbors_in_radius = spatial_kdtree.query_ball_point(
        physical_um_pts, r=CLINICAL_HPF_RADIUS_UM
    )
    densities = [len(n) for n in neighbors_in_radius]
    peak_idx = int(np.argmax(densities))
    peak_cell_count = densities[peak_idx]

    optimal_center_um = physical_um_pts[peak_idx]
    center_px = (
        float(optimal_center_um[0] / mpp_x),
        float(optimal_center_um[1] / mpp_y),
    )
    rad_px_x = CLINICAL_HPF_RADIUS_UM / mpp_x
    rad_px_y = CLINICAL_HPF_RADIUS_UM / mpp_y
    mean_radius_px = (rad_px_x + rad_px_y) * 0.5

    return DiagnosticFieldSummary(
        peak_count=peak_cell_count,
        is_eoe_positive=(peak_cell_count >= EOE_DIAGNOSTIC_HOTSPOT_THRESHOLD),
        center_px=center_px,
        radius_px=mean_radius_px,
        bounding_box_px=(
            center_px[0] - rad_px_x,
            center_px[1] - rad_px_y,
            center_px[0] + rad_px_x,
            center_px[1] + rad_px_y,
        ),
        density_per_mm2=round(peak_cell_count / CLINICAL_HPF_AREA_MM2, 1),
    )


def compute_cellular_performance_metrics(
    ground_truth: list[GroundTruthCell],
    predictions: list[EosinophilCell],
    mpp_x: float,
    mpp_y: float,
    match_distance_um: float = CENTROID_MATCH_TOLERANCE_UM,
) -> EvaluationSummary:
    """Evaluates Precision, Recall, F1, and mean centroid error against ground truth."""
    gt_len = len(ground_truth)
    pred_len = len(predictions)

    if gt_len == 0 or pred_len == 0:
        return EvaluationSummary(
            true_positives=0,
            false_positives=pred_len,
            false_negatives=gt_len,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            f2_score=0.0,
            mean_centroid_error_um=0.0,
        )

    gt_points_um = np.array(
        [[g.centroid_px[0] * mpp_x, g.centroid_px[1] * mpp_y] for g in ground_truth],
        dtype=np.float32,
    )
    pred_points_um = np.array(
        [[p.centroid_px[0] * mpp_x, p.centroid_px[1] * mpp_y] for p in predictions],
        dtype=np.float32,
    )
    confidence_levels = np.array([p.confidence for p in predictions], dtype=np.float32)

    gt_kdtree = KDTree(gt_points_um)
    ranked_indices = np.argsort(-confidence_levels)
    claimed_gt_indices: set[int] = set()
    claimed_pred_indices: set[int] = set()
    centroid_displacements: list[float] = []

    for pred_idx in ranked_indices:
        probe_point = pred_points_um[pred_idx]
        distances, candidate_indices = gt_kdtree.query(probe_point, k=min(10, gt_len))
        dist_list = [float(d) for d in np.atleast_1d(distances)]
        idx_list = [int(i) for i in np.atleast_1d(candidate_indices)]

        for dist_val, gt_idx in zip(dist_list, idx_list, strict=False):
            if dist_val <= match_distance_um and gt_idx not in claimed_gt_indices:
                claimed_gt_indices.add(gt_idx)
                claimed_pred_indices.add(int(pred_idx))
                centroid_displacements.append(dist_val)
                break

    tp = len(claimed_gt_indices)
    fp = pred_len - len(claimed_pred_indices)
    fn = gt_len - len(claimed_gt_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    f2 = (
        (5.0 * precision * recall) / (4.0 * precision + recall)
        if (4.0 * precision + recall) > 0
        else 0.0
    )
    mean_err = float(np.mean(centroid_displacements)) if centroid_displacements else 0.0

    return EvaluationSummary(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        f2_score=f2,
        mean_centroid_error_um=mean_err,
    )


def parse_qupath_geojson(geojson_path: str | Path) -> list[GroundTruthCell]:
    """Loads and standardizes QuPath GeoJSON annotations."""
    with open(geojson_path, encoding="utf-8") as f:
        payload = json.load(f)

    raw_features = payload.get("features", []) if isinstance(payload, dict) else payload
    annotations: list[GroundTruthCell] = []

    for idx, feat in enumerate(raw_features):
        if not isinstance(feat, dict):
            continue

        geometry = feat.get("geometry", {})
        coords = geometry.get("coordinates", [])
        feature_id = str(feat.get("id") or f"qupath_cell_{idx}")
        props = feat.get("properties") or {}

        classification_name = "eosinophil"
        if isinstance(props, dict):
            cls_obj = props.get("classification")
            if isinstance(cls_obj, dict):
                classification_name = str(cls_obj.get("name", "eosinophil"))
            else:
                classification_name = str(props.get("objectType", "eosinophil"))

        extracted_polygons = []
        if geometry.get("type") == "Polygon" and coords:
            extracted_polygons.append(coords[0])
        elif geometry.get("type") == "MultiPolygon" and coords:
            for poly in coords:
                if poly:
                    extracted_polygons.append(poly[0])

        for sub_idx, ring in enumerate(extracted_polygons):
            points = [(float(p[0]), float(p[1])) for p in ring if len(p) >= 2]
            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            annotations.append(
                GroundTruthCell(
                    feature_id=f"{feature_id}_{sub_idx}"
                    if len(extracted_polygons) > 1
                    else feature_id,
                    bbox_coordinates=(min(xs), min(ys), max(xs), max(ys)),
                    centroid_px=(float(np.mean(xs)), float(np.mean(ys))),
                    boundary_points=points,
                    cell_class=classification_name,
                )
            )

    return annotations


def serialize_qupath_geojson(
    export_destination: str | Path,
    detected_cells: list[EosinophilCell],
    diagnostic_field: DiagnosticFieldSummary | None,
) -> None:
    """Exports detections and diagnostic peak HPF field to QuPath GeoJSON."""
    features = []
    for cell in detected_cells:
        x1, y1, x2, y2 = cell.bbox_coordinates
        polygon_ring = [
            [round(x1, 2), round(y1, 2)],
            [round(x2, 2), round(y1, 2)],
            [round(x2, 2), round(y2, 2)],
            [round(x1, 2), round(y2, 2)],
            [round(x1, 2), round(y1, 2)],
        ]
        features.append(
            {
                "type": "Feature",
                "id": f"esopha_cell_{cell.cell_id}",
                "geometry": {"type": "Polygon", "coordinates": [polygon_ring]},
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": cell.cell_class, "colorRGB": -65536},
                    "isLocked": False,
                    "measurements": [
                        {
                            "name": "Detection Confidence",
                            "value": round(cell.confidence, 4),
                        }
                    ],
                },
            }
        )

    if diagnostic_field is not None:
        cx, cy = diagnostic_field.center_px
        r = diagnostic_field.radius_px
        angles = np.linspace(0, 2 * math.pi, 36)
        circular_boundary = [
            [round(cx + r * math.cos(theta), 2), round(cy + r * math.sin(theta), 2)]
            for theta in angles
        ]
        circular_boundary.append(circular_boundary[0])

        features.append(
            {
                "type": "Feature",
                "id": "esopha_diagnostic_peak_hpf",
                "geometry": {"type": "Polygon", "coordinates": [circular_boundary]},
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": f"Peak HPF ({diagnostic_field.peak_count} eos/HPF)",
                        "colorRGB": -16711936
                        if diagnostic_field.is_eoe_positive
                        else -256,
                    },
                    "isLocked": True,
                    "measurements": [
                        {
                            "name": "Peak Eosinophil Count",
                            "value": diagnostic_field.peak_count,
                        },
                        {
                            "name": "Diagnostic Density (cells/mm²)",
                            "value": diagnostic_field.density_per_mm2,
                        },
                    ],
                },
            }
        )

    with open(export_destination, "w", encoding="utf-8") as out_stream:
        json.dump(
            {"type": "FeatureCollection", "features": features}, out_stream, indent=2
        )


# ==============================================================================
# SECTION 2B: ASYNCHRONOUS SLIDE LOADING
# ==============================================================================
def _parse_mpp(raw: object) -> float | None:
    """Parses one microns-per-pixel property; None when missing or unusable."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def resolve_slide_calibration(
    raw_x: object, raw_y: object
) -> tuple[float, float, bool, tuple[str, ...]]:
    """Returns (mpp_x, mpp_y, used_fallback, notes) from raw OpenSlide MPP values.

    Missing/invalid calibration is never hidden: whenever a value is substituted
    (the other axis, or TARGET_MODEL_MPP when both are unusable) the flag is set
    and a human-readable note is returned so the UI can report it.
    """
    mpp_x, mpp_y = _parse_mpp(raw_x), _parse_mpp(raw_y)
    notes: list[str] = []
    used_fallback = False

    if mpp_x is None and mpp_y is None:
        notes.append(
            f"No valid MPP metadata (X={raw_x!r}, Y={raw_y!r}). Assuming "
            f"{TARGET_MODEL_MPP} µm/px on both axes; physical measurements "
            "(HPF area, density, tolerances) are NOT calibrated."
        )
        return TARGET_MODEL_MPP, TARGET_MODEL_MPP, True, tuple(notes)

    if mpp_x is None:
        notes.append(f"X MPP missing/invalid ({raw_x!r}); using the Y value {mpp_y}.")
        mpp_x, used_fallback = mpp_y, True
    elif mpp_y is None:
        notes.append(f"Y MPP missing/invalid ({raw_y!r}); using the X value {mpp_x}.")
        mpp_y, used_fallback = mpp_x, True

    low, high = PLAUSIBLE_MPP_RANGE_UM
    for axis, value in (("X", mpp_x), ("Y", mpp_y)):
        if not low <= value <= high:
            notes.append(
                f"{axis} MPP {value} µm/px is outside the usual range "
                f"{low}-{high}; verify the slide calibration."
            )
    return float(mpp_x), float(mpp_y), used_fallback, tuple(notes)


@dataclass(frozen=True)
class SlideLoadResult:
    """Everything the GUI thread needs to show a freshly opened slide."""

    spec: SlideSpecification
    thumbnail_rgb: np.ndarray
    tissue_mask: np.ndarray
    tissue_overlay_rgba: np.ndarray


class SlideLoadSignals(QObject):
    status = Signal(int, str)
    loaded = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal(int)


class SlideLoadTask(QRunnable):
    """Opens a slide, reads metadata, builds the overview and tissue mask off-thread.

    Only plain Python/NumPy data crosses back to the GUI thread; QPixmap and
    scene items are created there. The OpenSlide handle is closed in every path.
    """

    def __init__(
        self,
        request_id: int,
        slide_path: str,
        config: EsophaScopeConfig,
        signals: SlideLoadSignals,
    ) -> None:
        super().__init__()
        # The window keeps a Python reference until `finished` arrives.
        self.setAutoDelete(False)
        self._request_id = request_id
        self._slide_path = slide_path
        self._config = config
        self._signals = signals
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        rid = self._request_id
        try:
            result = self._load()
            if result is not None and not self._cancel.is_set():
                self._signals.loaded.emit(rid, result)
        except (
            openslide.OpenSlideError,
            OSError,
            ValueError,
            RuntimeError,
            MemoryError,
            cv2.error,
        ) as exc:
            logger.exception("Failed to load slide %s", self._slide_path)
            self._signals.failed.emit(rid, str(exc) or type(exc).__name__)
        except Exception as exc:  # last resort: the GUI must always leave LOADING
            logger.exception("Unexpected slide-loading failure")
            self._signals.failed.emit(rid, f"{type(exc).__name__}: {exc}")
        finally:
            self._signals.finished.emit(rid)

    def _load(self) -> SlideLoadResult | None:
        rid = self._request_id
        path = Path(self._slide_path)
        if not path.is_file():
            raise FileNotFoundError(f"Slide not found: {path}")

        self._signals.status.emit(rid, f"Opening {path.name}...")
        with openslide.OpenSlide(str(path)) as slide:
            width, height = slide.dimensions
            props = slide.properties
            mpp_x, mpp_y, mpp_fallback, notes = resolve_slide_calibration(
                props.get(openslide.PROPERTY_NAME_MPP_X),
                props.get(openslide.PROPERTY_NAME_MPP_Y),
            )
            vendor = str(props.get(openslide.PROPERTY_NAME_VENDOR, "Generic Scanner"))
            magnification = str(props.get("openslide.objective-power", "n/a"))
            level_count = slide.level_count
            level_dimensions = [tuple(d) for d in slide.level_dimensions]
            level_downsamples = [float(d) for d in slide.level_downsamples]

            if self._cancel.is_set():
                return None
            self._signals.status.emit(rid, "Generating overview thumbnail...")
            scale = min(
                self._config.thumbnail_max_dimension / float(width),
                self._config.thumbnail_max_dimension / float(height),
                1.0,
            )
            request_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            thumbnail = np.ascontiguousarray(
                np.asarray(slide.get_thumbnail(request_size).convert("RGB"))
            )

        if self._cancel.is_set():
            return None
        self._signals.status.emit(rid, "Computing tissue mask...")
        # PIL may round the thumbnail by a pixel; always use the real array shape.
        thumb_h, thumb_w = thumbnail.shape[:2]
        tissue_mask = generate_tissue_mask_blue_chromaticity(
            thumbnail,
            self._config.blue_chromaticity_min,
            self._config.blue_chromaticity_max,
            self._config.blue_chromaticity_gamma,
        )
        tissue_pixels = int(np.count_nonzero(tissue_mask))
        tissue_area_mm2 = (
            tissue_pixels
            * ((width / thumb_w) * mpp_x)
            * ((height / thumb_h) * mpp_y)
            / 1_000_000.0
        )
        spec = SlideSpecification(
            slide_path=str(path),
            file_name=path.name,
            width_px=width,
            height_px=height,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            scanner_model=vendor,
            magnification=magnification,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            tissue_surface_area_mm2=tissue_area_mm2,
            mpp_is_fallback=mpp_fallback,
            calibration_notes=notes,
        )
        return SlideLoadResult(
            spec=spec,
            thumbnail_rgb=thumbnail,
            tissue_mask=tissue_mask,
            tissue_overlay_rgba=build_tissue_overlay_rgba(tissue_mask),
        )


# ==============================================================================
# SECTION 3: SLIDE HANDLE POOL, TILE STREAMING & PYRAMIDAL CANVAS GRAPHICS
# ==============================================================================
TileKey = tuple[int, int, int]  # (pyramid level, tile column, tile row)


class SlideClosedError(RuntimeError):
    """Raised when a handle is requested after its pool was shut down."""


class SlideHandlePool:
    """Bounded pool of OpenSlide handles shared by worker threads.

    A handle serves one read at a time and handles are created lazily, so open
    handles/file descriptors never exceed ``capacity`` even though QThreadPool
    recycles its threads. ``shutdown()`` never closes a borrowed handle: the last
    borrower to return closes everything.
    """

    def __init__(self, file_path: str, capacity: int) -> None:
        self._path = file_path
        self._capacity = max(1, capacity)
        self._cond = threading.Condition()
        self._idle: list[openslide.OpenSlide] = []
        self._all: list[openslide.OpenSlide] = []
        self._created = 0
        self._borrowed = 0
        self._closed = False

    @contextlib.contextmanager
    def borrow(self) -> Iterator[openslide.OpenSlide]:
        handle = self._acquire()
        try:
            yield handle
        finally:
            self._release(handle)

    def _acquire(self) -> openslide.OpenSlide:
        with self._cond:
            while True:
                if self._closed:
                    raise SlideClosedError(self._path)
                if self._idle:
                    self._borrowed += 1
                    return self._idle.pop()
                if self._created < self._capacity:
                    self._created += 1
                    self._borrowed += 1
                    break
                self._cond.wait()
        try:
            handle = openslide.OpenSlide(self._path)
        except BaseException:
            with self._cond:
                self._created -= 1
            self._release(None)
            raise
        with self._cond:
            self._all.append(handle)
        return handle

    def _release(self, handle: openslide.OpenSlide | None) -> None:
        with self._cond:
            self._borrowed -= 1
            if handle is not None and not self._closed:
                self._idle.append(handle)
            self._cond.notify()
            close_now = self._closed and self._borrowed == 0
        if close_now:
            self._close_all()

    def shutdown(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            close_now = self._borrowed == 0
        if close_now:
            self._close_all()

    def _close_all(self) -> None:
        with self._cond:
            handles, self._all = self._all, []
            self._idle.clear()
        for handle in handles:
            with contextlib.suppress(openslide.OpenSlideError, OSError):
                handle.close()


def select_pyramid_level(downsamples: Sequence[float], screen_downsample: float) -> int:
    """Coarsest level whose downsample does not exceed the on-screen downsample.

    ``screen_downsample`` is level-0 pixels per device pixel (1 / effective zoom),
    so the chosen level always has at least screen resolution.
    """
    level = 0
    for index, ds in enumerate(downsamples):
        if ds <= screen_downsample * (1.0 + 1e-6):
            level = index
        else:
            break
    return level


class PyramidalTileSignals(QObject):
    tile_ready = Signal(int, int, int, QImage)
    tile_failed = Signal(int, int, int)


class PyramidalTileFetchTask(QRunnable):
    """Reads one pyramid tile and converts it to a paint-ready QImage off-thread."""

    def __init__(
        self,
        handles: SlideHandlePool,
        key: TileKey,
        origin_level0: tuple[int, int],
        size: tuple[int, int],
        signals: PyramidalTileSignals,
        closed: threading.Event,
    ) -> None:
        super().__init__()
        self._handles = handles
        self._key = key
        self._origin = origin_level0
        self._size = size
        self._signals = signals
        self._closed = closed

    def run(self) -> None:
        if self._closed.is_set():
            return
        level, col, row = self._key
        width, height = self._size
        try:
            with self._handles.borrow() as slide:
                tile = slide.read_region(self._origin, level, self._size)
            raw = tile.tobytes()
            tile.close()
            # convertToFormat deep-copies, so `raw` may be released afterwards.
            image = QImage(
                raw, width, height, 4 * width, QImage.Format.Format_RGBA8888
            ).convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        except SlideClosedError:
            return
        except (openslide.OpenSlideError, OSError, RuntimeError, ValueError) as exc:
            logger.debug("Tile fault at level %d (%d, %d): %s", level, col, row, exc)
            if not self._closed.is_set():
                self._signals.tile_failed.emit(level, col, row)
            return
        if not self._closed.is_set():
            self._signals.tile_ready.emit(level, col, row, image)


class PyramidalSlideCanvas(QGraphicsItem):
    """Multi-resolution slide item: overview backdrop + viewport-driven tiles.

    Coordinates: the scene is level-0 pixels. Tile (col, row) at level L covers
    level-L pixels [col*T, col*T+w) x [row*T, row*T+h) with T = tile edge, which is
    scene rect (col*T*ds, row*T*ds, w*ds, h*ds) and is read from level-0 origin
    round(col*T*ds), round(row*T*ds) - the same ``ds`` OpenSlide itself uses.
    """

    def __init__(
        self,
        spec: SlideSpecification,
        overview_pixmap: QPixmap,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        # Without this flag Qt reports the *whole* item as exposed on every paint.
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)
        self.setZValue(0.0)

        self._width = spec.width_px
        self._height = spec.height_px
        self._level_dims = [tuple(d) for d in spec.level_dimensions]
        self._downsamples = list(spec.level_downsamples)
        self._overview = overview_pixmap
        self._overview_ds = max(
            self._width / float(max(1, overview_pixmap.width())),
            self._height / float(max(1, overview_pixmap.height())),
        )

        workers = tile_worker_count()
        self._handles = SlideHandlePool(spec.slide_path, workers)
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(workers)
        self._max_in_flight = workers + 2

        self._cache: OrderedDict[TileKey, QImage] = OrderedDict()
        self._cache_bytes = 0
        self._cache_budget = tile_cache_budget_bytes()
        self._in_flight: set[TileKey] = set()
        self._wanted: deque[TileKey] = deque()
        self._failures: dict[TileKey, int] = {}
        self._hits = 0
        self._misses = 0
        self._closed = threading.Event()

        self.signals = PyramidalTileSignals()
        self.signals.tile_ready.connect(
            self._on_tile_ready, Qt.ConnectionType.QueuedConnection
        )
        self.signals.tile_failed.connect(
            self._on_tile_failed, Qt.ConnectionType.QueuedConnection
        )

    # ---- geometry -----------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    def tile_geometry(
        self, level: int, col: int, row: int
    ) -> tuple[tuple[int, int], tuple[int, int], QRectF]:
        """(level-0 read origin, level-pixel size, scene rect) of a tile."""
        ds = self._downsamples[level]
        level_w, level_h = self._level_dims[level]
        lx = col * PYRAMIDAL_TILE_EDGE_PX
        ly = row * PYRAMIDAL_TILE_EDGE_PX
        w = min(PYRAMIDAL_TILE_EDGE_PX, level_w - lx)
        h = min(PYRAMIDAL_TILE_EDGE_PX, level_h - ly)
        return (
            (round(lx * ds), round(ly * ds)),
            (w, h),
            QRectF(lx * ds, ly * ds, w * ds, h * ds),
        )

    def tiles_for_rect(self, level: int, rect: QRectF) -> list[tuple[int, int]]:
        """(col, row) of every level tile intersecting a level-0 scene rect."""
        if rect.width() <= 0 or rect.height() <= 0:
            return []
        ds = self._downsamples[level]
        level_w, level_h = self._level_dims[level]
        edge = PYRAMIDAL_TILE_EDGE_PX
        cols = math.ceil(level_w / edge)
        rows = math.ceil(level_h / edge)
        c0 = max(0, math.floor(rect.left() / ds / edge))
        c1 = min(cols - 1, math.ceil(rect.right() / ds / edge) - 1)
        r0 = max(0, math.floor(rect.top() / ds / edge))
        r1 = min(rows - 1, math.ceil(rect.bottom() / ds / edge) - 1)
        return [(c, r) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def cache_stats(self) -> dict[str, float]:
        lookups = self._hits + self._misses
        return {
            "tiles": len(self._cache),
            "bytes": self._cache_bytes,
            "budget": self._cache_budget,
            "hit_rate": (self._hits / lookups) if lookups else 0.0,
        }

    # ---- lifecycle ----------------------------------------------------------
    def close(self, wait: bool = False) -> None:
        """Stops streaming. Handles close as soon as no worker is using them."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._wanted.clear()
        self._in_flight.clear()
        self._pool.clear()
        if wait:
            self._pool.waitForDone(3000)
        self._handles.shutdown()
        logger.debug("Tile cache at close: %s", self.cache_stats())
        self._cache.clear()
        self._cache_bytes = 0

    # ---- tile arrival -------------------------------------------------------
    @Slot(int, int, int, QImage)
    def _on_tile_ready(self, level: int, col: int, row: int, image: QImage) -> None:
        if self._closed.is_set():
            return
        key = (level, col, row)
        self._in_flight.discard(key)
        self._failures.pop(key, None)
        previous = self._cache.pop(key, None)
        if previous is not None:
            self._cache_bytes -= previous.sizeInBytes()
        self._cache[key] = image
        self._cache_bytes += image.sizeInBytes()
        while self._cache_bytes > self._cache_budget and len(self._cache) > 1:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.sizeInBytes()
        self.update(self.tile_geometry(level, col, row)[2])
        self._pump()

    @Slot(int, int, int)
    def _on_tile_failed(self, level: int, col: int, row: int) -> None:
        key = (level, col, row)
        self._in_flight.discard(key)
        self._failures[key] = self._failures.get(key, 0) + 1
        self._pump()

    def _pump(self) -> None:
        """Starts wanted tiles, nearest-first, never exceeding the in-flight cap."""
        while (
            self._wanted
            and len(self._in_flight) < self._max_in_flight
            and not self._closed.is_set()
        ):
            key = self._wanted.popleft()
            if key in self._in_flight or key in self._cache:
                continue
            if self._failures.get(key, 0) > TILE_FETCH_MAX_RETRIES:
                continue
            level, col, row = key
            origin, size, _ = self.tile_geometry(level, col, row)
            self._in_flight.add(key)
            self._pool.start(
                PyramidalTileFetchTask(
                    self._handles, key, origin, size, self.signals, self._closed
                )
            )

    # ---- painting -----------------------------------------------------------
    def _owning_view(self, widget: QWidget | None) -> QGraphicsView | None:
        scene = self.scene()
        views = scene.views() if scene is not None else []
        for view in views:
            if widget is not None and view.viewport() is widget:
                return view
        return views[0] if views else None

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        bounds = self.boundingRect()
        exposed = option.exposedRect.intersected(bounds)
        if exposed.isEmpty() or self._closed.is_set():
            return
        painter.save()
        painter.setClipRect(bounds, Qt.ClipOperation.IntersectClip)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self._paint_overview(painter, exposed)
        self._paint_tiles(painter, exposed, widget)
        painter.restore()

    def _paint_overview(self, painter: QPainter, exposed: QRectF) -> None:
        scale_x = self._overview.width() / float(self._width)
        scale_y = self._overview.height() / float(self._height)
        source = QRectF(
            exposed.left() * scale_x,
            exposed.top() * scale_y,
            exposed.width() * scale_x,
            exposed.height() * scale_y,
        )
        painter.drawPixmap(exposed, self._overview, source)

    def _paint_tiles(
        self, painter: QPainter, exposed: QRectF, widget: QWidget | None
    ) -> None:
        view = self._owning_view(widget)
        dpr = view.devicePixelRatioF() if view is not None else 1.0
        zoom = QStyleOptionGraphicsItem.levelOfDetailFromTransform(
            painter.worldTransform()
        )
        if zoom <= 0.0:
            return
        screen_ds = 1.0 / (zoom * dpr)
        if screen_ds >= self._overview_ds:
            self._wanted.clear()  # the overview already has enough resolution
            return

        level = select_pyramid_level(self._downsamples, screen_ds)
        ds = self._downsamples[level]
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform, zoom * dpr * ds < 1.0
        )

        # Fetch for the whole viewport, not just the (possibly partial) exposed rect.
        if view is not None:
            view_rect = view.mapToScene(view.viewport().rect()).boundingRect()
            view_rect = view_rect.intersected(self.boundingRect())
        else:
            view_rect = exposed
        centre = view_rect.center()

        missing: list[tuple[float, TileKey]] = []
        for col, row in self.tiles_for_rect(level, view_rect):
            key = (level, col, row)
            _, _, target = self.tile_geometry(level, col, row)
            image = self._cache.get(key)
            if image is not None:
                self._hits += 1
                self._cache.move_to_end(key)
                if target.intersects(exposed):
                    painter.drawImage(target, image)
                continue
            self._misses += 1
            delta = target.center() - centre
            missing.append((delta.x() ** 2 + delta.y() ** 2, key))
            if target.intersects(exposed):
                self._paint_coarse_fallback(painter, level, target)

        missing.sort(key=lambda item: item[0])
        self._wanted = deque(key for _, key in missing)
        self._pump()

    def _paint_coarse_fallback(
        self, painter: QPainter, level: int, target: QRectF
    ) -> None:
        """Shows already-cached coarser tiles while the sharp tile is loading."""
        last_level = len(self._downsamples) - 1
        for coarse in range(min(level + 2, last_level), level, -1):
            coarse_ds = self._downsamples[coarse]
            for col, row in self.tiles_for_rect(coarse, target):
                image = self._cache.get((coarse, col, row))
                if image is None:
                    continue
                coarse_rect = self.tile_geometry(coarse, col, row)[2]
                overlap = coarse_rect.intersected(target)
                if overlap.isEmpty():
                    continue
                source = QRectF(
                    (overlap.left() - coarse_rect.left()) / coarse_ds,
                    (overlap.top() - coarse_rect.top()) / coarse_ds,
                    overlap.width() / coarse_ds,
                    overlap.height() / coarse_ds,
                )
                painter.drawImage(overlap, image, source)


# ==============================================================================
# SECTION 4: VECTORIZED VIEWPORT OVERLAYS
# ==============================================================================
class FastVectorOverlay(QGraphicsItem):
    """Vectorized overlay item using NumPy coordinate filtering and batched painting."""

    def __init__(
        self, color_hex: str, z_index: float, parent: QGraphicsItem | None = None
    ):
        super().__init__(parent)
        # Needed so Qt reports only the exposed part of the item, not all of it.
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)
        self.setZValue(z_index)
        self._pen = QPen(QColor(color_hex), 2)
        self._pen.setCosmetic(True)

        self._boxes = np.empty((0, 4), dtype=np.float32)
        self._rects: list[QRectF] = []
        self._polygons: list[QPolygonF] = []
        self._bounds = QRectF()

    def set_vector_data(
        self,
        boxes: list[tuple[float, float, float, float]],
        polygons: list[list[tuple[float, float]]] | None = None,
    ) -> None:
        self.prepareGeometryChange()
        self._polygons = [
            QPolygonF([QPointF(x, y) for x, y in poly]) for poly in (polygons or [])
        ]

        unified_rect = QRectF()

        if boxes:
            self._boxes = np.array(boxes, dtype=np.float32)
            self._rects = [QRectF(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
            min_x = float(np.min(self._boxes[:, 0]))
            min_y = float(np.min(self._boxes[:, 1]))
            max_x = float(np.max(self._boxes[:, 2]))
            max_y = float(np.max(self._boxes[:, 3]))
            unified_rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
        else:
            self._boxes = np.empty((0, 4), dtype=np.float32)
            self._rects = []

        for poly in self._polygons:
            poly_rect = poly.boundingRect()
            unified_rect = (
                poly_rect if unified_rect.isEmpty() else unified_rect.united(poly_rect)
            )

        self._bounds = unified_rect
        self.update()

    def clear(self) -> None:
        self.prepareGeometryChange()
        self._boxes = np.empty((0, 4), dtype=np.float32)
        self._rects = []
        self._polygons.clear()
        self._bounds = QRectF()
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        if self._boxes.size == 0 and not self._polygons:
            return

        exposed = option.exposedRect
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(self._pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if self._boxes.size > 0:
            # Pad by the cosmetic pen width so edge strokes are not clipped.
            zoom = QStyleOptionGraphicsItem.levelOfDetailFromTransform(
                painter.worldTransform()
            )
            pad = 3.0 / zoom if zoom > 0 else 0.0
            left = exposed.left() - pad
            top = exposed.top() - pad
            right = exposed.right() + pad
            bottom = exposed.bottom() + pad

            visible_mask = (
                (self._boxes[:, 0] <= right)
                & (self._boxes[:, 2] >= left)
                & (self._boxes[:, 1] <= bottom)
                & (self._boxes[:, 3] >= top)
            )
            if visible_mask.all():
                painter.drawRects(self._rects)
            elif visible_mask.any():
                painter.drawRects(
                    [self._rects[i] for i in np.flatnonzero(visible_mask)]
                )

        for poly in self._polygons:
            if exposed.intersects(poly.boundingRect()):
                painter.drawPolygon(poly)

        painter.restore()


class DiagnosticHotspotOverlay(QGraphicsItem):
    """Renders diagnostic peak High-Power Field (HPF) boundary circle and reticle."""

    def __init__(self, z_index: float = 30.0, parent: QGraphicsItem | None = None):
        super().__init__(parent)
        self.setZValue(z_index)
        self._summary: DiagnosticFieldSummary | None = None
        self._pen = QPen(QColor(COLOR_HOTSPOT_BOUNDARY), 3)
        self._pen.setCosmetic(True)

    def set_diagnostic_summary(self, summary: DiagnosticFieldSummary | None) -> None:
        self.prepareGeometryChange()
        self._summary = summary
        self.update()

    def boundingRect(self) -> QRectF:
        if self._summary is None:
            return QRectF()
        cx, cy = self._summary.center_px
        r = self._summary.radius_px
        return QRectF(cx - r - 60, cy - r - 60, 2 * (r + 60), 2 * (r + 60))

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        if self._summary is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self._pen)
        painter.setBrush(QColor(217, 119, 6, 40))

        cx, cy = self._summary.center_px
        r = self._summary.radius_px
        painter.drawEllipse(QPointF(cx, cy), r, r)

        cross_reticle = r * 0.15
        painter.drawLine(
            QPointF(cx - cross_reticle, cy), QPointF(cx + cross_reticle, cy)
        )
        painter.drawLine(
            QPointF(cx, cy - cross_reticle), QPointF(cx, cy + cross_reticle)
        )
        painter.restore()


# ==============================================================================
# SECTION 5: ONNX DETECTOR, MODEL MANAGER & ASYNC INFERENCE WORKER
# ==============================================================================
_CPU_PROVIDER = "CPUExecutionProvider"
_ACCELERATOR_PREFERENCE = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
)
_PROVIDER_LABELS = {
    "CUDAExecutionProvider": "GPU · CUDA",
    "ROCMExecutionProvider": "GPU · ROCm",
    "DmlExecutionProvider": "GPU · DirectML",
    _CPU_PROVIDER: "CPU",
}
_PROVIDER_SHARED_LIBS = {
    "CUDAExecutionProvider": "onnxruntime_providers_cuda",
    "ROCMExecutionProvider": "onnxruntime_providers_rocm",
}
_ONNX_FLOAT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16}

_provider_state_lock = threading.Lock()
_provider_failures: dict[str, str] = {}
_runtime_dlls_preloaded = False


def describe_provider(provider: str) -> str:
    return _PROVIDER_LABELS.get(provider, provider)


def _provider_library_path(provider: str) -> Path | None:
    stem = _PROVIDER_SHARED_LIBS.get(provider)
    if stem is None:
        return None
    if sys.platform == "win32":
        name = f"{stem}.dll"
    elif sys.platform == "darwin":
        name = f"lib{stem}.dylib"
    else:
        name = f"lib{stem}.so"
    path = Path(ort.__file__).resolve().parent / "capi" / name
    return path if path.is_file() else None


def probe_provider_library(provider: str) -> str | None:
    """Loads the provider's shared library and returns the loader error, if any.

    ONNX Runtime lists CUDA as 'available' whenever the provider library ships
    with the wheel, even if its CUDA/cuDNN dependencies are missing. Loading the
    library surfaces the real missing dependency (e.g. libcublasLt.so.12).
    """
    library = _provider_library_path(provider)
    if library is None:
        return None
    try:
        ctypes.CDLL(str(library))
    except OSError as exc:
        return str(exc)
    return None


def _preload_runtime_dlls() -> None:
    """Lets ONNX Runtime (>=1.21) find CUDA/cuDNN pip wheels; best effort."""
    global _runtime_dlls_preloaded
    if _runtime_dlls_preloaded:
        return
    _runtime_dlls_preloaded = True
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception:  # optional helper; any failure is non-fatal
        logger.debug("onnxruntime.preload_dlls() failed", exc_info=True)


def _record_provider_failure(provider: str, reason: str) -> None:
    with _provider_state_lock:
        _provider_failures[provider] = reason


def usable_accelerator() -> str | None:
    """First accelerator that is installed, not known-broken and loadable."""
    available = set(ort.get_available_providers())
    for provider in _ACCELERATOR_PREFERENCE:
        if provider not in available:
            continue
        with _provider_state_lock:
            if provider in _provider_failures:
                continue
        if provider == "CUDAExecutionProvider":
            _preload_runtime_dlls()
        error = probe_provider_library(provider)
        if error is not None:
            _record_provider_failure(provider, error)
            logger.warning("%s cannot be loaded: %s", provider, error)
            continue
        return provider
    return None


def provider_failure_reason(provider: str) -> str | None:
    with _provider_state_lock:
        return _provider_failures.get(provider)


def build_accelerator_diagnostic(provider: str, reason: str) -> str:
    runtime = "CUDA and cuDNN" if provider.startswith("CUDA") else "GPU runtime"
    return (
        f"{describe_provider(provider)} is not usable: {reason}. Running on CPU. "
        f"To enable it, install the {runtime} libraries that match this "
        "onnxruntime-gpu build (see the ONNX Runtime execution-provider "
        "requirements) and make them visible to the dynamic loader; this cannot "
        "be fixed from application code."
    )


@dataclass(frozen=True)
class ProviderReport:
    """Which provider actually runs the model, and why a GPU was not used."""

    active: str
    fallback_reason: str | None = None


@dataclass(frozen=True)
class _ModelContract:
    input_name: str
    input_dtype: type
    static_batch: int | None


class OnnxEosinophilDetector:
    """ONNX Runtime wrapper reproducing Ultralytics detect pre/post-processing.

    Expects a single-output detection export (`yolo export format=onnx`):
      * raw head  -> (B, 4 + num_classes, N), boxes as cx, cy, w, h in input pixels
      * end-to-end / NMS-embedded head -> (B, N, 6) as x1, y1, x2, y2, score, class
    Tiles are already model-sized (640 px), so no letterbox padding is needed.
    """

    def __init__(self, model_path: str | Path, conf_threshold: float) -> None:
        path = Path(model_path)
        if path.suffix.lower() != ".onnx":
            raise ValueError(
                f"Expected an .onnx model, got '{path.name}'. "
                "Export with: yolo export model=best.pt format=onnx"
            )
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        self._session, self._contract, self.provider_report = self._open(path)
        self._conf = conf_threshold

    # ---- session creation ---------------------------------------------------
    @classmethod
    def _open(
        cls, path: Path
    ) -> tuple[ort.InferenceSession, _ModelContract, ProviderReport]:
        accelerator = usable_accelerator()
        gpu_error: str | None = None
        if accelerator is not None:
            try:
                session, contract = cls._create_validated(
                    path, [accelerator, _CPU_PROVIDER]
                )
                active = session.get_providers()
                if not active or active[0] != accelerator:
                    raise RuntimeError(
                        f"{accelerator} was requested but ONNX Runtime registered "
                        f"{active}"
                    )
                logger.info("ONNX Runtime provider: %s", describe_provider(active[0]))
                return session, contract, ProviderReport(active[0])
            except (RuntimeError, ValueError, OSError) as exc:
                gpu_error = str(exc)
                logger.warning("%s initialisation failed: %s", accelerator, exc)

        # A genuine model error re-raises here and is reported as such.
        session, contract = cls._create_validated(path, [_CPU_PROVIDER])
        reason = None
        if accelerator is not None and gpu_error is not None:
            _record_provider_failure(accelerator, gpu_error)
            reason = build_accelerator_diagnostic(accelerator, gpu_error)
        else:
            for provider in _ACCELERATOR_PREFERENCE:
                known = provider_failure_reason(provider)
                if known is not None and provider in ort.get_available_providers():
                    reason = build_accelerator_diagnostic(provider, known)
                    break
        if reason is not None:
            logger.warning(reason)
        logger.info("ONNX Runtime provider: CPU")
        return session, contract, ProviderReport(_CPU_PROVIDER, reason)

    @classmethod
    def _create_validated(
        cls, path: Path, providers: list[str]
    ) -> tuple[ort.InferenceSession, _ModelContract]:
        options = ort.SessionOptions()
        if "DmlExecutionProvider" in providers:
            options.enable_mem_pattern = False  # required by the DirectML provider
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        contract = cls._validate_contract(session)
        cls._smoke_test(session, contract)
        return session, contract

    @staticmethod
    def _validate_contract(session: ort.InferenceSession) -> _ModelContract:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected a single-input detector, found {len(inputs)}.")
        if len(outputs) != 1:
            raise ValueError(
                f"Expected a single-output detection model, found {len(outputs)} "
                "outputs (segmentation models are not supported)."
            )
        model_input = inputs[0]
        if model_input.type not in _ONNX_FLOAT_DTYPES:
            raise ValueError(f"Unsupported ONNX input type: {model_input.type}")
        if outputs[0].type not in _ONNX_FLOAT_DTYPES:
            raise ValueError(f"Unsupported ONNX output type: {outputs[0].type}")
        if len(model_input.shape) != 4:
            raise ValueError(
                f"Expected a 4-D NCHW input, got shape {model_input.shape}."
            )
        batch_dim, channel_dim, height_dim, width_dim = model_input.shape
        if isinstance(channel_dim, int) and channel_dim != 3:
            raise ValueError(f"Model expects {channel_dim} channels, tiles are RGB.")
        for dim in (height_dim, width_dim):
            if isinstance(dim, int) and dim != PATCH_PIXEL_DIMENSION:
                raise ValueError(
                    f"Model input is {height_dim}x{width_dim} but tiles are "
                    f"{PATCH_PIXEL_DIMENSION}x{PATCH_PIXEL_DIMENSION}; re-export with "
                    f"imgsz={PATCH_PIXEL_DIMENSION}."
                )
        return _ModelContract(
            input_name=model_input.name,
            input_dtype=_ONNX_FLOAT_DTYPES[model_input.type],
            # Exports are static-batch unless made with dynamic=True.
            static_batch=batch_dim if isinstance(batch_dim, int) else None,
        )

    @staticmethod
    def _smoke_test(session: ort.InferenceSession, contract: _ModelContract) -> None:
        """One zero-tensor run: initialises kernels and checks the output layout."""
        batch = contract.static_batch or 1
        dummy = np.zeros(
            (batch, 3, PATCH_PIXEL_DIMENSION, PATCH_PIXEL_DIMENSION),
            dtype=contract.input_dtype,
        )
        out = np.asarray(session.run(None, {contract.input_name: dummy})[0])
        if out.ndim != 3 or out.shape[0] != batch or min(out.shape[1:]) < 5:
            raise ValueError(
                f"Unsupported detector output shape {tuple(out.shape)}; expected "
                "(B, 4+nc, N) or (B, N, 6)."
            )

    # ---- inference ----------------------------------------------------------
    @property
    def active_provider(self) -> str:
        return self.provider_report.active

    def effective_batch_size(self, requested: int) -> int:
        return self._contract.static_batch or requested

    def predict(
        self, tiles_rgb: list[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Runs HxWx3 uint8 RGB tiles; returns (xyxy boxes, scores) per tile."""
        step = self._contract.static_batch or len(tiles_rgb)
        results: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(tiles_rgb), step):
            results.extend(self._run_batch(tiles_rgb[start : start + step]))
        return results

    def _run_batch(
        self, tiles_rgb: list[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        count = len(tiles_rgb)
        static = self._contract.static_batch
        # Static-batch exports need a full batch; zero rows pad it and are dropped.
        rows = static if static is not None and count < static else count
        batch = np.zeros(
            (rows, 3, PATCH_PIXEL_DIMENSION, PATCH_PIXEL_DIMENSION), dtype=np.float32
        )
        for index, tile in enumerate(tiles_rgb):
            np.divide(
                tile.transpose(2, 0, 1),
                np.float32(255.0),
                out=batch[index],
                dtype=np.float32,
            )
        if self._contract.input_dtype != np.float32:
            batch = batch.astype(self._contract.input_dtype)

        raw = self._session.run(None, {self._contract.input_name: batch})[0]
        raw = np.asarray(raw, dtype=np.float32)[:count]
        return [self._decode(sample) for sample in raw]

    def _decode(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        limit = float(PATCH_PIXEL_DIMENSION)
        if pred.shape[0] > pred.shape[1]:
            # End-to-end layout (N, 6): already xyxy + score, NMS done in-graph.
            keep = pred[:, 4] > self._conf
            boxes, scores = pred[keep, :4], pred[keep, 4]
        else:
            # Raw layout (4 + nc, N): class-agnostic score, xywh -> xyxy, then NMS.
            scores_all = pred[4:].max(axis=0)
            keep = scores_all > self._conf
            cx, cy, w, h = pred[:4, keep]
            scores = scores_all[keep]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            kept = non_max_suppression(boxes, scores, PATCH_NMS_IOU_THRESHOLD)
            kept = kept[:PATCH_MAX_DETECTIONS]
            boxes, scores = boxes[kept], scores[kept]
        return np.clip(boxes, 0.0, limit), scores


# ------------------------------------------------------------------------------
# Model lifecycle: one background-initialised detector, reused across runs.
# ------------------------------------------------------------------------------
@dataclass
class ModelEntry:
    """A (possibly still loading) detector; `ready` is set when it finishes."""

    key: tuple[str, int | None, int | None]
    ready: threading.Event = field(default_factory=threading.Event)
    detector: OnnxEosinophilDetector | None = None
    error: str | None = None


class _ModelLoadTask(QRunnable):
    def __init__(self, manager: ModelManager, entry: ModelEntry, path: str) -> None:
        super().__init__()
        self._manager = manager
        self._entry = entry
        self._path = path

    def run(self) -> None:
        try:
            self._entry.detector = OnnxEosinophilDetector(
                self._path, BASE_TENSOR_EXTRACTION_CONFIDENCE
            )
        except (ValueError, OSError, RuntimeError) as exc:
            logger.exception("Model initialisation failed")
            self._entry.error = str(exc) or type(exc).__name__
        except Exception as exc:  # last resort: waiters must always be released
            logger.exception("Unexpected model initialisation failure")
            self._entry.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._entry.ready.set()
            self._manager.publish(self._entry)


class ModelManager(QObject):
    """Loads the detector once in the background and reuses it while unchanged.

    The cache key is (resolved path, mtime, size), so replacing the file on disk
    triggers one reload while an unchanged model is never re-initialised. A failed
    load for an unchanged key is cached too (no repeated failing attempts).
    """

    # state: loading | ready | warning | error, short label, tooltip/log detail
    state_changed = Signal(str, str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._entry: ModelEntry | None = None
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(1)

    @staticmethod
    def _make_key(model_path: str) -> tuple[str, int | None, int | None]:
        path = Path(model_path).expanduser()
        try:
            resolved = path.resolve()
            stat = resolved.stat()
        except OSError:
            return (str(path.absolute()), None, None)
        return (str(resolved), stat.st_mtime_ns, stat.st_size)

    def request(self, model_path: str) -> ModelEntry:
        key = self._make_key(model_path)
        with self._lock:
            if self._entry is not None and self._entry.key == key:
                return self._entry
            entry = ModelEntry(key)
            self._entry = entry
        self.state_changed.emit(
            "loading", "Model: loading…", f"Loading {Path(model_path).name}"
        )
        self._pool.start(_ModelLoadTask(self, entry, model_path))
        return entry

    def publish(self, entry: ModelEntry) -> None:
        """Called from the loader thread; ignores superseded entries."""
        with self._lock:
            if self._entry is not entry:
                return
        name = Path(entry.key[0]).name
        if entry.error is not None:
            self.state_changed.emit("error", "Model: error", entry.error)
            return
        report = entry.detector.provider_report  # type: ignore[union-attr]
        if report.fallback_reason:
            self.state_changed.emit(
                "warning",
                f"{describe_provider(report.active)} (GPU unavailable)",
                report.fallback_reason,
            )
        else:
            self.state_changed.emit(
                "ready",
                describe_provider(report.active),
                f"{name} on {describe_provider(report.active)}",
            )

    def shutdown(self, timeout_ms: int = 5000) -> None:
        self._pool.waitForDone(timeout_ms)
        with self._lock:
            self._entry = None


# ------------------------------------------------------------------------------
# Tiled inference
# ------------------------------------------------------------------------------
def patch_geometry(
    spec: SlideSpecification, overlap_ratio: float
) -> tuple[int, int, int, int]:
    """(patch_w, patch_h, stride_x, stride_y) in level-0 pixels (MPP-normalised)."""
    patch_w = max(1, round(PATCH_PIXEL_DIMENSION * (TARGET_MODEL_MPP / spec.mpp_x)))
    patch_h = max(1, round(PATCH_PIXEL_DIMENSION * (TARGET_MODEL_MPP / spec.mpp_y)))
    stride_x = max(1, round(patch_w * (1.0 - overlap_ratio)))
    stride_y = max(1, round(patch_h * (1.0 - overlap_ratio)))
    return patch_w, patch_h, stride_x, stride_y


def plan_tissue_patches(
    full_w: int,
    full_h: int,
    geometry: tuple[int, int, int, int],
    tissue_mask: np.ndarray,
    min_coverage: float,
) -> list[tuple[int, int, int, int]]:
    """Sliding-window patches (x, y, w, h) whose tissue coverage passes the floor."""
    patch_w, patch_h, stride_x, stride_y = geometry
    mask_h, mask_w = tissue_mask.shape
    scale_x = mask_w / float(full_w)
    scale_y = mask_h / float(full_h)

    patches: list[tuple[int, int, int, int]] = []
    for py in range(0, full_h, stride_y):
        for px in range(0, full_w, stride_x):
            pw = min(patch_w, full_w - px)
            ph = min(patch_h, full_h - py)
            if pw < patch_w // 2 or ph < patch_h // 2:
                continue

            mx1 = int(px * scale_x)
            my1 = int(py * scale_y)
            mx2 = max(mx1 + 1, int((px + pw) * scale_x))
            my2 = max(my1 + 1, int((py + ph) * scale_y))

            region = tissue_mask[my1:my2, mx1:mx2]
            if region.size > 0:
                coverage = np.count_nonzero(region) / float(region.size)
                if coverage >= min_coverage:
                    patches.append((px, py, pw, ph))
    return patches


def read_model_tile(
    handles: SlideHandlePool,
    px: int,
    py: int,
    pw: int,
    ph: int,
    patch_w: int,
    patch_h: int,
) -> tuple[np.ndarray, int, int]:
    """Reads a level-0 patch as a 640x640 uint8 RGB model tile.

    Full patches are resized to 640x640 exactly as before. Partial edge patches
    (pw < patch_w or ph < patch_h) are resized with the *same* scale as full
    patches and padded at the bottom/right with PATCH_PAD_VALUE, so the model
    never sees a stretched image. Returns (tile, content_w, content_h), where the
    content size is the unpadded area in tile pixels.
    """
    with handles.borrow() as slide:
        tile_rgba = slide.read_region((px, py), 0, (pw, ph))
    tile_rgb = tile_rgba.convert("RGB")
    tile_rgba.close()
    size = PATCH_PIXEL_DIMENSION
    content_w = min(size, max(1, round(pw * size / patch_w)))
    content_h = min(size, max(1, round(ph * size / patch_h)))
    if pw == patch_w and ph == patch_h:
        content_w = content_h = size
    if (content_w, content_h) != tile_rgb.size:
        resized = tile_rgb.resize((content_w, content_h), Image.Resampling.BILINEAR)
        tile_rgb.close()
        tile_rgb = resized
    content = np.asarray(tile_rgb)
    tile_rgb.close()
    if content_w == size and content_h == size:
        return content, content_w, content_h
    padded = np.full((size, size, 3), PATCH_PAD_VALUE, dtype=np.uint8)
    padded[:content_h, :content_w] = content
    return padded, content_w, content_h


def crop_to_content(
    boxes: np.ndarray, scores: np.ndarray, content_w: int, content_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Drops boxes centred in the padding and clips the rest to the real image."""
    if content_w >= PATCH_PIXEL_DIMENSION and content_h >= PATCH_PIXEL_DIMENSION:
        return boxes, scores
    centre_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centre_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    keep = (centre_x < content_w) & (centre_y < content_h)
    boxes, scores = boxes[keep].copy(), scores[keep]
    boxes[:, 0::2] = np.minimum(boxes[:, 0::2], content_w)
    boxes[:, 1::2] = np.minimum(boxes[:, 1::2], content_h)
    return boxes, scores


def to_slide_coordinates(
    boxes: np.ndarray, origin_x: int, origin_y: int, scale_x: float, scale_y: float
) -> np.ndarray:
    """Maps patch-pixel xyxy boxes to level-0 slide pixels (float64 maths)."""
    mapped = boxes.astype(np.float64)
    mapped[:, 0::2] = origin_x + mapped[:, 0::2] * scale_x
    mapped[:, 1::2] = origin_y + mapped[:, 1::2] * scale_y
    return mapped


class EsophaInferenceWorker(QObject):
    """Asynchronous worker for patched sliding-window inference and cell extraction."""

    progress_updated = Signal(int, int, int)
    status_updated = Signal(str)
    inference_finished = Signal(object, object)
    inference_cancelled = Signal()
    inference_failed = Signal(str)
    worker_finished = Signal()

    def __init__(
        self,
        slide_path: str,
        config: EsophaScopeConfig,
        spec: SlideSpecification,
        tissue_mask: np.ndarray,
        model_entry: ModelEntry,
    ):
        super().__init__()
        self._slide_path = slide_path
        self._config = config
        self._spec = spec
        self._tissue_mask = tissue_mask
        self._model_entry = model_entry
        self._cancel_signal = threading.Event()

    def cancel(self) -> None:
        self._cancel_signal.set()

    def _await_model(self) -> OnnxEosinophilDetector | None:
        entry = self._model_entry
        if not entry.ready.is_set():
            self.status_updated.emit("Waiting for the model to finish loading...")
        while not entry.ready.wait(MODEL_LOAD_POLL_SECONDS):
            if self._cancel_signal.is_set():
                return None
        if entry.error is not None or entry.detector is None:
            raise RuntimeError(entry.error or "Model failed to initialise.")
        return entry.detector

    def run(self) -> None:
        executor: ThreadPoolExecutor | None = None
        handles: SlideHandlePool | None = None
        try:
            detector = self._await_model()
            if detector is None:
                self.inference_cancelled.emit()
                return
            batch_size = detector.effective_batch_size(self._config.batch_size)
            self.status_updated.emit(
                f"Model ready on {describe_provider(detector.active_provider)}."
            )

            spec = self._spec
            geometry = patch_geometry(spec, self._config.patch_overlap_ratio)
            patch_w, patch_h = geometry[0], geometry[1]
            patches = plan_tissue_patches(
                spec.width_px,
                spec.height_px,
                geometry,
                self._tissue_mask,
                self._config.tissue_min_coverage,
            )
            total = len(patches)
            self.status_updated.emit(f"Evaluating {total:,} tissue patches...")
            self.progress_updated.emit(0, total, 0)

            readers = patch_reader_count()
            handles = SlideHandlePool(self._slide_path, readers)
            executor = ThreadPoolExecutor(
                max_workers=readers, thread_name_prefix="esopha-read"
            )
            lookahead = max(2 * batch_size, 2 * readers)
            pending: deque[Future[tuple[np.ndarray, int, int]]] = deque()
            submitted = 0

            box_chunks: list[np.ndarray] = []
            score_chunks: list[np.ndarray] = []
            detection_count = 0
            batch_tiles: list[np.ndarray] = []
            batch_meta: list[
                tuple[int, int, int, int]
            ] = []  # x, y, content_w, content_h

            for idx in range(total):
                # Keep the read-ahead window full; results are consumed in order.
                while submitted < total and len(pending) < lookahead:
                    px, py, pw, ph = patches[submitted]
                    pending.append(
                        executor.submit(
                            read_model_tile, handles, px, py, pw, ph, patch_w, patch_h
                        )
                    )
                    submitted += 1

                if self._cancel_signal.is_set():
                    self.inference_cancelled.emit()
                    return

                tile, content_w, content_h = pending.popleft().result()
                batch_tiles.append(tile)
                batch_meta.append(
                    (patches[idx][0], patches[idx][1], content_w, content_h)
                )

                if len(batch_tiles) == batch_size or idx == total - 1:
                    results = detector.predict(batch_tiles)
                    for (cx, cy, content_w, content_h), (boxes_arr, scores_arr) in zip(
                        batch_meta, results, strict=True
                    ):
                        boxes_arr, scores_arr = crop_to_content(
                            boxes_arr, scores_arr, content_w, content_h
                        )
                        if len(scores_arr) == 0:
                            continue
                        # Uniform scale: identical for full and partial patches.
                        mapped = to_slide_coordinates(
                            boxes_arr,
                            cx,
                            cy,
                            patch_w / float(PATCH_PIXEL_DIMENSION),
                            patch_h / float(PATCH_PIXEL_DIMENSION),
                        )
                        box_chunks.append(mapped.astype(np.float32))
                        score_chunks.append(scores_arr.astype(np.float32))
                        detection_count += len(scores_arr)

                    batch_tiles.clear()
                    batch_meta.clear()
                    self.progress_updated.emit(idx + 1, total, detection_count)

            self.status_updated.emit("Applying slide-level NMS merge...")
            if box_chunks:
                boxes_np = np.concatenate(box_chunks)
                scores_np = np.concatenate(score_chunks)
            else:
                boxes_np = np.empty((0, 4), dtype=np.float32)
                scores_np = np.empty((0,), dtype=np.float32)
            del box_chunks, score_chunks

            retained = non_max_suppression(
                boxes_np, scores_np, self._config.nms_iou_threshold
            )
            cells: list[EosinophilCell] = []
            for cell_id, k in enumerate(retained, start=1):
                b = boxes_np[k]
                cells.append(
                    EosinophilCell(
                        cell_id=cell_id,
                        confidence=round(float(scores_np[k]), 4),
                        bbox_coordinates=(
                            round(float(b[0]), 2),
                            round(float(b[1]), 2),
                            round(float(b[2]), 2),
                            round(float(b[3]), 2),
                        ),
                    )
                )
            del boxes_np, scores_np, retained

            hotspot = identify_peak_diagnostic_hpf(cells, spec.mpp_x, spec.mpp_y)
            self.inference_finished.emit(cells, hotspot)

        except (
            RuntimeError,
            ValueError,
            OSError,
            openslide.OpenSlideError,
            TypeError,
            MemoryError,
        ) as exc:
            logger.exception("EsophaScope inference error")
            self.inference_failed.emit(str(exc) or type(exc).__name__)
        except Exception as exc:  # last resort: the GUI must never stay EXECUTING
            logger.exception("Unexpected EsophaScope inference failure")
            self.inference_failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if handles is not None:
                handles.shutdown()
            self.worker_finished.emit()


# ==============================================================================
# SECTION 6: WORKBENCH VIEWPORT & MAIN WINDOW
# ==============================================================================
class WorkbenchOperationalState(Enum):
    READY = "ready"
    LOADING = "loading"
    EXECUTING = "executing"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class EsophaConfidenceFader(QFrame):
    """Interactive vertical confidence fader with quick F1 calibration trigger."""

    confidence_adjusted = Signal(float)
    calibrate_requested = Signal()

    def __init__(
        self,
        initial_val: float = DEFAULT_OPERATIONAL_CONFIDENCE,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("hudConfidenceFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 4, 3, 4)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._lbl_pct = QLabel(f"{round(initial_val * 100)}%")
        self._lbl_pct.setProperty("numeric", "true")
        self._lbl_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_pct.setStyleSheet(
            f"font-size: 7.5pt; font-weight: 700; color: {COLOR_ACCENT_BASE};"
        )
        layout.addWidget(self._lbl_pct)

        self._slider = QSlider(Qt.Orientation.Vertical)
        self._slider.setObjectName("verticalConfidenceSlider")
        self._slider.setRange(5, 95)
        self._slider.setValue(round(initial_val * 100))
        self._slider.setFixedHeight(75)
        self._slider.valueChanged.connect(self._on_moved)
        layout.addWidget(self._slider, alignment=Qt.AlignmentFlag.AlignCenter)

        self._btn_f1 = QPushButton("F1")
        self._btn_f1.setObjectName("btnCalibrateF1")
        self._btn_f1.setToolTip("Auto-calibrate threshold for optimal F1 score")
        self._btn_f1.clicked.connect(self.calibrate_requested.emit)
        layout.addWidget(self._btn_f1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.adjustSize()

    def _on_moved(self, val: int) -> None:
        self._lbl_pct.setText(f"{val}%")
        self.confidence_adjusted.emit(val / 100.0)

    def set_confidence(self, val: float) -> None:
        val_int = round(val * 100)
        self._slider.blockSignals(True)
        self._slider.setValue(val_int)
        self._slider.blockSignals(False)
        self._lbl_pct.setText(f"{val_int}%")

    def confidence(self) -> float:
        return self._slider.value() / 100.0


def dropped_local_paths(mime_data) -> list[str]:
    """Local files in a drag payload that are slides or annotation files."""
    allowed = SLIDE_EXTENSIONS | ANNOTATION_EXTENSIONS
    return [
        url.toLocalFile()
        for url in mime_data.urls()
        if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in allowed
    ]


class EsophaViewport(QGraphicsView):
    """Pannable and zoomable graphics viewport with anchored HUD frames."""

    files_dropped = Signal(list)

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
        )

        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

        self._hud_layer_controls: QWidget | None = None
        self._hud_confidence_fader: QWidget | None = None
        self._hud_hotspot_action: QWidget | None = None

        # Explicit margins for the HPF button
        self.hpf_button_margin_left = DEFAULT_HPF_BUTTON_MARGIN_LEFT
        self.hpf_button_margin_bottom = DEFAULT_HPF_BUTTON_MARGIN_BOTTOM

    def bind_huds(
        self,
        layer_controls: QWidget,
        confidence_fader: QWidget,
        hotspot_action: QWidget,
    ) -> None:
        self._hud_layer_controls = layer_controls
        self._hud_confidence_fader = confidence_fader
        self._hud_hotspot_action = hotspot_action
        self._reposition_huds()

    def set_hpf_button_margins(self, margin_left: int, margin_bottom: int) -> None:
        """Sets custom left and bottom margin offsets for the HPF button."""
        self.hpf_button_margin_left = margin_left
        self.hpf_button_margin_bottom = margin_bottom
        self._reposition_huds()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._reposition_huds()

    def _reposition_huds(self) -> None:
        if self._hud_layer_controls:
            self._hud_layer_controls.move(14, 14)

        if self._hud_confidence_fader:
            fh = self._hud_confidence_fader.height()
            self._hud_confidence_fader.move(14, self.height() - fh - 14)

        # Reposition HPF button using left and bottom margin specifications
        if self._hud_hotspot_action:
            ah = self._hud_hotspot_action.height()
            fader_w = (
                self._hud_confidence_fader.width() if self._hud_confidence_fader else 0
            )
            # Ensure it clears the confidence fader while honoring left margin
            pos_x = max(self.hpf_button_margin_left, 14 + fader_w + 10)
            pos_y = self.height() - ah - self.hpf_button_margin_bottom
            self._hud_hotspot_action.move(pos_x, pos_y)

    # QGraphicsView's own drag handlers track scene drag state. This view takes
    # over enter/move/drop for file drops, so it must also take over leave:
    # the inherited dragLeaveEvent warns "drag leave received before drag enter".
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if dropped_local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if dropped_local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = dropped_local_paths(event.mimeData())
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() == 0:
            return
        factor = 1.20 if event.angleDelta().y() > 0 else (1.0 / 1.20)
        self.scale(factor, factor)
        event.accept()


class EsophaScopeWindow(QMainWindow):
    """Main application window for the EsophaScope pathology workstation."""

    def __init__(self, config: EsophaScopeConfig) -> None:
        super().__init__()
        self._config = config
        self.setWindowTitle("EsophaScope — Quantitative WSI Hotspot Diagnostics")
        self.setWindowIcon(load_app_icon())
        self.resize(1420, 920)
        self.setAcceptDrops(True)

        self._slide_spec: SlideSpecification | None = None
        self._tissue_mask: np.ndarray | None = None
        self._all_cells: list[EosinophilCell] = []
        self._visible_cells: list[EosinophilCell] = []
        self._ground_truth: list[GroundTruthCell] = []
        self._hotspot: DiagnosticFieldSummary | None = None

        self._canvas_item: PyramidalSlideCanvas | None = None
        self._inference_thread: QThread | None = None
        self._inference_worker: EsophaInferenceWorker | None = None
        self._state = WorkbenchOperationalState.READY
        self._close_pending = False

        self._models = ModelManager(self)
        self._model_entry: ModelEntry | None = None
        self._load_pool = QThreadPool(self)
        self._load_pool.setMaxThreadCount(1)  # loads are serialised; newest wins
        self._load_signals = SlideLoadSignals(self)
        self._load_tasks: dict[int, SlideLoadTask] = {}
        self._load_request_id = 0

        # White glyphs: both icons sit on the filled primary/critical button.
        self._icon_play = build_svg_icon("play", COLOR_TEXT_LIGHT)
        self._icon_stop = build_svg_icon("stop", COLOR_TEXT_LIGHT)

        self._setup_ui()
        self._setup_shortcuts()
        self._sync_control_states()

        self._load_signals.status.connect(self._on_slide_status)
        self._load_signals.loaded.connect(self._on_slide_loaded)
        self._load_signals.failed.connect(self._on_slide_load_failed)
        self._load_signals.finished.connect(self._on_slide_load_finished)
        self._models.state_changed.connect(self._on_model_state)
        # Start initialising the model immediately so it is ready before "Run".
        self._model_entry = self._models.request(self._config.model_weights_path)

    def _setup_shortcuts(self) -> None:
        self._shortcut_reset = QShortcut(QKeySequence("Ctrl+W"), self)
        self._shortcut_reset.activated.connect(self._reset_workspace)

        self._shortcut_quit = QShortcut(QKeySequence("Ctrl+Q"), self)
        self._shortcut_quit.activated.connect(self.close)

    def _setup_ui(self) -> None:
        self._scene = QGraphicsScene(self)
        self._viewport = EsophaViewport(self._scene)
        self._viewport.setObjectName("esophaViewport")
        self._viewport.files_dropped.connect(self._handle_dropped_files)

        self._tissue_overlay_item = QGraphicsPixmapItem()
        self._tissue_overlay_item.setZValue(5.0)
        self._scene.addItem(self._tissue_overlay_item)

        self._gt_overlay = FastVectorOverlay(COLOR_GROUND_TRUTH, z_index=10.0)
        self._cell_overlay = FastVectorOverlay(COLOR_PREDICTION_BOX, z_index=20.0)
        self._hotspot_overlay = DiagnosticHotspotOverlay(z_index=30.0)
        self._scene.addItem(self._gt_overlay)
        self._scene.addItem(self._cell_overlay)
        self._scene.addItem(self._hotspot_overlay)

        self._create_viewport_huds()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_vbox = QVBoxLayout(central_widget)
        root_vbox.setContentsMargins(12, 12, 12, 12)
        root_vbox.setSpacing(8)

        root_vbox.addWidget(self._build_toolbar())

        content_hbox = QHBoxLayout()
        content_hbox.setSpacing(6)
        content_hbox.addWidget(self._viewport, stretch=1)

        toggle_frame = QWidget()
        toggle_layout = QVBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._btn_toggle_sidebar = QPushButton()
        self._icon_sidebar_closed = build_svg_icon(
            "chevron-left", "#64748B", active_color=COLOR_ACCENT_BASE
        )
        self._icon_sidebar_open = build_svg_icon(
            "chevron-right", "#64748B", active_color=COLOR_ACCENT_BASE
        )
        self._btn_toggle_sidebar.setIcon(self._icon_sidebar_closed)
        self._btn_toggle_sidebar.setIconSize(QSize(14, 14))
        self._btn_toggle_sidebar.setObjectName("btnSidebarToggle")
        self._btn_toggle_sidebar.setToolTip("Toggle diagnostics and metrics sidebar")
        self._btn_toggle_sidebar.clicked.connect(self._toggle_sidebar)
        toggle_layout.addWidget(self._btn_toggle_sidebar)
        content_hbox.addWidget(toggle_frame)

        content_hbox.addWidget(self._build_sidebar(), stretch=0)
        root_vbox.addLayout(content_hbox, stretch=1)

        root_vbox.addWidget(self._build_status_panel())

    def _create_viewport_huds(self) -> None:
        # Layer Selection HUD
        self._hud_layers = QFrame(self._viewport)
        self._hud_layers.setObjectName("hudLayerFrame")
        layers_layout = QVBoxLayout(self._hud_layers)
        layers_layout.setContentsMargins(8, 6, 8, 6)
        layers_layout.setSpacing(4)

        lbl_layers = QLabel("Overlays")
        layers_layout.addWidget(lbl_layers)

        self._chk_tissue = QCheckBox("Tissue Mask")
        self._chk_tissue.setProperty("overlayLayer", "tissue")
        self._chk_tissue.setChecked(False)
        self._chk_tissue.toggled.connect(self._tissue_overlay_item.setVisible)
        layers_layout.addWidget(self._chk_tissue)

        self._chk_gt = QCheckBox("Ground Truth")
        self._chk_gt.setProperty("overlayLayer", "groundTruth")
        self._chk_gt.setChecked(True)
        self._chk_gt.toggled.connect(self._gt_overlay.setVisible)
        layers_layout.addWidget(self._chk_gt)

        self._chk_detections = QCheckBox("Predictions")
        self._chk_detections.setProperty("overlayLayer", "detection")
        self._chk_detections.setChecked(True)
        self._chk_detections.toggled.connect(self._cell_overlay.setVisible)
        layers_layout.addWidget(self._chk_detections)

        self._chk_hotspot = QCheckBox("Peak HPF Hotspot")
        self._chk_hotspot.setProperty("overlayLayer", "hotspot")
        self._chk_hotspot.setChecked(True)
        self._chk_hotspot.toggled.connect(self._hotspot_overlay.setVisible)
        layers_layout.addWidget(self._chk_hotspot)
        self._hud_layers.adjustSize()

        # Confidence Fader HUD
        self._hud_fader = EsophaConfidenceFader(
            self._config.confidence_threshold, self._viewport
        )
        self._hud_fader.confidence_adjusted.connect(self._on_confidence_changed)
        self._hud_fader.calibrate_requested.connect(self._auto_calibrate_f1)

        # Hotspot Focus FAB with Left and Bottom Margins
        self._btn_fab_hotspot = QPushButton(self._viewport)
        self._btn_fab_hotspot.setIcon(build_svg_icon("crosshair", COLOR_TEXT_LIGHT))
        self._btn_fab_hotspot.setIconSize(QSize(20, 20))
        self._btn_fab_hotspot.setObjectName("btnHotspotFAB")
        self._btn_fab_hotspot.setToolTip("Center on diagnostic peak HPF field")
        self._btn_fab_hotspot.setEnabled(False)
        self._btn_fab_hotspot.clicked.connect(self._focus_diagnostic_hotspot)
        self._btn_fab_hotspot.adjustSize()

        self._viewport.bind_huds(
            self._hud_layers, self._hud_fader, self._btn_fab_hotspot
        )

    def _build_toolbar(self) -> QFrame:
        toolbar = QFrame()
        toolbar.setObjectName("esophaToolbar")
        vbox = QVBoxLayout(toolbar)
        vbox.setContentsMargins(10, 8, 10, 8)
        vbox.setSpacing(6)

        actions = QHBoxLayout()
        actions.setSpacing(10)

        self._btn_open_slide = QPushButton("Open WSI Slide")
        self._btn_open_slide.setIcon(build_svg_icon("folder"))
        self._btn_open_slide.clicked.connect(self._prompt_open_slide)
        actions.addWidget(self._btn_open_slide)

        self._btn_open_gt = QPushButton("Load GeoJSON")
        self._btn_open_gt.setIcon(build_svg_icon("file-text"))
        self._btn_open_gt.clicked.connect(self._prompt_open_gt)
        actions.addWidget(self._btn_open_gt)

        actions.addWidget(self._create_divider())

        self._btn_run_cancel = QPushButton("Run Tiled Inference")
        self._btn_run_cancel.setProperty("role", "primary")
        self._btn_run_cancel.setIcon(self._icon_play)
        self._btn_run_cancel.clicked.connect(self._on_run_or_cancel_clicked)
        actions.addWidget(self._btn_run_cancel)

        actions.addWidget(self._create_divider())

        self._btn_export_geojson = QPushButton("Export QuPath GeoJSON")
        self._btn_export_geojson.setIcon(build_svg_icon("download"))
        self._btn_export_geojson.clicked.connect(self._export_geojson)
        actions.addWidget(self._btn_export_geojson)

        actions.addStretch(1)

        hints = QLabel("Clear: Ctrl+W  |  Quit: Ctrl+Q")
        hints.setStyleSheet(
            f"color: {COLOR_TEXT_DISABLED}; font-size: 8pt; font-weight: 500;"
        )
        actions.addWidget(hints)

        vbox.addLayout(actions)
        return toolbar

    @staticmethod
    def _create_divider() -> QFrame:
        div = QFrame()
        div.setObjectName("panelSeparator")
        div.setFrameShape(QFrame.Shape.VLine)
        return div

    def _build_sidebar(self) -> QFrame:
        self._sidebar_panel = QFrame()
        self._sidebar_panel.setObjectName("esophaSidebar")
        self._sidebar_panel.setFixedWidth(340)
        self._sidebar_panel.setVisible(False)
        main_layout = QVBoxLayout(self._sidebar_panel)
        main_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setSpacing(10)

        lbl_diag = QLabel("CLINICAL DIAGNOSIS")
        lbl_diag.setProperty("sidebarHeading", "true")
        vbox.addWidget(lbl_diag)

        card_diag = QFrame()
        card_diag.setProperty("sidebarCard", "true")
        g1 = QGridLayout(card_diag)
        g1.setVerticalSpacing(8)

        self._lbl_status_badge = QLabel("UNANALYZED")
        self._lbl_status_badge.setObjectName("diagnosticStatusBadge")
        self._lbl_status_badge.setProperty("clinicalEvaluation", "none")
        self._lbl_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._lbl_peak_count = QLabel("—")
        self._lbl_peak_density = QLabel("—")
        for lbl in (self._lbl_peak_count, self._lbl_peak_density):
            lbl.setProperty("numeric", "true")

        g1.addWidget(QLabel("Diagnostic Status:"), 0, 0)
        g1.addWidget(self._lbl_status_badge, 0, 1)
        g1.addWidget(QLabel("Peak Eos Count:"), 1, 0)
        g1.addWidget(self._lbl_peak_count, 1, 1)
        g1.addWidget(QLabel("Peak Density:"), 2, 0)
        g1.addWidget(self._lbl_peak_density, 2, 1)
        vbox.addWidget(card_diag)

        lbl_eval = QLabel("CELLULAR VALIDATION METRICS")
        lbl_eval.setProperty("sidebarHeading", "true")
        vbox.addWidget(lbl_eval)

        card_eval = QFrame()
        card_eval.setProperty("sidebarCard", "true")
        g2 = QGridLayout(card_eval)
        g2.setVerticalSpacing(6)

        self._lbl_f1 = QLabel("—")
        self._lbl_prec = QLabel("—")
        self._lbl_rec = QLabel("—")
        self._lbl_dist_err = QLabel("—")

        for lbl in (self._lbl_f1, self._lbl_prec, self._lbl_rec, self._lbl_dist_err):
            lbl.setProperty("numeric", "true")

        g2.addWidget(QLabel("Centroid F1-Score:"), 0, 0)
        g2.addWidget(self._lbl_f1, 0, 1)
        g2.addWidget(QLabel("Precision:"), 1, 0)
        g2.addWidget(self._lbl_prec, 1, 1)
        g2.addWidget(QLabel("Recall (Sensitivity):"), 2, 0)
        g2.addWidget(self._lbl_rec, 2, 1)
        g2.addWidget(QLabel("Mean Centroid Error:"), 3, 0)
        g2.addWidget(self._lbl_dist_err, 3, 1)
        vbox.addWidget(card_eval)

        lbl_meta = QLabel("SLIDE METADATA")
        lbl_meta.setProperty("sidebarHeading", "true")
        vbox.addWidget(lbl_meta)

        card_meta = QFrame()
        card_meta.setProperty("sidebarCard", "true")
        g3 = QGridLayout(card_meta)
        g3.setVerticalSpacing(6)

        self._lbl_meta_dims = QLabel("—")
        self._lbl_meta_mpp = QLabel("—")
        self._lbl_meta_vendor = QLabel("—")
        self._lbl_meta_tissue = QLabel("—")
        for lbl in (
            self._lbl_meta_dims,
            self._lbl_meta_mpp,
            self._lbl_meta_vendor,
            self._lbl_meta_tissue,
        ):
            lbl.setProperty("numeric", "true")

        g3.addWidget(QLabel("Dimensions:"), 0, 0)
        g3.addWidget(self._lbl_meta_dims, 0, 1)
        g3.addWidget(QLabel("Resolution:"), 1, 0)
        g3.addWidget(self._lbl_meta_mpp, 1, 1)
        g3.addWidget(QLabel("Scanner:"), 2, 0)
        g3.addWidget(self._lbl_meta_vendor, 2, 1)
        g3.addWidget(QLabel("Tissue Surface:"), 3, 0)
        g3.addWidget(self._lbl_meta_tissue, 3, 1)
        vbox.addWidget(card_meta)
        vbox.addStretch(1)

        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        return self._sidebar_panel

    def _build_status_panel(self) -> QFrame:
        """Constructs enhanced bottom status HUD with status pills, progress, and telemetry log."""
        panel = QFrame()
        panel.setObjectName("esophaStatusPanel")
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(12, 8, 12, 8)
        vbox.setSpacing(6)

        # 1. Top Ribbon: Diagnostic Indicators & Hardware Tags
        ribbon = QHBoxLayout()
        ribbon.setSpacing(8)

        self._pill_state = QLabel("● READY")
        self._pill_state.setObjectName("hudStatePill")
        self._pill_state.setProperty("state", "ready")

        self._pill_slide = QLabel("No slide loaded")
        self._pill_slide.setObjectName("hudInfoPill")

        self._pill_detections = QLabel("Detections: 0")
        self._pill_detections.setObjectName("hudInfoPill")
        self._pill_detections.setProperty("numeric", "true")

        self._pill_hotspot = QLabel("Peak HPF: —")
        self._pill_hotspot.setObjectName("hudInfoPill")

        ribbon.addWidget(self._pill_state)
        ribbon.addWidget(self._pill_slide)
        ribbon.addWidget(self._pill_detections)
        ribbon.addWidget(self._pill_hotspot)
        ribbon.addStretch(1)

        self._pill_hardware = QLabel("Model: loading…")
        self._pill_hardware.setObjectName("hudHardwarePill")
        self._pill_hardware.setProperty("modelState", "loading")
        ribbon.addWidget(self._pill_hardware)

        vbox.addLayout(ribbon)

        # 2. Activity / Progress Bar Strip
        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("hudProgressBar")
        self._progress_bar.setTextVisible(False)

        self._lbl_progress_detail = QLabel("0 / 0 patches")
        self._lbl_progress_detail.setObjectName("hudProgressDetail")
        self._lbl_progress_detail.setProperty("numeric", "true")

        progress_row.addWidget(self._progress_bar, stretch=1)
        progress_row.addWidget(self._lbl_progress_detail)
        vbox.addLayout(progress_row)

        # 3. Illuminated Telemetry Console Log
        console_frame = QFrame()
        console_frame.setObjectName("hudLogBox")
        console_layout = QHBoxLayout(console_frame)
        console_layout.setContentsMargins(8, 4, 8, 4)
        console_layout.setSpacing(8)

        self._lbl_log_prefix = QLabel("LOG")
        self._lbl_log_prefix.setObjectName("hudLogPrefix")

        self._lbl_status_msg = QLabel(
            "Ready. Drag & drop a whole-slide image to begin."
        )
        self._lbl_status_msg.setObjectName("hudLogMessage")

        console_layout.addWidget(self._lbl_log_prefix)
        console_layout.addWidget(self._lbl_status_msg, stretch=1)
        vbox.addWidget(console_frame)

        return panel

    def _toggle_sidebar(self) -> None:
        visible = not self._sidebar_panel.isVisible()
        self._sidebar_panel.setVisible(visible)
        self._btn_toggle_sidebar.setIcon(
            self._icon_sidebar_open if visible else self._icon_sidebar_closed
        )

    def _reset_workspace(self) -> None:
        """Clears all loaded slides, annotations, and UI state."""
        if self._is_worker_busy():
            QMessageBox.warning(
                self,
                "EsophaScope Busy",
                "Please cancel inference before resetting workspace.",
            )
            return

        self._cancel_pending_loads()
        if self._canvas_item:
            self._scene.removeItem(self._canvas_item)
            self._canvas_item.close()
            self._canvas_item = None

        self._tissue_overlay_item.setPixmap(QPixmap())
        self._gt_overlay.clear()
        self._cell_overlay.clear()
        self._hotspot_overlay.set_diagnostic_summary(None)
        self._btn_fab_hotspot.setEnabled(False)

        self._slide_spec = None
        self._tissue_mask = None
        self._all_cells.clear()
        self._visible_cells.clear()
        self._ground_truth.clear()
        self._hotspot = None

        self._pill_slide.setText("No slide loaded")
        self._pill_detections.setText("Detections: 0")
        self._pill_hotspot.setText("Peak HPF: —")
        self._set_idle_progress()

        self._lbl_status_badge.setText("UNANALYZED")
        self._lbl_status_badge.setProperty("clinicalEvaluation", "none")
        self._refresh_styling(self._lbl_status_badge)

        self._lbl_peak_count.setText("—")
        self._lbl_peak_density.setText("—")
        self._lbl_f1.setText("—")
        self._lbl_prec.setText("—")
        self._lbl_rec.setText("—")
        self._lbl_dist_err.setText("—")
        self._lbl_meta_dims.setText("—")
        self._lbl_meta_mpp.setText("—")
        self._lbl_meta_mpp.setProperty("calibration", "ok")
        self._lbl_meta_mpp.setToolTip("")
        self._refresh_styling(self._lbl_meta_mpp)
        self._lbl_meta_vendor.setText("—")
        self._lbl_meta_tissue.setText("—")

        self._transition_state(
            WorkbenchOperationalState.READY, "Workspace cleared (Ctrl+W)."
        )

    @staticmethod
    def _refresh_styling(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _prompt_open_slide(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Pathology Whole-Slide Image",
            "",
            "WSI Files (*.svs *.ndpi *.mrxs *.tif *.tiff);;All (*)",
        )
        if path:
            self._load_slide(path)

    def _prompt_open_gt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open QuPath GeoJSON", "", "GeoJSON (*.geojson *.json);;All (*)"
        )
        if path:
            self._load_gt(path)

    def _is_loading(self) -> bool:
        return self._state == WorkbenchOperationalState.LOADING

    def _cancel_pending_loads(self) -> None:
        """Flags every outstanding load and invalidates their results."""
        self._load_request_id += 1
        for task in self._load_tasks.values():
            task.cancel()

    def _set_idle_progress(self) -> None:
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._lbl_progress_detail.setText("0 / 0 patches")

    def _load_slide(self, slide_path: str) -> None:
        """Queues a background load; a newer request supersedes any older one."""
        if self._is_worker_busy():
            return
        self._cancel_pending_loads()
        task = SlideLoadTask(
            self._load_request_id, slide_path, self._config, self._load_signals
        )
        self._load_tasks[self._load_request_id] = task
        self._progress_bar.setRange(0, 0)  # busy indicator
        self._lbl_progress_detail.setText("Opening slide…")
        self._transition_state(
            WorkbenchOperationalState.LOADING, f"Opening {Path(slide_path).name}..."
        )
        self._load_pool.start(task)

    @Slot(int, str)
    def _on_slide_status(self, request_id: int, message: str) -> None:
        if request_id == self._load_request_id:
            self._log_status(message)

    @Slot(int, object)
    def _on_slide_loaded(self, request_id: int, result: SlideLoadResult) -> None:
        """GUI-thread half of loading: builds pixmaps/items and swaps the slide."""
        if request_id != self._load_request_id:
            return  # stale result from a superseded request
        spec = result.spec
        thumb_h, thumb_w = result.thumbnail_rgb.shape[:2]

        overview = QPixmap.fromImage(
            QImage(
                result.thumbnail_rgb.data,
                thumb_w,
                thumb_h,
                3 * thumb_w,
                QImage.Format.Format_RGB888,
            )
        )
        tissue_pixmap = QPixmap.fromImage(
            QImage(
                result.tissue_overlay_rgba.data,
                thumb_w,
                thumb_h,
                4 * thumb_w,
                QImage.Format.Format_RGBA8888,
            )
        )

        previous = self._canvas_item
        if previous is not None:
            self._scene.removeItem(previous)
            previous.close()
        self._canvas_item = PyramidalSlideCanvas(spec, overview)
        self._scene.addItem(self._canvas_item)
        self._scene.setSceneRect(QRectF(0, 0, spec.width_px, spec.height_px))
        self._viewport.fitInView(
            self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

        self._tissue_overlay_item.setPixmap(tissue_pixmap)
        # Independent X/Y scale: the thumbnail's aspect may differ by a pixel.
        self._tissue_overlay_item.setTransform(
            QTransform.fromScale(spec.width_px / thumb_w, spec.height_px / thumb_h)
        )
        self._tissue_overlay_item.setVisible(self._chk_tissue.isChecked())

        # Detections/hotspot belong to the previous slide; ground truth is kept.
        self._all_cells = []
        self._visible_cells = []
        self._hotspot = None
        self._cell_overlay.clear()
        self._hotspot_overlay.set_diagnostic_summary(None)
        self._btn_fab_hotspot.setEnabled(False)
        self._pill_detections.setText("Detections: 0")
        self._pill_hotspot.setText("Peak HPF: —")

        self._slide_spec = spec
        self._tissue_mask = result.tissue_mask

        self._pill_slide.setText(spec.file_name)
        self._lbl_meta_dims.setText(f"{spec.width_px:,} × {spec.height_px:,} px")
        mpp_text = f"{spec.mpp_x:.4f} × {spec.mpp_y:.4f} µm/px"
        if spec.mpp_is_fallback:
            mpp_text += " (ASSUMED)"
        self._lbl_meta_mpp.setText(mpp_text)
        self._lbl_meta_mpp.setProperty(
            "calibration", "fallback" if spec.mpp_is_fallback else "ok"
        )
        self._lbl_meta_mpp.setToolTip("\n".join(spec.calibration_notes))
        self._refresh_styling(self._lbl_meta_mpp)
        self._lbl_meta_vendor.setText(f"{spec.scanner_model} ({spec.magnification}x)")
        self._lbl_meta_tissue.setText(f"{spec.tissue_surface_area_mm2:.2f} mm²")

        self._set_idle_progress()
        self._update_metrics_view()
        self._transition_state(
            WorkbenchOperationalState.READY, "Slide ready for analysis."
        )

        for note in spec.calibration_notes:
            logger.warning("Calibration: %s", note)
        if spec.mpp_is_fallback:
            self._log_status("WARNING: slide calibration missing; MPP is assumed.")
            QMessageBox.warning(
                self,
                "Slide calibration",
                "\n\n".join(spec.calibration_notes),
            )

    @Slot(int, str)
    def _on_slide_load_failed(self, request_id: int, message: str) -> None:
        if request_id != self._load_request_id:
            return
        self._set_idle_progress()
        self._transition_state(
            WorkbenchOperationalState.READY, "Slide could not be opened."
        )
        QMessageBox.critical(self, "OpenSlide Loading Error", message)

    @Slot(int)
    def _on_slide_load_finished(self, request_id: int) -> None:
        self._load_tasks.pop(request_id, None)
        if request_id == self._load_request_id and self._is_loading():
            # Finished without a result (e.g. cancelled): leave LOADING.
            self._set_idle_progress()
            self._transition_state(WorkbenchOperationalState.READY)
        if self._close_pending and not self._load_tasks and not self._is_worker_busy():
            self.close()

    @Slot(str, str, str)
    def _on_model_state(self, state: str, label: str, detail: str) -> None:
        self._pill_hardware.setText(label)
        self._pill_hardware.setProperty("modelState", state)
        self._pill_hardware.setToolTip(detail)
        self._refresh_styling(self._pill_hardware)
        if state == "ready":
            self._log_status(f"Model ready: {detail}.")
        elif state == "warning":
            self._log_status("Model ready on CPU; GPU unavailable (see tooltip).")
        elif state == "error":
            self._log_status(f"Model unavailable: {detail}")

    def _load_gt(self, path: str) -> None:
        try:
            self._ground_truth = parse_qupath_geojson(path)
            boxes = [
                g.bbox_coordinates for g in self._ground_truth if not g.boundary_points
            ]
            polys = [g.boundary_points for g in self._ground_truth if g.boundary_points]
            self._gt_overlay.set_vector_data(boxes, polys)
            self._update_metrics_view()
            self._log_status(f"Imported {len(self._ground_truth):,} GT annotations.")
        except (json.JSONDecodeError, OSError, ValueError, KeyError) as exc:
            logger.exception("Failed to parse GeoJSON")
            QMessageBox.warning(self, "Annotation Import Error", str(exc))

    def _on_run_or_cancel_clicked(self) -> None:
        if self._state == WorkbenchOperationalState.EXECUTING:
            self._cancel_inference()
        else:
            self._execute_inference()

    def _execute_inference(self) -> None:
        if (
            not self._slide_spec
            or self._tissue_mask is None
            or self._is_worker_busy()
            or self._is_loading()
        ):
            return

        self._all_cells.clear()
        self._visible_cells.clear()
        self._cell_overlay.clear()
        self._hotspot_overlay.set_diagnostic_summary(None)

        self._transition_state(
            WorkbenchOperationalState.EXECUTING,
            "Dispatching batched inference worker...",
        )
        self._inference_thread = QThread(self)
        self._inference_worker = EsophaInferenceWorker(
            self._slide_spec.slide_path,
            self._config,
            self._slide_spec,
            self._tissue_mask,
            # Reuses the already-initialised session unless the model file changed.
            self._models.request(self._config.model_weights_path),
        )
        self._inference_worker.moveToThread(self._inference_thread)

        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.progress_updated.connect(self._on_progress)
        self._inference_worker.status_updated.connect(self._log_status)
        self._inference_worker.inference_failed.connect(self._on_inference_failed)
        self._inference_worker.inference_finished.connect(self._on_inference_finished)
        self._inference_worker.inference_cancelled.connect(self._on_inference_cancelled)

        self._inference_worker.worker_finished.connect(self._inference_thread.quit)
        self._inference_worker.worker_finished.connect(
            self._inference_worker.deleteLater
        )
        self._inference_thread.finished.connect(self._inference_thread.deleteLater)
        self._inference_thread.finished.connect(self._on_inference_thread_finished)

        self._sync_control_states()
        self._inference_thread.start()

    @Slot()
    def _on_inference_thread_finished(self) -> None:
        # The QThread/worker are deleteLater()'d; drop the dangling references.
        self._inference_thread = None
        self._inference_worker = None

    def _cancel_inference(self) -> None:
        if (
            self._inference_worker
            and self._state == WorkbenchOperationalState.EXECUTING
        ):
            self._transition_state(
                WorkbenchOperationalState.CANCELLING, "Halting background inference..."
            )
            self._inference_worker.cancel()

    @Slot(int, int, int)
    def _on_progress(self, current: int, total: int, count: int) -> None:
        self._progress_bar.setMaximum(max(1, total))
        self._progress_bar.setValue(current)
        self._pill_detections.setText(f"Detections: {count:,}")
        pct = (current / max(1, total)) * 100.0
        self._lbl_progress_detail.setText(f"{current:,} / {total:,} ({pct:.1f}%)")

    @Slot(object, object)
    def _on_inference_finished(
        self, cells: list[EosinophilCell], hotspot: DiagnosticFieldSummary | None
    ) -> None:
        self._all_cells = cells
        self._hotspot = hotspot
        self._apply_confidence_filtering()
        self._transition_state(
            WorkbenchOperationalState.COMPLETED,
            f"Done: {len(self._visible_cells):,} eosinophils identified.",
        )
        if self._close_pending:
            self.close()

    @Slot(str)
    def _on_inference_failed(self, message: str) -> None:
        # Without this transition a failed run (e.g. missing model) left the UI
        # stuck in EXECUTING.
        self._transition_state(WorkbenchOperationalState.ERROR, "Inference failed.")
        QMessageBox.critical(self, "Inference Failed", message)
        if self._close_pending:
            self.close()

    @Slot()
    def _on_inference_cancelled(self) -> None:
        self._transition_state(
            WorkbenchOperationalState.CANCELLED, "Inference cancelled."
        )
        if self._close_pending:
            self.close()

    def _on_confidence_changed(self, conf: float) -> None:
        self._config = replace(self._config, confidence_threshold=conf)
        if self._all_cells:
            self._apply_confidence_filtering()

    def _auto_calibrate_f1(self) -> None:
        """Finds optimal F1 threshold in a single pass over sorted detections."""
        if not self._ground_truth or not self._all_cells or not self._slide_spec:
            self._log_status(
                "Calibration requires both detections and ground truth GeoJSON."
            )
            return

        gt_count = len(self._ground_truth)
        gt_pts = np.array(
            [
                [
                    g.centroid_px[0] * self._slide_spec.mpp_x,
                    g.centroid_px[1] * self._slide_spec.mpp_y,
                ]
                for g in self._ground_truth
            ],
            dtype=np.float32,
        )

        sorted_cells = sorted(self._all_cells, key=lambda c: c.confidence, reverse=True)
        pred_pts = np.array(
            [
                [
                    c.centroid_px[0] * self._slide_spec.mpp_x,
                    c.centroid_px[1] * self._slide_spec.mpp_y,
                ]
                for c in sorted_cells
            ],
            dtype=np.float32,
        )

        gt_kdtree = KDTree(gt_pts)
        claimed_gt: set[int] = set()
        is_true_positive: list[bool] = []

        for pt in pred_pts:
            distances, candidates = gt_kdtree.query(pt, k=min(6, gt_count))
            d_list = [float(d) for d in np.atleast_1d(distances)]
            c_list = [int(c) for c in np.atleast_1d(candidates)]

            matched = False
            for d, c in zip(d_list, c_list, strict=False):
                if d <= self._config.centroid_tolerance_um and c not in claimed_gt:
                    claimed_gt.add(c)
                    is_true_positive.append(True)
                    matched = True
                    break
            if not matched:
                is_true_positive.append(False)

        confs = np.array([c.confidence for c in sorted_cells], dtype=np.float32)
        tp_array = np.array(is_true_positive, dtype=bool)

        thresholds = np.linspace(0.10, 0.85, 31)
        best_threshold = self._config.confidence_threshold
        best_f1 = -1.0

        for t in thresholds:
            mask = confs >= t
            if not np.any(mask):
                continue

            active_tp = int(np.sum(tp_array[mask]))
            active_total = int(np.sum(mask))
            active_fp = active_total - active_tp
            active_fn = gt_count - active_tp

            prec = (
                active_tp / (active_tp + active_fp)
                if (active_tp + active_fp) > 0
                else 0.0
            )
            rec = (
                active_tp / (active_tp + active_fn)
                if (active_tp + active_fn) > 0
                else 0.0
            )
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        self._hud_fader.set_confidence(best_threshold)
        self._config = replace(self._config, confidence_threshold=best_threshold)
        self._apply_confidence_filtering()
        self._log_status(
            f"Calibrated: Best F1 = {best_f1 * 100:.1f}% at {round(best_threshold * 100)}% confidence."
        )

    def _apply_confidence_filtering(self) -> None:
        th = self._config.confidence_threshold
        self._visible_cells = [c for c in self._all_cells if c.confidence >= th]
        self._pill_detections.setText(f"Detections: {len(self._visible_cells):,}")

        boxes = [c.bbox_coordinates for c in self._visible_cells]
        self._cell_overlay.set_vector_data(boxes)

        if self._slide_spec:
            self._hotspot = identify_peak_diagnostic_hpf(
                self._visible_cells, self._slide_spec.mpp_x, self._slide_spec.mpp_y
            )
            self._hotspot_overlay.set_diagnostic_summary(self._hotspot)
            self._btn_fab_hotspot.setEnabled(self._hotspot is not None)

            if self._hotspot:
                self._pill_hotspot.setText(f"Peak HPF: {self._hotspot.peak_count} eos")
            else:
                self._pill_hotspot.setText("Peak HPF: —")

        self._update_metrics_view()

    def _focus_diagnostic_hotspot(self) -> None:
        if self._hotspot:
            cx, cy = self._hotspot.center_px
            span = self._hotspot.radius_px * 1.6
            self._viewport.fitInView(
                QRectF(cx - span, cy - span, 2 * span, 2 * span),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def _update_metrics_view(self) -> None:
        if self._hotspot:
            is_pos = self._hotspot.is_eoe_positive
            status_text = "POSITIVE" if is_pos else "NEGATIVE"
            self._lbl_status_badge.setText(status_text)
            self._lbl_status_badge.setProperty(
                "clinicalEvaluation", "positive" if is_pos else "negative"
            )
            self._refresh_styling(self._lbl_status_badge)

            self._lbl_peak_count.setText(f"{self._hotspot.peak_count} eos / HPF")
            self._lbl_peak_density.setText(f"{self._hotspot.density_per_mm2} cells/mm²")
        else:
            self._lbl_status_badge.setText("UNANALYZED")
            self._lbl_status_badge.setProperty("clinicalEvaluation", "none")
            self._refresh_styling(self._lbl_status_badge)
            self._lbl_peak_count.setText("—")
            self._lbl_peak_density.setText("—")

        if self._ground_truth and self._visible_cells and self._slide_spec:
            summary = compute_cellular_performance_metrics(
                self._ground_truth,
                self._visible_cells,
                self._slide_spec.mpp_x,
                self._slide_spec.mpp_y,
            )
            self._lbl_f1.setText(f"{summary.f1_score * 100:.1f}%")
            self._lbl_prec.setText(f"{summary.precision * 100:.1f}%")
            self._lbl_rec.setText(f"{summary.recall * 100:.1f}%")
            self._lbl_dist_err.setText(f"{summary.mean_centroid_error_um:.2f} µm")
        else:
            self._lbl_f1.setText("—")
            self._lbl_prec.setText("—")
            self._lbl_rec.setText("—")
            self._lbl_dist_err.setText("—")

    def _export_geojson(self) -> None:
        if not self._visible_cells:
            QMessageBox.information(
                self, "Export", "No detections available for export."
            )
            return

        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save QuPath GeoJSON",
            "esophascope_annotations.geojson",
            "GeoJSON (*.geojson)",
        )
        if dest:
            serialize_qupath_geojson(dest, self._visible_cells, self._hotspot)

    def _is_worker_busy(self) -> bool:
        return self._state in (
            WorkbenchOperationalState.EXECUTING,
            WorkbenchOperationalState.CANCELLING,
        )

    def _log_status(self, message: str) -> None:
        """Appends timestamped diagnostic message to the bottom console log."""
        timestamp = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            .astimezone()
            .strftime("%H:%M:%S")
        )
        self._lbl_status_msg.setText(f"[{timestamp}] {message}")

    def _transition_state(
        self, state: WorkbenchOperationalState, message: str | None = None
    ) -> None:
        self._state = state
        self._pill_state.setText(f"● {state.value.upper()}")
        self._pill_state.setProperty("state", state.value)
        self._refresh_styling(self._pill_state)

        if message:
            self._log_status(message)
        self._sync_control_states()

    def _sync_control_states(self) -> None:
        busy = self._is_worker_busy()
        loading = self._is_loading()
        has_slide = self._slide_spec is not None

        # Opening stays enabled while loading: a newer request supersedes the old.
        self._btn_open_slide.setEnabled(not busy)
        self._btn_open_gt.setEnabled(not busy)
        self._btn_export_geojson.setEnabled(
            has_slide and not busy and not loading and len(self._visible_cells) > 0
        )

        if self._state == WorkbenchOperationalState.EXECUTING:
            self._btn_run_cancel.setText("Cancel Inference")
            self._btn_run_cancel.setProperty("role", "critical")
            self._btn_run_cancel.setIcon(self._icon_stop)
            self._btn_run_cancel.setEnabled(True)
        elif self._state == WorkbenchOperationalState.CANCELLING:
            self._btn_run_cancel.setText("Cancelling...")
            self._btn_run_cancel.setEnabled(False)
        else:
            self._btn_run_cancel.setText("Run Tiled Inference")
            self._btn_run_cancel.setProperty("role", "primary")
            self._btn_run_cancel.setIcon(self._icon_play)
            self._btn_run_cancel.setEnabled(has_slide and not loading)

        self._refresh_styling(self._btn_run_cancel)

    def _handle_dropped_files(self, paths: list[str]) -> None:
        if self._is_worker_busy():
            return
        slides = [p for p in paths if Path(p).suffix.lower() in SLIDE_EXTENSIONS]
        if slides:
            self._load_slide(slides[-1])  # only the last dropped slide is shown
        for p in paths:
            if Path(p).suffix.lower() in ANNOTATION_EXTENSIONS:
                self._load_gt(p)

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if dropped_local_paths(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e: QDragMoveEvent) -> None:
        if dropped_local_paths(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e: QDropEvent) -> None:
        paths = dropped_local_paths(e.mimeData())
        if paths:
            self._handle_dropped_files(paths)
            e.acceptProposedAction()
        else:
            e.ignore()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._is_worker_busy() or self._load_tasks:
            # Defer until background work has stopped and released its handles.
            self._close_pending = True
            self._cancel_inference()
            self._cancel_pending_loads()
            self._log_status("Closing once background work has stopped...")
            event.ignore()
            return
        thread = self._inference_thread
        if thread is not None:
            thread.quit()
            thread.wait(5000)
        self._models.shutdown()
        if self._canvas_item:
            self._canvas_item.close(wait=True)
            self._canvas_item = None
        event.accept()


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def announce_desktop_file_name(app: QApplication) -> None:
    """Sets the Linux desktop-file ID only when a matching desktop entry exists.

    Qt registers this ID with xdg-desktop-portal; without an installed
    '<id>.desktop' the portal rejects it ("App info not found"). Claiming an ID
    that does not exist achieves nothing, so it is only announced when real.
    """
    if not sys.platform.startswith("linux"):
        return
    entry = QStandardPaths.locate(
        QStandardPaths.StandardLocation.ApplicationsLocation, f"{APP_ID}.desktop"
    )
    if entry:
        app.setDesktopFileName(APP_ID)
    else:
        logger.info(
            "No '%s.desktop' entry installed; not announcing a desktop-file ID "
            "(window class falls back to the interpreter name on Wayland).",
            APP_ID,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EsophaScope: Digital pathology workstation for esophageal biopsy analysis."
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        default=str(resource_path(DEFAULT_MODEL_WEIGHTS)),
        help="Path to the exported YOLO detector (*.onnx)",
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        # Own AppUserModelID so the taskbar shows our icon, not python.exe's.
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)

    app = QApplication(sys.argv)
    app.setApplicationName("EsophaScope")
    announce_desktop_file_name(app)
    app.setWindowIcon(load_app_icon())
    app.setStyleSheet(build_workbench_stylesheet())
    workbench = EsophaScopeWindow(EsophaScopeConfig(model_weights_path=args.model_path))
    workbench.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""EsophaScope: Quantitative Whole-Slide Image Analysis & Hotspot Localization Suite.

Clinical pathology workstation for esophageal biopsy screening, automated
eosinophil detection, diagnostic High-Power Field (HPF) quantification, and
bidirectional QuPath GeoJSON synchronization.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import math
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openslide
import torch
from PIL import Image
from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QRunnable,
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
    QDragMoveEvent,
    QDropEvent,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QShortcut,
    QWheelEvent,
)
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
    QStyle,
    QStyleOptionGraphicsItem,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial import KDTree

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger("esophascope")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ==============================================================================
# CLINICAL BENCHMARKS & PIPELINE CONSTANTS
# ==============================================================================
DEFAULT_MODEL_WEIGHTS = "./best.pt"
TARGET_MODEL_MPP = 0.173
PATCH_PIXEL_DIMENSION = 640
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
PATCH_STRIDE_OVERLAP_RATIO = 0.20
TISSUE_SATURATION_FLOOR = 25
TISSUE_COVERAGE_MINIMUM_FRACTION = 0.05
OVERVIEW_THUMBNAIL_MAX_DIMENSION = 2048
INFERENCE_BATCH_CAPACITY = 8
PYRAMIDAL_TILE_EDGE_PX = 640
MAX_CONCURRENT_TILE_TASKS = 4

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
    tissue_sat_threshold: int = TISSUE_SATURATION_FLOOR
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
def apply_slide_wide_nms(
    boxes: np.ndarray, scores: np.ndarray, iou_thresh: float
) -> list[int]:
    """Slide-wide non-maximum suppression using TorchVision or vectorized NumPy."""
    if len(boxes) == 0:
        return []

    try:
        import torchvision.ops

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        scores_t = torch.as_tensor(scores, dtype=torch.float32)
        if torch.cuda.is_available():
            boxes_t = boxes_t.cuda()
            scores_t = scores_t.cuda()
        return torchvision.ops.nms(boxes_t, scores_t, iou_thresh).cpu().tolist()
    except (ImportError, RuntimeError, TypeError):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        survivors: list[int] = []

        while order.size > 0:
            idx = order[0]
            survivors.append(int(idx))
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[idx], x1[rest])
            yy1 = np.maximum(y1[idx], y1[rest])
            xx2 = np.minimum(x2[idx], x2[rest])
            yy2 = np.minimum(y2[idx], y2[rest])

            intersection_w = np.maximum(0.0, xx2 - xx1)
            intersection_h = np.maximum(0.0, yy2 - yy1)
            intersection_area = intersection_w * intersection_h

            union_area = areas[idx] + areas[rest] - intersection_area + 1e-7
            overlap_ratio = intersection_area / union_area
            viable = np.where(overlap_ratio <= iou_thresh)[0]
            order = rest[viable]

        return survivors


def generate_tissue_mask_hsv(
    thumbnail_rgb: np.ndarray, saturation_threshold: int = 25
) -> np.ndarray:
    """Computes binary tissue mask from whole-slide overview using HSV thresholding."""
    hsv_image = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv_image[:, :, 1]
    brightness = hsv_image[:, :, 2]
    binary_tissue = ((saturation >= saturation_threshold) & (brightness <= 245)).astype(
        np.uint8
    ) * 255
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(
        binary_tissue, cv2.MORPH_CLOSE, structuring_element, iterations=2
    )


def build_tissue_overlay_image(mask_binary: np.ndarray) -> Image.Image:
    """Creates a transparent RGBA architectural overlay map for the tissue layer."""
    h, w = mask_binary.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_binary > 0] = [37, 99, 235, 75]
    return Image.fromarray(rgba, mode="RGBA")


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
    with open(geojson_path, "r", encoding="utf-8") as f:
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
# SECTION 3: THREAD-SAFE TILE STREAMING & PYRAMIDAL CANVAS GRAPHICS
# ==============================================================================
class ThreadLocalSlideProvider:
    """Manages thread-local OpenSlide instances to eliminate per-tile open/close overhead."""

    def __init__(self, file_path: str):
        self._path = file_path
        self._thread_local = threading.local()
        self._open_handles: list[openslide.OpenSlide] = []
        self._lock = threading.Lock()

    def obtain_handle(self) -> openslide.OpenSlide:
        slide = getattr(self._thread_local, "handle", None)
        if slide is None:
            slide = openslide.OpenSlide(self._path)
            self._thread_local.handle = slide
            with self._lock:
                self._open_handles.append(slide)
        return slide

    def shutdown(self) -> None:
        with self._lock:
            for handle in self._open_handles:
                try:
                    handle.close()
                except (openslide.OpenSlideError, OSError):
                    pass
            self._open_handles.clear()


class PyramidalTileSignals(QObject):
    tile_ready = Signal(int, int, int, QImage)
    tile_failed = Signal(int, int, int)


class PyramidalTileFetchTask(QRunnable):
    """Fetches and decodes OpenSlide pyramid tiles in parallel thread-pool workers."""

    def __init__(
        self,
        provider: ThreadLocalSlideProvider,
        level: int,
        tile_x: int,
        tile_y: int,
        tile_w: int,
        tile_h: int,
        read_w: int,
        read_h: int,
        signals: PyramidalTileSignals,
    ):
        super().__init__()
        self._provider = provider
        self._level = level
        self._tx = tile_x
        self._ty = tile_y
        self._tw = tile_w
        self._th = tile_h
        self._rw = read_w
        self._rh = read_h
        self._signals = signals

    def run(self) -> None:
        try:
            slide = self._provider.obtain_handle()
            tile_rgba = slide.read_region(
                (self._tx, self._ty), self._level, (self._rw, self._rh)
            )
            image = QImage(
                tile_rgba.tobytes(),
                self._rw,
                self._rh,
                4 * self._rw,
                QImage.Format.Format_RGBA8888,
            ).copy()
            self._signals.tile_ready.emit(self._level, self._tx, self._ty, image)
        except (openslide.OpenSlideError, OSError, RuntimeError, ValueError) as exc:
            logger.debug(
                "Tile streaming fault at level %d (%d, %d): %s",
                self._level,
                self._tx,
                self._ty,
                exc,
            )
            self._signals.tile_failed.emit(self._level, self._tx, self._ty)


class PyramidalSlideCanvas(QGraphicsItem):
    """Interactive multi-resolution canvas item with true LRU tile caching."""

    def __init__(
        self,
        slide_path: str,
        overview_pixmap: QPixmap,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)
        self._slide_path = slide_path
        self._overview_pixmap = overview_pixmap
        self._provider = ThreadLocalSlideProvider(slide_path)

        inspection_handle = openslide.OpenSlide(slide_path)
        self._dimensions = inspection_handle.dimensions
        self._downsamples = list(inspection_handle.level_downsamples)
        inspection_handle.close()

        self._lru_tile_cache: OrderedDict[tuple[int, int, int], QPixmap] = OrderedDict()
        self._active_requests: set[tuple[int, int, int]] = set()
        self._cache_capacity = 128

        self.signals = PyramidalTileSignals()
        self.signals.tile_ready.connect(
            self._on_tile_ready, Qt.ConnectionType.QueuedConnection
        )
        self.signals.tile_failed.connect(
            self._on_tile_failed, Qt.ConnectionType.QueuedConnection
        )
        self._thread_pool = QThreadPool()
        self._thread_pool.setMaxThreadCount(MAX_CONCURRENT_TILE_TASKS)
        self.setZValue(0.0)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._dimensions[0], self._dimensions[1])

    @Slot(int, int, int, QImage)
    def _on_tile_ready(self, level: int, tx: int, ty: int, image: QImage) -> None:
        tile_key = (level, tx, ty)
        self._active_requests.discard(tile_key)
        pix = QPixmap.fromImage(image)

        if len(self._lru_tile_cache) >= self._cache_capacity:
            self._lru_tile_cache.popitem(last=False)

        self._lru_tile_cache[tile_key] = pix
        self.update(QRectF(tx, ty, PYRAMIDAL_TILE_EDGE_PX, PYRAMIDAL_TILE_EDGE_PX))

    @Slot(int, int, int)
    def _on_tile_failed(self, level: int, tx: int, ty: int) -> None:
        self._active_requests.discard((level, tx, ty))

    def close(self) -> None:
        self._thread_pool.clear()
        self._thread_pool.waitForDone(500)
        self._lru_tile_cache.clear()
        self._active_requests.clear()
        self._provider.shutdown()

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        visible_area = option.exposedRect.intersected(self.boundingRect())
        if visible_area.isEmpty():
            return

        # Paint scaled overview thumbnail without pre-allocating an enormous gigapixel pixmap
        src_w = float(self._overview_pixmap.width())
        src_h = float(self._overview_pixmap.height())
        scale_x = src_w / float(self._dimensions[0])
        scale_y = src_h / float(self._dimensions[1])

        thumb_src_rect = QRectF(
            visible_area.left() * scale_x,
            visible_area.top() * scale_y,
            visible_area.width() * scale_x,
            visible_area.height() * scale_y,
        )
        painter.drawPixmap(visible_area, self._overview_pixmap, thumb_src_rect)

        scale_factor = painter.worldTransform().m11()
        if scale_factor < 0.08:
            return

        inverse_zoom = 1.0 / scale_factor
        pyramid_level = 0
        for lvl_idx, ds in enumerate(self._downsamples):
            if ds <= inverse_zoom:
                pyramid_level = lvl_idx
            else:
                break

        level_ds = self._downsamples[pyramid_level]
        start_x = (
            math.floor(visible_area.left() / PYRAMIDAL_TILE_EDGE_PX)
            * PYRAMIDAL_TILE_EDGE_PX
        )
        end_x = (
            math.ceil(visible_area.right() / PYRAMIDAL_TILE_EDGE_PX)
            * PYRAMIDAL_TILE_EDGE_PX
        )
        start_y = (
            math.floor(visible_area.top() / PYRAMIDAL_TILE_EDGE_PX)
            * PYRAMIDAL_TILE_EDGE_PX
        )
        end_y = (
            math.ceil(visible_area.bottom() / PYRAMIDAL_TILE_EDGE_PX)
            * PYRAMIDAL_TILE_EDGE_PX
        )

        for tx in range(start_x, end_x, PYRAMIDAL_TILE_EDGE_PX):
            for ty in range(start_y, end_y, PYRAMIDAL_TILE_EDGE_PX):
                edge_w = min(PYRAMIDAL_TILE_EDGE_PX, self._dimensions[0] - tx)
                edge_h = min(PYRAMIDAL_TILE_EDGE_PX, self._dimensions[1] - ty)
                if edge_w <= 0 or edge_h <= 0:
                    continue

                tile_key = (pyramid_level, tx, ty)
                cached_tile = self._lru_tile_cache.get(tile_key)

                if cached_tile is not None:
                    self._lru_tile_cache.move_to_end(tile_key)
                    painter.drawPixmap(
                        QRectF(tx, ty, edge_w, edge_h),
                        cached_tile,
                        QRectF(0, 0, cached_tile.width(), cached_tile.height()),
                    )
                else:
                    if (
                        tile_key not in self._active_requests
                        and len(self._active_requests) < 32
                    ):
                        self._active_requests.add(tile_key)
                        task = PyramidalTileFetchTask(
                            self._provider,
                            pyramid_level,
                            tx,
                            ty,
                            edge_w,
                            edge_h,
                            round(edge_w / level_ds),
                            round(edge_h / level_ds),
                            self.signals,
                        )
                        self._thread_pool.start(task)


# ==============================================================================
# SECTION 4: VECTORIZED VIEWPORT OVERLAYS
# ==============================================================================
class FastVectorOverlay(QGraphicsItem):
    """Vectorized overlay item using NumPy coordinate filtering and batched painting."""

    def __init__(
        self, color_hex: str, z_index: float, parent: QGraphicsItem | None = None
    ):
        super().__init__(parent)
        self.setZValue(z_index)
        self._pen = QPen(QColor(color_hex), 2)
        self._pen.setCosmetic(True)

        self._boxes = np.empty((0, 4), dtype=np.float32)
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
            min_x = float(np.min(self._boxes[:, 0]))
            min_y = float(np.min(self._boxes[:, 1]))
            max_x = float(np.max(self._boxes[:, 2]))
            max_y = float(np.max(self._boxes[:, 3]))
            unified_rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
        else:
            self._boxes = np.empty((0, 4), dtype=np.float32)

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
            left = exposed.left()
            top = exposed.top()
            right = exposed.right()
            bottom = exposed.bottom()

            visible_mask = (
                (self._boxes[:, 0] <= right)
                & (self._boxes[:, 2] >= left)
                & (self._boxes[:, 1] <= bottom)
                & (self._boxes[:, 3] >= top)
            )
            visible_boxes = self._boxes[visible_mask]

            if visible_boxes.size > 0:
                rectangles = [
                    QRectF(row[0], row[1], row[2] - row[0], row[3] - row[1])
                    for row in visible_boxes
                ]
                painter.drawRects(rectangles)

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
# SECTION 5: ASYNC INFERENCE WORKER
# ==============================================================================
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
    ):
        super().__init__()
        self._slide_path = slide_path
        self._config = config
        self._spec = spec
        self._tissue_mask = tissue_mask
        self._cancel_signal = threading.Event()

    def cancel(self) -> None:
        self._cancel_signal.set()

    def run(self) -> None:
        slide_handle = None
        try:
            if YOLO is None:
                raise ImportError(
                    "Ultralytics YOLO is required: pip install ultralytics"
                )

            self.status_updated.emit(
                "Initializing neural network on computing device..."
            )
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            detector = YOLO(self._config.model_weights_path)

            slide_handle = openslide.OpenSlide(self._slide_path)
            full_w, full_h = self._spec.width_px, self._spec.height_px

            patch_w_px = round(
                PATCH_PIXEL_DIMENSION * (TARGET_MODEL_MPP / self._spec.mpp_x)
            )
            patch_h_px = round(
                PATCH_PIXEL_DIMENSION * (TARGET_MODEL_MPP / self._spec.mpp_y)
            )
            stride_x = round(patch_w_px * (1.0 - self._config.patch_overlap_ratio))
            stride_y = round(patch_h_px * (1.0 - self._config.patch_overlap_ratio))

            mask_h, mask_w = self._tissue_mask.shape
            scale_x = mask_w / float(full_w)
            scale_y = mask_h / float(full_h)

            scheduled_patches: list[tuple[int, int, int, int]] = []
            for py in range(0, full_h, stride_y):
                for px in range(0, full_w, stride_x):
                    pw = min(patch_w_px, full_w - px)
                    ph = min(patch_h_px, full_h - py)
                    if pw < patch_w_px // 2 or ph < patch_h_px // 2:
                        continue

                    mx1 = int(px * scale_x)
                    my1 = int(py * scale_y)
                    mx2 = max(mx1 + 1, int((px + pw) * scale_x))
                    my2 = max(my1 + 1, int((py + ph) * scale_y))

                    patch_region = self._tissue_mask[my1:my2, mx1:mx2]
                    if patch_region.size > 0:
                        coverage = np.count_nonzero(patch_region) / float(
                            patch_region.size
                        )
                        if coverage >= self._config.tissue_min_coverage:
                            scheduled_patches.append((px, py, pw, ph))

            total_patches = len(scheduled_patches)
            self.status_updated.emit(
                f"Evaluating {total_patches:,} tissue patches with YOLO..."
            )
            self.progress_updated.emit(0, total_patches, 0)

            collected_boxes: list[list[float]] = []
            collected_scores: list[float] = []
            batch_images: list[Image.Image] = []
            batch_metadata: list[tuple[int, int, int, int]] = []

            with torch.inference_mode():
                for idx, (px, py, pw, ph) in enumerate(scheduled_patches):
                    if self._cancel_signal.is_set():
                        self.inference_cancelled.emit()
                        return

                    tile_rgba = slide_handle.read_region((px, py), 0, (pw, ph))
                    tile_rgb = tile_rgba.convert("RGB")
                    tile_rgba.close()

                    if pw != PATCH_PIXEL_DIMENSION or ph != PATCH_PIXEL_DIMENSION:
                        tile_resized = tile_rgb.resize(
                            (PATCH_PIXEL_DIMENSION, PATCH_PIXEL_DIMENSION),
                            Image.Resampling.BILINEAR,
                        )
                        tile_rgb.close()
                        tile_rgb = tile_resized

                    batch_images.append(tile_rgb)
                    batch_metadata.append((px, py, pw, ph))

                    if (
                        len(batch_images) == self._config.batch_size
                        or idx == total_patches - 1
                    ):
                        results = detector.predict(
                            source=batch_images,
                            conf=BASE_TENSOR_EXTRACTION_CONFIDENCE,
                            device=target_device,
                            verbose=False,
                        )

                        for (cx, cy, cw, ch), patch_res in zip(
                            batch_metadata, results, strict=False
                        ):
                            boxes_wrapper: Any = getattr(patch_res, "boxes", None)
                            if boxes_wrapper is None:
                                continue

                            tensor_xyxy = getattr(boxes_wrapper, "xyxy", None)
                            tensor_conf = getattr(boxes_wrapper, "conf", None)
                            if tensor_xyxy is None or tensor_conf is None:
                                continue

                            boxes_arr = tensor_xyxy.detach().cpu().numpy()
                            scores_arr = tensor_conf.detach().cpu().numpy()

                            sx = cw / float(PATCH_PIXEL_DIMENSION)
                            sy = ch / float(PATCH_PIXEL_DIMENSION)

                            for b, s in zip(boxes_arr, scores_arr, strict=False):
                                collected_boxes.append(
                                    [
                                        cx + b[0] * sx,
                                        cy + b[1] * sy,
                                        cx + b[2] * sx,
                                        cy + b[3] * sy,
                                    ]
                                )
                                collected_scores.append(float(s))

                        for img in batch_images:
                            img.close()
                        batch_images.clear()
                        batch_metadata.clear()

                        # Emit throttled progress update
                        if (idx % 16 == 0) or (idx == total_patches - 1):
                            self.progress_updated.emit(
                                idx + 1, total_patches, len(collected_boxes)
                            )

                        # Clean VRAM every 64 batches to protect memory
                        if (
                            idx % (self._config.batch_size * 4) == 0
                            and torch.cuda.is_available()
                        ):
                            torch.cuda.empty_cache()

            self.status_updated.emit("Applying slide-level NMS merge...")
            boxes_np = np.array(collected_boxes, dtype=np.float32)
            scores_np = np.array(collected_scores, dtype=np.float32)
            del collected_boxes, collected_scores
            gc.collect()

            retained_indices = apply_slide_wide_nms(
                boxes_np, scores_np, self._config.nms_iou_threshold
            )

            cells: list[EosinophilCell] = []
            for cell_id, k in enumerate(retained_indices, start=1):
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

            del boxes_np, scores_np, retained_indices
            gc.collect()

            hotspot = identify_peak_diagnostic_hpf(
                cells, self._spec.mpp_x, self._spec.mpp_y
            )
            self.inference_finished.emit(cells, hotspot)

        except (
            RuntimeError,
            ValueError,
            OSError,
            openslide.OpenSlideError,
            TypeError,
        ) as exc:
            logger.exception("EsophaScope inference error")
            self.inference_failed.emit(str(exc))
        finally:
            if slide_handle:
                slide_handle.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.worker_finished.emit()


# ==============================================================================
# SECTION 6: WORKBENCH VIEWPORT & MAIN WINDOW
# ==============================================================================
class WorkbenchOperationalState(Enum):
    READY = "ready"
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

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
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

        self._setup_ui()
        self._setup_shortcuts()
        self._sync_control_states()

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

        self._btn_toggle_sidebar = QPushButton("‹")
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
        self._btn_fab_hotspot = QPushButton("+", self._viewport)
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
        widget_style = self.style()

        actions = QHBoxLayout()
        actions.setSpacing(10)

        self._btn_open_slide = QPushButton("Open WSI Slide")
        self._btn_open_slide.setIcon(
            widget_style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton)
        )
        self._btn_open_slide.clicked.connect(self._prompt_open_slide)
        actions.addWidget(self._btn_open_slide)

        self._btn_open_gt = QPushButton("Load GeoJSON")
        self._btn_open_gt.setIcon(
            widget_style.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        )
        self._btn_open_gt.clicked.connect(self._prompt_open_gt)
        actions.addWidget(self._btn_open_gt)

        actions.addWidget(self._create_divider())

        self._btn_run_cancel = QPushButton("Run Tiled Inference")
        self._btn_run_cancel.setProperty("role", "primary")
        self._btn_run_cancel.setIcon(
            widget_style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self._btn_run_cancel.clicked.connect(self._on_run_or_cancel_clicked)
        actions.addWidget(self._btn_run_cancel)

        actions.addWidget(self._create_divider())

        self._btn_export_geojson = QPushButton("Export QuPath GeoJSON")
        self._btn_export_geojson.setIcon(
            widget_style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
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

        self._pill_slide = QLabel("📁 No slide loaded")
        self._pill_slide.setObjectName("hudInfoPill")

        self._pill_detections = QLabel("🔬 Detections: 0")
        self._pill_detections.setObjectName("hudInfoPill")
        self._pill_detections.setProperty("numeric", "true")

        self._pill_hotspot = QLabel("🔥 Peak HPF: —")
        self._pill_hotspot.setObjectName("hudInfoPill")

        ribbon.addWidget(self._pill_state)
        ribbon.addWidget(self._pill_slide)
        ribbon.addWidget(self._pill_detections)
        ribbon.addWidget(self._pill_hotspot)
        ribbon.addStretch(1)

        device_identity = (
            f"CUDA ({torch.cuda.get_device_name(0)})"
            if torch.cuda.is_available()
            else "CPU Mode"
        )
        self._pill_hardware = QLabel(f"⚡ {device_identity}")
        self._pill_hardware.setObjectName("hudHardwarePill")
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
        self._btn_toggle_sidebar.setText("›" if visible else "‹")

    def _reset_workspace(self) -> None:
        """Clears all loaded slides, annotations, and UI state."""
        if self._is_worker_busy():
            QMessageBox.warning(
                self,
                "EsophaScope Busy",
                "Please cancel inference before resetting workspace.",
            )
            return

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

        self._pill_slide.setText("📁 No slide loaded")
        self._pill_detections.setText("🔬 Detections: 0")
        self._pill_hotspot.setText("🔥 Peak HPF: —")
        self._progress_bar.setValue(0)
        self._lbl_progress_detail.setText("0 / 0 patches")

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

    def _load_slide(self, slide_path: str) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._log_status("Decoding WSI pyramid and generating tissue mask...")
        QApplication.processEvents()

        try:
            if self._canvas_item:
                self._scene.removeItem(self._canvas_item)
                self._canvas_item.close()
                self._canvas_item = None

            slide = openslide.OpenSlide(slide_path)
            w, h = slide.dimensions
            mpp_x = float(
                slide.properties.get(openslide.PROPERTY_NAME_MPP_X, TARGET_MODEL_MPP)
            )
            mpp_y = float(
                slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, TARGET_MODEL_MPP)
            )
            vendor = str(
                slide.properties.get(openslide.PROPERTY_NAME_VENDOR, "Generic Scanner")
            )
            mag = str(slide.properties.get("openslide.objective-power", "40"))

            scale = min(
                self._config.thumbnail_max_dimension / float(w),
                self._config.thumbnail_max_dimension / float(h),
                1.0,
            )
            tw, th = max(1, round(w * scale)), max(1, round(h * scale))
            thumb_pil = slide.get_thumbnail((tw, th)).convert("RGB")
            thumb_np = np.asarray(thumb_pil)

            self._tissue_mask = generate_tissue_mask_hsv(
                thumb_np, self._config.tissue_sat_threshold
            )
            tissue_pixels = int(np.count_nonzero(self._tissue_mask))
            tissue_area_mm2 = (
                tissue_pixels * ((w / tw) * mpp_x) * ((h / th) * mpp_y)
            ) / 1_000_000.0

            self._slide_spec = SlideSpecification(
                slide_path=slide_path,
                file_name=Path(slide_path).name,
                width_px=w,
                height_px=h,
                mpp_x=mpp_x,
                mpp_y=mpp_y,
                scanner_model=vendor,
                magnification=mag,
                level_count=slide.level_count,
                level_dimensions=list(slide.level_dimensions),
                level_downsamples=list(slide.level_downsamples),
                tissue_surface_area_mm2=tissue_area_mm2,
            )
            slide.close()

            thumb_qimg = QImage(
                thumb_np.tobytes(), tw, th, 3 * tw, QImage.Format.Format_RGB888
            )
            overview_pixmap = QPixmap.fromImage(thumb_qimg)

            self._canvas_item = PyramidalSlideCanvas(slide_path, overview_pixmap)
            self._scene.addItem(self._canvas_item)
            self._scene.setSceneRect(QRectF(0, 0, w, h))
            self._viewport.fitInView(
                self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )

            tissue_rgba = build_tissue_overlay_image(self._tissue_mask)
            t_qimg = QImage(
                tissue_rgba.tobytes(), tw, th, 4 * tw, QImage.Format.Format_RGBA8888
            )
            self._tissue_overlay_item.setPixmap(QPixmap.fromImage(t_qimg))
            self._tissue_overlay_item.resetTransform()
            self._tissue_overlay_item.setScale(w / float(tw))
            self._tissue_overlay_item.setVisible(self._chk_tissue.isChecked())

            self._pill_slide.setText(f"📁 {self._slide_spec.file_name}")
            self._lbl_meta_dims.setText(f"{w:,} × {h:,} px")
            self._lbl_meta_mpp.setText(f"{mpp_x:.4f} × {mpp_y:.4f} µm/px")
            self._lbl_meta_vendor.setText(f"{vendor} ({mag}x)")
            self._lbl_meta_tissue.setText(f"{tissue_area_mm2:.2f} mm²")

            self._transition_state(
                WorkbenchOperationalState.READY, "Slide ready for analysis."
            )
        except (openslide.OpenSlideError, OSError, ValueError, RuntimeError) as exc:
            logger.exception("Failed to load slide")
            QMessageBox.critical(self, "OpenSlide Loading Error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self._sync_control_states()

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
        if not self._slide_spec or self._tissue_mask is None or self._is_worker_busy():
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
        )
        self._inference_worker.moveToThread(self._inference_thread)

        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.progress_updated.connect(self._on_progress)
        self._inference_worker.status_updated.connect(self._log_status)
        self._inference_worker.inference_failed.connect(
            lambda msg: QMessageBox.critical(self, "Inference Failed", msg)
        )
        self._inference_worker.inference_finished.connect(self._on_inference_finished)
        self._inference_worker.inference_cancelled.connect(
            lambda: self._transition_state(
                WorkbenchOperationalState.CANCELLED, "Inference cancelled."
            )
        )

        self._inference_worker.worker_finished.connect(self._inference_thread.quit)
        self._inference_worker.worker_finished.connect(
            self._inference_worker.deleteLater
        )
        self._inference_thread.finished.connect(self._inference_thread.deleteLater)

        self._sync_control_states()
        self._inference_thread.start()

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
        self._pill_detections.setText(f"🔬 Detections: {count:,}")
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
        if getattr(self, "_close_pending", False):
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
        self._pill_detections.setText(f"🔬 Detections: {len(self._visible_cells):,}")

        boxes = [c.bbox_coordinates for c in self._visible_cells]
        self._cell_overlay.set_vector_data(boxes)

        if self._slide_spec:
            self._hotspot = identify_peak_diagnostic_hpf(
                self._visible_cells, self._slide_spec.mpp_x, self._slide_spec.mpp_y
            )
            self._hotspot_overlay.set_diagnostic_summary(self._hotspot)
            self._btn_fab_hotspot.setEnabled(self._hotspot is not None)

            if self._hotspot:
                self._pill_hotspot.setText(
                    f"🔥 Peak HPF: {self._hotspot.peak_count} eos"
                )
            else:
                self._pill_hotspot.setText("🔥 Peak HPF: —")

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
        has_slide = self._slide_spec is not None
        widget_style = self.style()

        self._btn_open_slide.setEnabled(not busy)
        self._btn_open_gt.setEnabled(not busy)
        self._btn_export_geojson.setEnabled(
            has_slide and not busy and len(self._visible_cells) > 0
        )

        if self._state == WorkbenchOperationalState.EXECUTING:
            self._btn_run_cancel.setText("Cancel Inference")
            self._btn_run_cancel.setProperty("role", "critical")
            self._btn_run_cancel.setIcon(
                widget_style.standardIcon(QStyle.StandardPixmap.SP_MediaStop)
            )
            self._btn_run_cancel.setEnabled(True)
        elif self._state == WorkbenchOperationalState.CANCELLING:
            self._btn_run_cancel.setText("Cancelling...")
            self._btn_run_cancel.setEnabled(False)
        else:
            self._btn_run_cancel.setText("Run Tiled Inference")
            self._btn_run_cancel.setProperty("role", "primary")
            self._btn_run_cancel.setIcon(
                widget_style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )
            self._btn_run_cancel.setEnabled(has_slide)

        self._refresh_styling(self._btn_run_cancel)

    def _handle_dropped_files(self, paths: list[str]) -> None:
        if self._is_worker_busy():
            return
        for p in paths:
            ext = Path(p).suffix.lower()
            if ext in {".svs", ".ndpi", ".mrxs", ".tif", ".tiff"}:
                self._load_slide(p)
            elif ext in {".geojson", ".json"}:
                self._load_gt(p)

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e: QDragMoveEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e: QDropEvent) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._handle_dropped_files(paths)
            e.acceptProposedAction()
        else:
            e.ignore()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._is_worker_busy():
            self._close_pending = True
            self._cancel_inference()
            event.ignore()
            return
        if self._canvas_item:
            self._canvas_item.close()
        event.accept()


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EsophaScope: Digital pathology workstation for esophageal biopsy analysis."
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        default=DEFAULT_MODEL_WEIGHTS,
        help="Path to trained YOLO detector weights (*.pt)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(build_workbench_stylesheet())
    workbench = EsophaScopeWindow(EsophaScopeConfig(model_weights_path=args.model_path))
    workbench.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

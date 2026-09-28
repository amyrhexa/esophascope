# 🔬 EsophaScope

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/GUI-PySide6-41CD52.svg)](https://wiki.qt.io/Qt_for_Python)
[![Deep Learning](https://img.shields.io/badge/Inference-Ultralytics%20YOLO-111F68.svg)](https://github.com/ultralytics/ultralytics)
[![WSI Engine](https://img.shields.io/badge/WSI-OpenSlide-E53935.svg)](https://openslide.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**EsophaScope** is a digital pathology workbench designed to analyze whole-slide biopsy images (WSIs) for **Eosinophilic Esophagitis (EoE)** diagnosis.

It combines asynchronous gigapixel pyramid streaming, deep learning patch inference, spatial indexing, and automated **Peak High-Power Field (HPF)** hotspot discovery. Built with Python and PySide6, EsophaScope bridges automated cell detection with standard pathology tools like QuPath.

---

## 🌟 Key Highlights

- **Gigapixel Pyramidal Canvas**: Zoom and pan smoothly across gigapixel WSI files (`.svs`, `.ndpi`, `.mrxs`, `.tif`) powered by thread-local OpenSlide handles and an LRU tile cache.
- **Tiled Neural Inference**: Extracts overlapping patches normalized to physical resolution ($0.25\,\mu\text{m/px}$, standard $40\times$ equivalent) and skips blank background glass using HSV tissue masking.
- **Slide-Wide NMS**: De-duplicates cells that straddle patch boundaries via GPU-accelerated TorchVision or vectorized NumPy routines.
- **Automated Clinical Hotspot Discovery**: Finds the diagnostic peak High-Power Field ($0.238\,\text{mm}^2$, $r \approx 275.2\,\mu\text{m}$) using KD-Tree radius queries, calculates cell density, and evaluates EoE diagnostic status ($\ge 15\text{ eos/HPF}$).
- **Instant Confidence Tuning & One-Click F1 Calibration**: Scrub confidence thresholds in real time without re-running model inference, or click **F1** to run an $O(N \log N)$ prefix-match algorithm that snaps to the optimal threshold.
- **Bidirectional QuPath Interoperability**: Import expert GeoJSON annotations for validation and export detection boxes alongside the locked peak HPF field.
- **Sleek Viewport HUD**: On-canvas layer switches, a focus button for the peak HPF hotspot, and a bottom telemetry ribbon with timestamped console logs.

---

## 🏗️ Clinical & Architectural Pipeline

```
 ┌──────────────────────┐
 │   Whole-Slide WSI    │
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐      HSV Thresholding
 │  Tissue Segmentation ├────────────────────────────┐
 └──────────┬───────────┘                            │
            │ Gridded Patches                        ▼
            │ (0.25 µm/px)                 ┌──────────────────┐
            ▼                              │ Background Glass │
 ┌──────────────────────┐                  │ Culling (80% ↓)  │
 │ Tiled YOLO Inference │                  └──────────────────┘
 └──────────┬───────────┘
            │ Bounding Boxes & Confidence
            ▼
 ┌──────────────────────┐
 │    Slide-Wide NMS    │  (TorchVision GPU / NumPy CPU)
 └──────────┬───────────┘
            │ Global Slide Coordinates
            ▼
 ┌──────────────────────┐
 │  KD-Tree HPF Search  │  Standardized 0.238 mm² Field
 └──────────┬───────────┘
            │
    ┌───────┴────────────────────────┐
    ▼                                ▼
┌──────────────────────┐ ┌──────────────────────┐
│  Clinical Flagging   │ │ QuPath GeoJSON Sync  │
│ (≥ 15 eos/HPF? +/-)  │ │ (Cells + Peak Circle)│
└──────────────────────┘ └──────────────────────┘

```

---

## 🖥️ Viewport HUD & User Interface

- **Top-Left Layer HUD**: Toggle individual graphics layers on the fly:
  - 🟦 **Tissue Mask**: Translucent overlay highlighting detected tissue.
  - 🟩 **Ground Truth**: Reference annotations loaded from GeoJSON.
  - 🟥 **Predictions**: Detected eosinophils.
  - 🟧 **Peak HPF Boundary**: The field of maximum cell density with a measurement reticle.
- **Bottom-Left Confidence Fader & F1 Calibrator**:
  - Drag the vertical slider to filter detections by confidence in real time.
  - Click **F1** to evaluate candidate thresholds against loaded ground truth and automatically apply the best one.
- **Hotspot Focus FAB (`+`)**: Anchored to the viewport with configurable offsets. Click it to center and frame the diagnostic peak HPF field.
- **Illuminated Telemetry HUD**: Shows execution state (`READY`, `EXECUTING`, `COMPLETED`), loaded slide filename, live cell counts, hardware mode (CUDA/CPU), and timestamped progress logs.
- **Diagnostics Sidebar (‹ / ›)**: Collapsible drawer revealing peak cell counts, cellular density ($\text{cells/mm}^2$), slide metadata (MPP, scanner, tissue surface area), and validation metrics (Precision, Recall, F1, Mean Displacement Error).

---

## ⚙️ Installation & Setup (ELI5)

Whole-slide images are huge—often several gigabytes with billions of pixels. Normal image viewers crash when opening them. EsophaScope uses a specialized C library called **OpenSlide** to read tiny pieces of the image on demand, and **Python** to run the interface and AI model.

Follow these three steps to set it up:

### Step 1: Install OpenSlide (The Slide Reader Engine)

EsophaScope needs OpenSlide installed on your operating system before Python can use it.

<details open>
<summary><b>Ubuntu / Debian Linux</b></summary>

Open your terminal and run:

```bash
sudo apt update
sudo apt install openslide-tools libopenslide-dev
```

Open your terminal and run:

```bash
sudo dnf install openslide openslide-devel
```

Open your terminal and run:

```bash
brew install openslide
```

1. Go to the [OpenSlide Windows Download Page](https://openslide.org/download/?utm_source=gemini).
2. Download the **Windows binary zip** (choose the 64-bit version, e.g., `openslide-win64-...zip`).
3. Extract the `.zip` file to an easy location, such as `C:\openslide`.
4. Add OpenSlide's `bin` folder to your Windows Path:

- Press the `Windows Key`, type **Environment Variables**, and press **Enter**.
- Click the **Environment Variables...** button near the bottom right.
- Under **System variables**, select the line named **Path** and click **Edit...**.
- Click **New**, paste `C:\openslide\bin` (or wherever you extracted it), and click **OK** on all open windows.

---

### Step 2: Download EsophaScope & Create a Safe Sandbox

A virtual environment (`.venv`) creates an isolated sandbox for EsophaScope so its dependencies do not interfere with other Python software on your computer.

Open your terminal (or Command Prompt / PowerShell on Windows) and run:

```bash
# 1. Clone the project code
git clone [https://github.com/amyrhexa/esophascope.git](https://github.com/amyrhexa/esophascope.git)
cd esophascope

# 2. Create the virtual sandbox (requires Python 3.10 or newer)
python -m venv .venv
```

Now activate your sandbox:

- **Linux / macOS**:

```bash
source .venv/bin/activate
```

- **Windows (Command Prompt)**:

```cmd
.venv\Scripts\activate.bat
```

- **Windows (PowerShell)**:

```powershell
.venv\Scripts\Activate.ps1
```

_(You will see `(.venv)` appear at the beginning of your terminal prompt when activated)._

---

### Step 3: Install the Python Packages

With your virtual environment activated, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the installation by running the help command:

```bash
python esophascope.py --help
```

---

## 🚀 Quick Start

Launch EsophaScope by pointing it to your trained YOLO weights:

```bash
python esophascope.py --model path/to/best.pt
```

_(If you omit `--model`, it looks for `./best.pt` in the current folder by default)._

### Basic Workflow

1. **Load a Slide**: Click **Open WSI Slide** or drag-and-drop any `.svs`, `.ndpi`, `.mrxs`, or `.tif` file directly into the viewport.
2. **Import Annotations (Optional)**: Click **Load GeoJSON** (or drag-and-drop a `.geojson` file) to compare predictions against reference annotations.
3. **Run Detection**: Click **Run Tiled Inference**. Track patch evaluation progress in the bottom status panel.
4. **Locate Hotspot**: Inspect the clinical diagnosis status badge in the sidebar. Click the orange circular **`+`** button to center the viewport directly on the peak HPF field.
5. **Calibrate or Filter**: Move the vertical slider to filter out lower-confidence predictions in real time. If ground truth annotations are loaded, click **F1** to automatically set the optimal confidence threshold.
6. **Export**: Click **Export QuPath GeoJSON** to export predictions and the peak hotspot boundary for QuPath review.

---

## ⌨️ Shortcuts & Navigation

| Key Binding         | Action                                                |
| ------------------- | ----------------------------------------------------- |
| **Left Mouse Drag** | Pan across whole-slide canvas                         |
| **Mouse Wheel**     | Zoom in / Zoom out centered under cursor              |
| **Ctrl + W**        | Reset and clear workspace (slide, caches, and models) |
| **Ctrl + Q**        | Safely quit EsophaScope                               |
| **FAB (`+`)**       | Jump to Peak HPF Hotspot                              |

---

## 🧪 QuPath Interoperability

EsophaScope exports standard GeoJSON FeatureCollections compatible with QuPath:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "esopha_diagnostic_peak_hpf",
      "geometry": {
        "type": "Polygon",
        "coordinates": [...]
      },
      "properties": {
        "objectType": "annotation",
        "classification": { "name": "Peak HPF (24 eos/HPF)", "colorRGB": -16711936 },
        "measurements": [
          { "name": "Peak Eosinophil Count", "value": 24 },
          { "name": "Diagnostic Density (cells/mm²)", "value": 100.8 }
        ]
      }
    }
  ]
}

```

---

## 🏎️ Engineering & Performance Notes

- **Zero-Thrash Pyramidal Decoding**: Avoids opening and closing OpenSlide C pointers on every tile read. Worker threads inside `QThreadPool` reuse persistent, thread-safe handles via `ThreadLocalSlideProvider`.
- **SIMD-Accelerated Vector Culling**: Bounding boxes are stored in continuous float32 NumPy matrices. Viewport boundary intersections are resolved via vectorized array operations before dispatching batched `QPainter.drawRects()` calls.
- **Prefix-Match F1 Optimization**: Sweeping 30+ thresholds does not query the spatial KD-tree 30 times. Predictions are matched once in descending confidence order, allowing instant auto-calibration even with over 50,000 cells.

---

## 🩺 Clinical Disclaimer

_EsophaScope is intended for scientific research and educational investigations in quantitative histopathology and computer vision. It is not an FDA-cleared diagnostic medical device. Clinical diagnostic decisions must always be made by a qualified pathologist._

---

## 📜 License

Distributed under the **Apache 2.0 License**. See [`LICENSE`](./LICENSE) for complete details.

# Histopath CRC Detection

An end-to-end **Computer-Aided Detection (CAD) system** for automated detection of colorectal cancer from H&E stained histopathology images.

---

## Demo

> Upload an H&E patch → Get cancer prediction + Grad-CAM heatmap in real time.

```bash
streamlit run stage5_dashboard/app.py
```

---

## Results

| Metric | Value |
|---|---|
| Model | EfficientNet-B0 |
| Task | Binary classification (cancer vs normal) |
| Test Accuracy | **99.93%** |
| F1 Score | **0.9993** |
| Dataset | LC25000 colon subset (10,000 images) |
| Training | Google Colab T4 GPU, 7 epochs |

---

## Pipeline Architecture

```
Raw H&E Slide
      │
      ▼
┌─────────────────────────┐
│  Stage 1                │
│  Stain Normalization    │  Macenko algorithm (IEEE ISBI 2009)
│  macenko.py             │  Standardizes color across scanners
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 2                │
│  Patch Extraction       │  Sliding window, 224×224 px
│  extract_patches.py     │  Background filtering, WSI support
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 3                │
│  Cancer Detection       │  EfficientNet-B0 fine-tuned
│  Classifier             │  colon_aca vs colon_n
│  99.93% accuracy        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 4                │
│  Cancer Grading         │  3-class severity grading
│  Grading Model          │  Grade 1 / 2 / 3
│  (awaiting dataset)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 5                │
│  Streamlit Dashboard    │  Real-time prediction UI
│  app.py                 │  Grad-CAM heatmap visualization
└─────────────────────────┘
```

---

## Project Structure

```
histopath-crc-detection/
│
├── stage1_stain_normalization/
│   ├── macenko.py                 # Macenko normalizer (MATLAB-faithful port)
│   ├── normalize_dataset.py       # Batch normalization with multiprocessing
│   ├── find_best_reference.py     # Auto-select optimal reference image
│   ├── visualize_normalization.py # Before/after comparison plots
│   └── README.md
│
├── stage2_patch_extraction/
│   ├── extract_patches.py         # PatchExtractor — image + WSI mode
│   ├── filter_background.py       # BackgroundFilter — 3-criteria tissue detection
│   ├── visualize_patches.py       # Grid view + extraction demo
│   └── README.md
│
├── stage3_classification/
│   ├── model.py                   # ColonClassifier (EfficientNet-B0)
│   ├── checkpoints/
│   │   ├── efficientnet_b0_colon.pth   # trained weights
│   │   ├── confusion_matrix.png
│   │   ├── training_curves.png
│   │   └── gradcam_preview.png
│   └── README.md
│
├── stage4_grading/
│   ├── model.py                   # CancerGradingModel (3-class)
│   ├── dataset.py                 # GradingDataset with class balancing
│   ├── train.py                   # Training loop with quadratic kappa
│   ├── evaluate.py                # Evaluation + confusion matrix
│   ├── predict.py                 # Single image grade prediction
│   └── README.md
│
├── stage5_dashboard/
│   ├── app.py                     # Streamlit dashboard
│   ├── gradcam.py                 # Grad-CAM implementation
│   ├── utils.py                   # Model loading + prediction utils
│   └── README.md
│
├── notebooks/
│   └── 03_model_training.ipynb    # Colab training notebook (Stage 3)
│
├── tests/
│   ├── test_normalization.py      # 17 tests — Stage 1
│   └── test_patch_extraction.py   # 18 tests — Stage 2
│
├── data/                          # gitignored — download separately
│   ├── raw/
│   ├── normalized/
│   └── patches/
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ashwajeetgedam12/histopath-crc-detection
cd histopath-crc-detection
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

Download the LC25000 colon subset from Kaggle:
👉 https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

Extract only `colon_image_sets/` to `data/raw/`.

### 5. Download model weights

Download `efficientnet_b0_colon.pth` from:
👉 *(Google Drive link — add after making repo public)*

Place in `stage3_classification/checkpoints/`.

---

## Running Each Stage

### Stage 1 — Stain Normalization

```bash
# Find best reference image automatically
python stage1_stain_normalization/find_best_reference.py \
    --folder data/raw/colon_image_sets/colon_aca \
    --output data/sample/reference.jpeg --ext .jpeg

# Normalize dataset
python stage1_stain_normalization/normalize_dataset.py \
    --input  data/raw/colon_image_sets \
    --output data/normalized \
    --reference data/sample/reference.jpeg \
    --workers 4 --ext .jpeg

# Visualize results
python stage1_stain_normalization/visualize_normalization.py \
    --reference data/sample/reference.jpeg \
    --folder data/raw/colon_image_sets \
    --n 6 --ext .jpeg \
    --output outputs/normalization_comparison.png --histograms
```

### Stage 2 — Patch Extraction

```bash
# Extract patches from images
python stage2_patch_extraction/extract_patches.py \
    --input  data/raw/colon_image_sets \
    --output data/patches \
    --size 224 --stride 224 --ext .jpeg --mode image

# Visualize extraction
python stage2_patch_extraction/visualize_patches.py \
    --source data/raw/colon_image_sets/colon_aca/colonca1.jpeg \
    --output outputs/extraction_demo.png
```

### Stage 3 — Train Classifier (Google Colab)

Open `notebooks/03_model_training.ipynb` in Google Colab with T4 GPU.
Expected training time: ~15 minutes. Expected accuracy: ~99.9%.

### Stage 5 — Run Dashboard

```bash
streamlit run stage5_dashboard/app.py
```

Opens at `http://localhost:8501`

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Stage 1 only
pytest tests/test_normalization.py -v

# Stage 2 only
pytest tests/test_patch_extraction.py -v
```

**Test coverage: 35 tests, 0 failures**

---

## Key References

| Paper | Stage | DOI |
|---|---|---|
| Macenko et al. (2009) — Stain Normalization | Stage 1 | 10.1109/ISBI.2009.5193250 |
| Selvaraju et al. (2017) — Grad-CAM | Stage 5 | 10.1109/ICCV.2017.74 |
| Kather et al. (2018) — NCT-CRC Dataset | Dataset | 10.5281/zenodo.1214456 |
| Borkowski et al. (2019) — LC25000 Dataset | Dataset | arXiv:1912.12142 |
| Srinidhi et al. (2021) — DL in Histopathology | Survey | 10.1109/TMI.2020.3026353 |

---

## Tech Stack

| Category | Technology |
|---|---|
| Deep Learning | PyTorch, timm (EfficientNet-B0) |
| Explainability | Grad-CAM (custom implementation) |
| Image Processing | Pillow, OpenCV, NumPy |
| Dashboard | Streamlit |
| Data | LC25000, NCT-CRC-HE-100K |
| Training | Google Colab T4 GPU |
| Testing | pytest (35 tests) |

---

## Scope & Limitations

This project demonstrates the full CAD pipeline architecture. Current limitations:

- **Stage 4 (Grading):** Code complete, training deferred pending access to a pathologist-graded colorectal cancer dataset (e.g., TCGA-COAD with grade annotations)
- **Dataset:** LC25000 is a balanced, clean research dataset. Clinical performance on real-world data from different scanners may vary
- **Not for clinical use:** This system is for research and demonstration purposes only

---

## Disclaimer

> This system is intended for **research purposes only**.
> It has not been validated for clinical deployment and must not be
> used to make medical decisions. Always consult a qualified pathologist.

---

## Author

**Ashwajeet** · [@ashwajeetgedam12](https://github.com/ashwajeetgedam12)


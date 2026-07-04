<div align="center">

# 🎥 Hierarchical Video Surveillance Pipeline

### A real-time, explainable multi-stage video anomaly detection system

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00CCFF?style=for-the-badge)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> Upload a surveillance video → get **bounding-box tracking**, **action classification**, **GradCAM heatmaps**, and a **natural-language incident report** — all from one unified pipeline.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Pipeline Stages](#-pipeline-stages)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Model Weights](#-model-weights)
- [Running the Project](#-running-the-project)
- [Configuration](#-configuration)
- [Class Labels](#-class-labels)
- [Web Interface](#-web-interface)
- [License](#-license)

---

## 🧭 Overview

This project implements a **5-stage hierarchical inference pipeline** for video surveillance anomaly detection. It is designed to:

- 🔍 **Detect and track** every person in the video using YOLOv8 + ByteTrack
- 🧠 **Classify behaviour** using fine-tuned ViViT (Video Vision Transformer) models — first binary (Normal/Abnormal), then fine-grained (e.g. Fighting, Robbery, Walking)
- 🔥 **Explain decisions** spatially with GradCAM attention heatmaps
- 📝 **Generate human-readable reports** using Llama 3.1 (via Ollama) with structured prompt engineering
- 🌐 **Serve everything** through a clean web interface built with FastAPI + vanilla HTML/CSS

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT VIDEO                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 0 — Detection & Tracking                                 │
│  YOLOv8n (person detector) → ByteTrack → Per-track Action Tubes │
└──────────────────────────┬──────────────────────────────────────┘
                           │  Action Tubes (per-person frame crops)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Binary Classification                                │
│  ViViT (32 frames, 224×224) → Normal / Abnormal                 │
└──────────┬────────────────────────────────────┬─────────────────┘
           │ Normal                             │ Abnormal
           ▼                                   ▼
┌──────────────────────┐           ┌───────────────────────────┐
│  STAGE 2             │           │  STAGE 3                  │
│  Normal Fine-Grained │           │  Abnormal Fine-Grained    │
│  ViViT Classifier    │           │  ViViT Classifier         │
│  Basketball / Biking │           │  Fighting / Robbery /     │
│  Walking / etc.      │           │  Shooting / Assault / etc.│
└──────────┬───────────┘           └──────────┬────────────────┘
           └──────────────┬────────────────────┘
                          │ Predictions + Confidence
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Explainability (GradCAM)                             │
│  Spatial attention heatmaps overlaid on original frames         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5 — LLM Report (Llama 3.1 via Ollama)                    │
│  Structured prompt → Natural-language surveillance report        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Web Dashboard        │
              │   FastAPI + HTML/CSS   │
              └────────────────────────┘
```

---

## 🔬 Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| **0** | `detector.py` · `tracker.py` · `tube_builder.py` | YOLOv8n person detection, ByteTrack multi-object tracking, rolling 64-frame action tube construction per track |
| **1** | `vivit_model.py` · `pipeline.py` | Binary ViViT classifier — routes each track to Normal or Abnormal branch (confidence threshold: 0.60) |
| **2** | `vivit_model.py` · `pipeline.py` | Fine-grained classifier for **Normal** tracks (5 activity classes) |
| **3** | `vivit_model.py` · `pipeline.py` | Fine-grained classifier for **Abnormal** tracks (5 crime-event classes) |
| **4** | `gradcam.py` | Class-Activation Maps via GradCAM — highlights which regions drove the classification |
| **5** | `llm/` | Llama 3.1 structured prompt → paragraph-style incident report with location, confidence, and recommended action |

---

## 📁 Project Structure

```
├── 📄 api_server.py              # FastAPI backend — file upload, SSE progress, results
├── 📄 pipeline.py                # Full pipeline orchestrator (all 5 stages)
├── 📄 config.py                  # Centralised configuration (paths, thresholds, classes)
├── 📄 detector.py                # YOLOv8n person detector wrapper
├── 📄 tracker.py                 # ByteTrack multi-object tracker
├── 📄 tube_builder.py            # Per-track frame buffer + action tube logic
├── 📄 vivit_model.py             # ViViT model definition + inference helpers
├── 📄 visualiser.py              # Annotated video/frame rendering with labels & scores
├── 📄 gradcam.py                 # GradCAM hook registration and heatmap generation
├── 📄 run_final.py               # CLI entry point (non-web usage)
├── 📄 requirements.txt           # Python package dependencies
│
├── 📁 llm/                       # LLM report generation module
│   ├── __init__.py
│   ├── llama_engine.py           # Ollama API client for Llama 3.1
│   ├── prompt_builder.py         # Structures the prompt (system / user / assistant tokens)
│   ├── explanation_generator.py  # Combines predictions → prompt payload
│   └── report_writer.py         # Post-processes and formats the LLM output
│
├── 📁 web/                       # Frontend (pure HTML + CSS, no framework)
│   ├── index.html                # Landing page
│   ├── upload.html               # Video upload interface
│   ├── dashboard.html            # Live results dashboard
│   └── css/main.css              # Stylesheet
│
├── 📁 model/                     # ← Place downloaded .pth weight files here
│   └── .gitkeep
├── 📁 output/                    # Auto-created: processed videos, heatmaps, reports
│   └── .gitkeep
└── 📁 uploads/                   # Auto-created: temporary uploaded video storage
    └── .gitkeep
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Person Detection | [YOLOv8n](https://docs.ultralytics.com) — Ultralytics |
| Multi-Object Tracking | ByteTrack (via `supervision`) |
| Action Classification | [ViViT](https://huggingface.co/google/vivit-b-16x2-kinetics400) — Video Vision Transformer |
| Explainability | GradCAM (gradient-weighted class activation maps) |
| LLM Reports | [Llama 3.1](https://ollama.com/library/llama3.1) via Ollama |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML5 / CSS3 |
| Deep Learning | PyTorch 2.1+ · Torchvision · HuggingFace Transformers |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.9 or higher**
- [Git](https://git-scm.com/)
- [Ollama](https://ollama.com/) (for LLM stage)
- [FFmpeg](https://ffmpeg.org/) (for video writing)
- A CUDA-capable GPU is **strongly recommended** (pipeline runs on CPU too, but much slower)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Make sure your PyTorch build matches your CUDA version.
> Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct install command.

### Step 4 — Install & configure Ollama

```bash
# 1. Download Ollama from https://ollama.com and install it
# 2. Pull the Llama 3.1 model
ollama pull llama3.1

# 3. Start the Ollama server (leave this running in a separate terminal)
ollama serve
```

### Step 5 — Install FFmpeg

```bash
# Windows (via winget)
winget install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## 📦 Model Weights

The three ViViT model weight files are **not stored in this repository** (each ~338 MB). Download them and place them inside the `model/` folder.

| File | Description | Size |
|------|-------------|------|
| `binary.pth` | Stage 1 — Binary classifier (Normal / Abnormal) | ~338 MB |
| `abnormal_vivit_32f.pth` | Stage 3 — Fine-grained abnormal action classifier | ~338 MB |
| `normal_vivit_32f.pth` | Stage 2 — Fine-grained normal action classifier | ~338 MB |

> 📥 **Download:** [Google Drive / HuggingFace — add your link here]

After downloading, your `model/` folder should look like:

```
model/
├── binary.pth
├── abnormal_vivit_32f.pth
└── normal_vivit_32f.pth
```

> **Note:** `yolov8n.pt` (YOLOv8 weights) is downloaded **automatically** by Ultralytics on first run — no manual step needed.

---

## 🚀 Running the Project

### Option A — Web Application *(recommended)*

```bash
python api_server.py
```

Then open your browser at: **[http://localhost:8000](http://localhost:8000)**

The web interface lets you:
- Upload any `.mp4` / `.avi` video
- Watch real-time pipeline progress via Server-Sent Events
- View annotated output video, GradCAM heatmaps, and LLM report

---

### Option B — Command Line

```bash
python run_final.py --input path/to/your/video.mp4
```

Outputs are saved to the `output/` directory.

---

## 🔧 Configuration

All pipeline parameters are controlled from a **single file**: `config.py`. No need to dig into individual modules.

| Setting | Default | Description |
|---------|---------|-------------|
| `DetectorConfig.conf_threshold` | `0.40` | YOLO detection confidence minimum |
| `DetectorConfig.iou_threshold` | `0.45` | YOLO NMS IoU threshold |
| `TubeConfig.min_frames_for_inference` | `32` | Minimum frames before first classification |
| `TubeConfig.inference_stride` | `8` | Classify every N new frames per track |
| `TubeConfig.smoothing_window` | `5` | Temporal label smoothing window size |
| `TubeConfig.missing_detection_tolerance` | `15` | Frames a track can be absent before eviction |
| `InferenceConfig.binary_confidence_threshold` | `0.60` | Below this → label as "Uncertain" |
| `InferenceConfig.device` | `"auto"` | `"auto"` \| `"cuda"` \| `"cpu"` |
| `ViViTConfig.hf_model_name` | `google/vivit-b-16x2-kinetics400` | HuggingFace backbone |

---

## 🏷 Class Labels

### Binary Classes
| Index | Label |
|-------|-------|
| 0 | Normal |
| 1 | Abnormal |

### Normal Fine-Grained Classes (Stage 2)
| Index | Label |
|-------|-------|
| 0 | Basketball |
| 1 | Biking |
| 2 | BoxingPunchingBag |
| 3 | JumpingJack |
| 4 | WalkingWithDog |

### Abnormal Fine-Grained Classes (Stage 3)
| Index | Label |
|-------|-------|
| 0 | Fighting |
| 1 | Shooting |
| 2 | Robbery |
| 3 | Abuse |
| 4 | Assault |

---

## 🌐 Web Interface

The web app consists of three pages served by the FastAPI backend:

| Page | File | Description |
|------|------|-------------|
| **Home** | `web/index.html` | Landing page with project overview |
| **Upload** | `web/upload.html` | Drag-and-drop video upload with live progress tracking |
| **Dashboard** | `web/dashboard.html` | Results view: annotated video, GradCAM heatmaps, LLM report |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ using PyTorch · YOLOv8 · ViViT · Llama 3.1

</div>

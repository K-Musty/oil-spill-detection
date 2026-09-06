# 🛢️ Oil Spill Detection from SAR Imagery

A complete, reproducible pipeline for oil spill detection in Sentinel-1 satellite imagery using deep learning segmentation models.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

This repository provides an end-to-end framework for automated oil spill detection in SAR imagery:

| Component | Description |
|-----------|-------------|
| **Preprocessing** | Parses Pascal VOC XML annotations to binary segmentation masks |
| **Models** | U-Net, DeepLabV3, and FPN with ResNet34 backbone |
| **Optimization** | ONNX conversion and quantization for CPU deployment |
| **Deployment** | Streamlit web application and Docker containerization |

---

## 📊 Dataset

**Eastern Mediterranean Oil Slick Dataset**  
Source: [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.980773)

| Property | Value |
|----------|-------|
| Total patches | 3,655 (256×256) |
| Oil slicks | 1,365 |
| Look-alikes | 2,290 |
| Classes | `oil`, `look-alike`, `land`, `ship`, `sea` |
| Format | JPEG + Pascal VOC XML |

Preprocessed splits:

| Split | Samples | Oil Pixels |
|-------|---------|------------|
| Train | 955 | 4,013,111 |
| Validation | 205 | 631,935 |
| Test | 205 | 916,882 |

---

## 🏗️ Architecture

```
Raw SAR Image (640×640)
        ↓
Preprocessing (resize to 256×256)
        ↓
┌───────┴───────┐
│   U-Net       │
│   DeepLabV3   │
│   FPN         │
└───────┬───────┘
        ↓
Binary Segmentation Mask
        ↓
ONNX Quantization (CPU inference)
```

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/K-Musty/oil-spill-detection.git
cd oil-spill-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run preprocessing
cd src
python preprocess.py

# Train model
python train.py --model unet --epochs 50 --batch_size 8

# Launch web app
streamlit run app/main.py
```

### Docker

```bash
docker build -t oil-spill-detection .
docker run -p 8501:8501 oil-spill-detection
```

---

## 📈 Results

| Model | IoU (%) | Dice (%) | Inference (ms) | Size (MB) |
|-------|---------|----------|----------------|-----------|
| **U-Net (ResNet34)** | 68.5 | 75.2 | 120 | 45 |
| DeepLabV3 (ResNet34) | 67.1 | 73.8 | 145 | 52 |
| FPN (ResNet34) | 65.8 | 72.3 | 130 | 48 |

---

## 📁 Repository Structure

```
oil-spill-detection/
├── src/
│   ├── preprocess.py      # Data preprocessing
│   ├── dataset.py         # PyTorch DataLoader
│   ├── train.py           # Training script
│   └── export_onnx.py     # ONNX conversion
├── app/
│   └── main.py            # Streamlit application
├── data/
│   ├── raw/               # Raw data (ignored by git)
│   └── processed/         # .npz files (included)
├── checkpoints/           # Model weights (ignored by git)
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Data Access

**Raw data is not included** due to size. To obtain it:

1. Download from [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.980773)
2. Extract to `data/raw/`
3. Run `src/preprocess.py`

---

## 🧪 Training

```bash
cd src
python train.py --model unet --epochs 50 --batch_size 8
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `unet` | `unet`, `deeplabv3`, or `fpn` |
| `--epochs` | `50` | Number of epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--encoder` | `resnet34` | Backbone network |

---

## 🙏 Acknowledgments

- **Dataset**: Eastern Mediterranean Oil Slick Dataset (PANGAEA)
- **Framework**: `segmentation-models-pytorch`
- **Compute**: Google Colab (free-tier T4 GPU)

---

## 📧 Contact

**Abdulrahman Kalli Mustapha**  
Email: kmustapha9564@gmail.com  
GitHub: [K-Musty](https://github.com/K-Musty)

---

## 📄 License

MIT License

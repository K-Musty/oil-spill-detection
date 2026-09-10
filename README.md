# 🛢️ Oil Spill Detection from SAR Imagery

A complete, reproducible pipeline for oil spill detection in Sentinel-1 satellite imagery using deep learning segmentation models.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/🤗%20Streamlit-Live%20Demo-brightgreen)](https://oil-spill-detection-app.streamlit.app/)

---

## 📋 Overview

This repository provides an end-to-end framework for automated oil spill detection in SAR imagery:

| Component | Description |
|-----------|-------------|
| **Preprocessing** | Parses Pascal VOC XML annotations to binary segmentation masks |
| **Models** | U-Net, DeepLabV3, and FPN with ResNet34 backbone |
| **Training** | Unified training script with mixed precision and early stopping |
| **Deployment** | Streamlit web application (native PyTorch inference) |

---

## 📊 Dataset

**Eastern Mediterranean Oil Slick Dataset**
Source: [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.980773)

| Property | Value |
|----------|-------|
| Total patches | 3,655 (256×256) |
| Annotated (oil-containing) patches | 1,365 |
| Look-alike patches | 2,290 |
| Classes | `oil`, `look-alike`, `land`, `ship`, `sea` |
| Format | JPEG + Pascal VOC XML |

**Preprocessed splits** (from the 1,365 annotated oil-containing patches):

| Split | Samples | Oil Pixels |
|-------|---------|------------|
| Train | 955 | 4,013,111 |
| Validation | 205 | 631,935 |
| Test | 205 | 916,882 |

> **Note**: Only the 1,365 annotated patches with oil objects were used for segmentation, as pixel-level ground truth is required. The 2,290 look-alike patches were used as negative examples during validation only.

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
Streamlit Web Application (PyTorch inference)
```

All three models use a shared **ResNet-34** backbone pre-trained on ImageNet, imported via `segmentation_models_pytorch`.

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/K-Musty/oil-spill-detection.git
cd oil-spill-detection
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run preprocessing
cd src
python preprocess.py

# Train a model
python train.py --model unet --epochs 50 --batch_size 8

# Launch web app
cd ../app
streamlit run app.py
```

### Live Demo

🔗 **https://oil-spill-detection-app.streamlit.app/**

---

## 📈 Results

Final benchmark results on the independent test split (205 patches):

| Model | Val IoU (%) | Test IoU (%) | Test Dice (%) | Params (M) | Best Epoch |
|-------|-------------|--------------|---------------|------------|------------|
| U-Net | 51.70 | 46.90 | 62.60 | 24.4 | 35 |
| FPN | 51.80 | 46.35 | 62.02 | 23.2 | 10 |
| **DeepLabV3** | **59.51** | **53.76** | **68.12** | **26.0** | **48** |

**DeepLabV3** achieves the best performance due to its **Atrous Spatial Pyramid Pooling (ASPP)** module, which captures multi-scale, sprawling slick geometries.

### Key Findings

- **DeepLabV3 outperforms** U-Net and FPN by ~8 percentage points on validation IoU.
- **U-Net overfits** after epoch 35 (validation IoU declines while training loss continues falling).
- **Data is the bottleneck**: changing architecture provides <8% improvement, confirming that dataset size and quality are the limiting factors.

---

## 📁 Repository Structure

```
oil-spill-detection/
├── src/
│   ├── preprocess.py         # Data preprocessing (XML → binary masks)
│   ├── dataset.py            # PyTorch Dataset + DataLoader
│   ├── train.py              # Unified training script (U-Net, FPN, DeepLabV3)
│   └── train_deeplab.py      # Standalone DeepLabV3 training (512×512 input)
├── app/
│   ├── app.py                # Streamlit application
│   └── best_model.pt         # Trained DeepLabV3 weights (for deployment)
├── data/
│   ├── raw/                  # Raw data (ignored by git)
│   └── processed/            # .npz files (ignored by git, see Data Access)
├── checkpoints/              # Model weights (ignored by git)
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

---

## 📊 Data Access

**Raw data and processed files are not included** in this repository due to size.

### Download Raw Data

1. Download the Eastern Mediterranean Oil Slick Dataset from [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.980773)
2. Extract to `data/raw/`
3. Run `python src/preprocess.py` to generate `data/processed/*.npz`

### Preprocessed Files

The processed `.npz` files (~300 MB total) can be regenerated from the raw data using the preprocessing script. Alternatively, they are available upon request.

---

## 🧪 Training

```bash
cd src

# Train U-Net
python train.py --model unet --epochs 50 --batch_size 8 --lr 1e-4

# Train FPN
python train.py --model fpn --epochs 50 --batch_size 8 --lr 1e-4

# Train DeepLabV3 (requires 512×512 input)
python train_deeplab.py --epochs 50 --batch_size 4 --lr 5e-5
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `unet` | `unet`, `deeplabv3`, or `fpn` |
| `--encoder` | `resnet34` | Backbone network |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--patience` | `10` | Early stopping patience |

---

## ⚠️ Note on ONNX Export

We initially attempted to export the trained models to ONNX format for lightweight CPU inference. However, the export failed due to dynamic interpolation and bilinear upsampling layers in the decoders, which cannot be dynamically mapped by PyTorch's trace utility. The resulting ONNX file was a broken 0.27 MB stub.

**We therefore deploy the model using native PyTorch weights**, which load directly in the Streamlit application without serialization-induced precision drops.

---

## 🙏 Acknowledgments

- **Dataset**: Eastern Mediterranean Oil Slick Dataset (Yang & Singha, 2025), hosted on PANGAEA
- **Framework**: [`segmentation-models-pytorch`](https://github.com/qubvel/segmentation_models.pytorch)
- **Compute**: Google Colab (free-tier T4 GPU)

---

## 📧 Contact

**Abdulrahman Kalli Mustapha**
- Email: kmustapha9564@gmail.com
- GitHub: [@K-Musty](https://github.com/K-Musty)
- Affiliation: Al-Qalam University, Katsina, Nigeria

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{mustapha2026oilspill,
  author = {Mustapha, Abdulrahman Kalli},
  title = {A Reproducible Pipeline for Oil Spill Detection in SAR Imagery: Comparing U-Net, DeepLabV3, and FPN under Data-Constrained Conditions},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/K-Musty/oil-spill-detection}}
}
```

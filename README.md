# 🌍 Adaptive WGAST: A Weakly-Supervised Generative Framework for High-Resolution Land Surface Temperature Reconstruction

<p align="center">
  <img src="https://img.shields.io/badge/DeepLearning-PyTorch-blue?style=flat-square">
  <img src="https://img.shields.io/badge/Satellite-LST%20Reconstruction-green?style=flat-square">
  <img src="https://img.shields.io/badge/Status-Active-orange?style=flat-square">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square">
</p>

Official repository for the independent research project  
**“Adaptive WGAST: A Weakly-Supervised Generative Framework for High-Resolution Land Surface Temperature Reconstruction”**  
by **Devashish Komiya** and **Venkat Saahit Kamu**  
Department of Computer Science and Engineering,  
**Birla Institute of Technology, Mesra, Ranchi, India**

---

## 📘 Overview

**Adaptive WGAST** is an enhanced deep generative framework for reconstructing **high-resolution Land Surface Temperature (LST)** from coarse satellite observations under **weak supervision**.

It builds upon the original **Weighted Generative Adversarial Spatio-Temporal (WGAST)** model by introducing:
1. 🧠 **Adaptive Denoising Block (ADB)** – A learnable spatially-aware filter that corrects thermal noise adaptively without over-smoothing heterogeneous regions.  
2. 🔄 **Similarity Feature Refinement (SFR)** – A cross-sensor alignment mechanism leveraging cosine similarity to enhance coherence between **Landsat** and **Sentinel** representations.  
3. 📉 **Weakly-Supervised Training** – Enables accurate fine-resolution predictions even with sparse or coarse MODIS labels.

These innovations yield higher reconstruction fidelity while preserving local spatial variability.

---

## 🛰️ Motivation

High-resolution LST is essential for:
- 🌆 **Urban Heat Island (UHI)** analysis and sustainable city planning  
- 🌾 **Agricultural drought** and evapotranspiration modeling  
- 🌍 **Climate resilience** and energy balance studies  

However, satellite sensors suffer from spatial-temporal trade-offs:
- **MODIS** → frequent but coarse (1 km)  
- **Landsat/Sentinel** → fine but infrequent (10–30 m)  

**Adaptive WGAST** bridges this gap by fusing multi-sensor data through a generative adversarial architecture.

---

## ⚙️ Model Architecture

Adaptive WGAST extends WGAST’s generator-decoder backbone with two learnable modules:

### 🔸 Adaptive Denoising Block (ADB)
Learns both a residual map \(R(x)\) and a gating mask \(M(x)\):
\[
Y_{\text{final}} = \hat{Y} + M(x) \odot R(x)
\]
This enables pixel-wise control over denoising intensity.

### 🔸 Similarity Feature Refinement (SFR)
Computes cross-sensor cosine similarity between latent features:
\[
S = \frac{F_L \cdot F_S}{\sqrt{(F_L^2)} \sqrt{(F_S^2)}}
\]
Used to align Landsat and Sentinel features before decoding.

---

## 📊 Results Summary

| Model | MAE (°C) | RMSE (°C) | R² | Corr |
|:------|:---------:|:----------:|:----:|:------:|
| **Original WGAST** | 2.260 | 2.928 | -0.424 | 0.713 |
| **Adaptive WGAST v1** | 2.011 | 2.621 | -0.212 | 0.709 |
| **Adaptive WGAST v2** | **1.849** | **2.454** | **-0.000** | **0.703** |

✅ **16% RMSE reduction** and **18% MAE improvement** over the baseline WGAST.

---

## 🧠 Key Features

- ✅ Weakly-supervised training with coarse MODIS guidance  
- 🧩 Learnable denoising for texture preservation  
- 🌐 Cross-sensor alignment between Landsat and Sentinel features  
- 📈 Quantitative evaluation: MAE, RMSE, R², and Pearson correlation  
- 🛰️ Compatible with multi-mission data (Landsat 8/9, MODIS, Sentinel-2)

---

## 🧪 Dataset and Region of Interest

**Region:** UTM Zone 31N (EPSG:32631)  
**Bounding Box:** `[412850.0, 5299550.0, 424850.0, 5311550.0]`

**Acquisition Dates:**
- Primary: `19 Sep 2024`, `21 Oct 2024`  
- Extended: `18 Jun 2025`, `05 Aug 2025`

**Ground Truth:**  
Generated using **Google Earth Engine (GEE)** via Landsat 8/9 Collection 2 Level-2 Surface Temperature composites.  
Cloud masking applied using `QA_PIXEL`.

---

## 🧩 Repository Structure
WGAST/  
├── data_download/        # Scripts to download MODIS, Landsat, Sentinel data  
├── data_loader/          # Dataset loading and utilities  
├── data_preparation/     # Data normalization and triplet creation  
├── model/                # WGAST + Adaptive WGAST architectures  
├── predict/              # Inference and evaluation scripts  
├── runner/               # Experiment control and training pipeline  
├── tutorials/            # Jupyter notebooks (data → training → evaluation)  
└── README.md             # Project documentation  

---

## 💻 Usage Workflow

1. **Download datasets** → MODIS, Landsat 8/9, and Sentinel-2  
2. **Preprocess data** using triplet generation scripts  
3. **Train Adaptive WGAST model** using `experiment.py`  
4. **Evaluate results** on validation region using `predict2.py`  
5. **Compare outputs** (MAE, RMSE, R², Corr) with baseline WGAST  

---

## 🌏 Applications

- 🌆 **Urban heat island** and microclimate monitoring  
- 🌾 **Agricultural drought** and evapotranspiration mapping  
- ☀️ **Climate change** modeling and environmental policy analysis  
- 🚨 **Disaster risk** and heatwave detection  

---
## 🙏 Acknowledgments
We thank **Google Earth Engine (GEE)**, **NASA**, **USGS**, and **Kaggle** for providing open-access satellite data and GPU resources.

---

<p align="center">
⭐ If you find this repository useful, please consider starring it! ⭐
</p>

# A Clinical Monkeypox Assessment Support System Based On Machine Learning Approach

This repository contains the source code and research materials for the project titled:

**"A Clinical Monkeypox Assessment Support System Based on Machine Learning Approach"**

✅ This work has been published in the proceedings of the **IEEE International Conference on Intelligent and Cloud Computing (ICoICC 2025)**.

🔗 **Published Paper**: [IEEE Xplore Link](https://ieeexplore.ieee.org/document/11051356)

---

## 📌 Abstract

Monkeypox is a re-emerging zoonotic viral disease with symptoms resembling smallpox but generally less severe. This research presents a hybrid deep learning-based clinical decision support system for automatic classification of monkeypox using skin lesion images.

The approach integrates:
- **Convolutional Neural Networks (CNNs)** for spatial feature extraction,
- A custom **Self-Attention mechanism** to enhance focus on significant regions, and
- **Bidirectional Gated Recurrent Units (Bi-GRU)** to capture temporal lesion progression patterns.

This hybrid architecture enhances classification capabilities across various skin lesion categories including Monkeypox, Measles, Chickenpox, Cowpox, HFMD, and Healthy skin.

---

## 🏗️ Methodology Overview

The proposed pipeline includes:

1. **Preprocessing**:
   - Image normalization
   - Data augmentation (e.g., shearing)
   - Histogram equalization for contrast enhancement

2. **Model Architecture**:
   - CNN layers for extracting low- and mid-level spatial features
   - Self-attention layers to prioritize important regions
   - Bi-GRU for modeling temporal dependencies in lesion evolution
   - Dense layers with softmax output for classification

3. **Datasets**:
   - 🖼️ [MSLD (Monkeypox Skin Lesion Dataset)](https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset)
   - 🖼️ [MSLDv2.0](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20)

---

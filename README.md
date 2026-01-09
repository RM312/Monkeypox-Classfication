# 🧠 Monkeypox Classification  
## A Clinical Monkeypox Assessment Support System Based on Machine Learning Approach

---

## 📌 Overview

This repository contains the source code, experimental implementations, and research artifacts for the project entitled:

**A Clinical Monkeypox Assessment Support System Based on Machine Learning Approach**

This work has been **published in the proceedings of the IEEE International Conference on Intelligent and Cloud Computing (ICoICC 2025)**.

📄 **Published Paper (IEEE Xplore):** [https://ieeexplore.ieee.org/document/11051356](https://ieeexplore.ieee.org/document/11051356)

---

## 📖 Abstract

Monkeypox (Mpox) is a re-emerging zoonotic viral disease with symptoms resembling smallpox, though generally less severe. Early and accurate diagnosis is critical for outbreak control and clinical decision-making.

This research proposes a **hybrid deep learning–based clinical decision support system** for automatic classification of monkeypox using skin lesion images. The proposed approach integrates:

* **Convolutional Neural Networks (CNNs)** for spatial feature extraction 
* A **Self-Attention mechanism** to emphasize clinically significant lesion regions 
* **Bidirectional Gated Recurrent Units (Bi-GRU)** to capture sequential feature dependencies 

The hybrid architecture improves classification performance across multiple skin lesion categories, including Monkeypox, Measles, Chickenpox, Cowpox, Hand-Foot-Mouth Disease (HFMD), and Healthy Skin.

---

## 🧠 Methodology Overview

### 1. Preprocessing
* Image resizing and normalization 
* Data augmentation (shearing, rotation, flipping) 
* Histogram equalization for contrast enhancement 

### 2. Model Architecture
* CNN layers for low- and mid-level spatial feature extraction 
* Self-Attention layers to prioritize discriminative lesion regions 
* Bi-GRU layers to model sequential relationships between extracted features 
* Fully connected layers with Softmax activation for multi-class classification 

### 3. Training Strategy
* Multi-class classification framework 
* Categorical cross-entropy loss 
* Adam optimizer 
* Performance evaluation using Accuracy, Precision, Recall, and F1-score 

---

## 🖼️ Datasets Used

The models are trained and evaluated using publicly available benchmark datasets:

* **MSLD (Monkeypox Skin Lesion Dataset)** [https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset](https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset) 

* **MSLDv2.0 (Expanded Monkeypox Dataset)** [https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20) 

---

## 📂 Repository Structure

```text
Monkeypox-Classification/
├── CNN/                # CNN-based baseline classifier
├── LSTM/               # LSTM-based classifier
├── GRU/                # GRU-based classifier
├── Bi-LSTM/            # Bidirectional LSTM model
├── Bi-GRU/             # Bidirectional GRU model
└── Proposed/           # Final Hybrid Model (CNN + Self-Attention + Bi-GRU)
    ├── Training scripts
    ├── Evaluation code
    ├── Saved models
    ├── Performance metrics
    └── Result visualizations

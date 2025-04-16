
#Monkeypox Image Classification Using Self-Attention CNN + Bi-GRU

This repository contains the implementation of a deep learning-based Monkeypox diagnostic support system. The model integrates a Self-Attention-based Convolutional Neural Network (CNN) with a Bidirectional Gated Recurrent Unit (Bi-GRU) for accurate classification of skin lesion images.

Table of Contents

Overview

Architecture

Datasets

Preprocessing

Training and Evaluation

Results

Requirements

Usage

Contributors

License



---

Overview

Monkeypox is a viral zoonotic disease with symptoms similar to smallpox. Early detection is crucial for treatment and containment. This work introduces a hybrid model combining self-attention-enhanced CNN and Bi-GRU to detect Monkeypox from skin lesion images.

Architecture

The proposed model is built with:

CNN Layers for spatial feature extraction

Self-Attention Layers to focus on important image regions

Bi-GRU Layers for temporal context and feature enhancement

Dense Layers with Softmax for final classification


Datasets

Two publicly available datasets were used:

Monkeypox Skin Lesion Dataset (MSLD)

MSLD v2.0


Each dataset includes images of Monkeypox and other similar diseases like Chickenpox, Measles, HFMD, Cowpox, and Healthy skin.

Preprocessing

Normalization of pixel values to [0, 1]

Data Augmentation using shear transformation

Histogram Equalization for contrast enhancement


Training and Evaluation

Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, and Loss


Results

The proposed model achieved 100% accuracy on both MSLD and MSLDv2.0 datasets, outperforming traditional CNNs, LSTMs, GRUs, and even Bi-LSTM and Bi-GRU models.

Requirements

Python 3.8+

TensorFlow 2.x

NumPy

Matplotlib

scikit-learn

OpenCV


Install dependencies using:

pip install -r requirements.txt

Usage

1. Clone the repository


2. Download the datasets from Kaggle and place them in ./data/


3. Run the training script:



python train.py

4. Evaluate the model:



python evaluate.py

Contributors

Pradip Dhal, KIIT University – Email

Rivu Mahata, KIIT University – Email


License

This project is licensed under the MIT License.


---

Let me know if you’d like me to generate the actual requirements.txt, train.py, or any of the scripts referenced.

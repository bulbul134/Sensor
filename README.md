# A Sensor-Independent Hybrid CNN–LSTM Framework for Urban Traffic Congestion Classification: Evidence from Dhaka, Bangladesh

This repository contains the official implementation, architecture configurations, and a benchmark subset of the **TFP-BD dataset** for our sensor-independent hybrid CNN–LSTM traffic congestion classification framework.

---

## 📌 Project Overview
Unlike traditional sensor-dependent architectures that require extensive loop detectors or GPS trajectory graphs, this framework processes **standard traffic-camera image sequences** to perform a three-class categorical congestion classification: **Low (0)**, **Medium (1)**, and **High (2)**. 

### Key Features:
* **Sensor-Independent Topology:** Operates purely on visual camera streams.
* **Spatio-Temporal Integration:** Combines TimeDistributed 2D-CNNs for spatial density encoding with stacked LSTMs for sequence modeling.
* **Imbalance Mitigation:** Built-in inverse-frequency class-weight regularization and targeted minority-class data augmentation.
* **Edge-Ready Efficiency:** Optimized for low-resolution ($64 \times 64 \times 3$) inputs to achieve an inference latency of **11.5 ms per sequence** with only ~1.9M parameters.

---

## 📂 Repository Structure
```text
├── data/
│   └── tfp_bd_subset/          # Stratified subset of the TFP-BD dataset
├── models/
│   └── cnn_lstm_backbone.py    # Proposed CNN-LSTM model architecture
├── utils/
│   ├── data_pipeline.py        # Preprocessing, normalization, and augmentation
│   └── losses.py               # Weighted sparse categorical cross-entropy loss
├── train.py                    # Script to train the framework from scratch
├── evaluate.py                 # Evaluation script to compute F1, Confusion Matrix, etc.
├── requirements.txt            # Required dependencies and libraries
└── README.md                   # This instruction manual

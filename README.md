# Open Set Recognition (OSR) Project

## Overview
This project addresses the **Open Set Recognition (OSR)** problem by combining **embedding space analysis**, **triplet loss learning**, **probability thresholds**, and **density-based clustering** to improve classification accuracy and effectively identify unknown samples.

## Authors
- Lee Ben Gigi
- Ron Gurevich
- Omer Ben Arie

## Methodology
Our method is built on a multi-stage classification pipeline:

### 1. Feature Extraction with Combined Losses
- A **CNN model** is trained on the **MNIST dataset** using **cross-entropy loss** and **triplet loss**.
- **Cross-entropy loss** ensures classification accuracy.
- **Triplet loss** promotes well-separated embeddings in feature space.

### 2. DBSCAN Clustering
- **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** is applied to the embedding space.
- Each class is represented by **two clusters** instead of a single prototype, allowing for intra-class variability.

### 3. Two-Stage Classification
- **Stage 1: Softmax Probability Check**
  - If the highest softmax probability falls below a predefined threshold, the sample is classified as "Unknown."
- **Stage 2: Cluster Distance Check**
  - If a sample is close enough to a cluster centroid, it retains its predicted label; otherwise, it is reclassified as "Unknown."

## Hyperparameters
Key hyperparameters and configurations used:
- **Embedding Dimension:** 64
- **Triplet Loss Margin:** 1.0
- **DBSCAN Parameters:**
  - `eps` = 2.2
  - `min_samples` = 9
- **Classification Thresholds:**
  - **Softmax Probability Threshold:** 0.6
  - **Distance Factor:** 1.0

## Results
The model was evaluated on **MNIST (in-distribution)** and various **out-of-distribution (OOD)** datasets:

| Dataset       | MNIST Accuracy | OOD Accuracy | Total Accuracy |
|--------------|---------------|--------------|---------------|
| **FashionMNIST** | 95.32% | 100.00% | 95.58% |
| **CIFAR10** | 95.32% | 100.00% | 95.58% |
| **SVHN** | 95.32% | 100.00% | 95.58% |
| **USPS** | 95.32% | 98.50% | 95.50% |
| **EMNIST** | 95.32% | 81.50% | 94.54% |

The model achieved **high accuracy on MNIST** while successfully detecting OOD samples.

## Limitations
- The method performs well on **distinct** OOD datasets (e.g., natural images, text).
- It struggles with **similar** distributions, such as **handwritten letters (EMNIST)**.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook final_project_OSR.ipynb
   ```

## Future Work
- Improve handling of **handwritten letters** (EMNIST) as OOD.
- Explore **alternative clustering techniques** to enhance robustness.
- Extend the model to support **real-world applications** beyond MNIST.

## License
This project is open-source and available under the **MIT License**.

---
For any questions, feel free to reach out to the authors!


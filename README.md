# Open Set Recognition (OSR) Project

## Overview
This project addresses the **Open Set Recognition (OSR)** problem, where a model encounters data it has never seen before. OSR is essential in applications such as **fraud detection, medical diagnosis, and security systems**, where correctly identifying unknown instances is crucial.

Our method integrates **embedding space analysis, triplet loss learning, probability thresholds, and density-based clustering** to improve classification accuracy and distinguish between known and unknown samples effectively.

## Methodology
### Feature Extraction with Combined Losses
- We train a **CNN architecture** on MNIST using **cross-entropy (CE) loss** and **triplet loss**.
- CE loss ensures optimal classification accuracy, while triplet loss encourages distinct clustering of known classes in the embedding space.

### DBSCAN Clustering
- We apply **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to detect intra-class variability.
- Each class typically forms **two clusters** in the embedding space, capturing diverse data distributions.

### Two-Stage Classification
1. **Softmax Probability Check:** If the highest probability score is below a set threshold, the sample is classified as "Unknown."
2. **Cluster Distance Check:** If the probability is high enough, we compute the sample’s distance to the closest DBSCAN cluster centroid. If the distance exceeds a threshold, the sample is reclassified as "Unknown."

This approach ensures robustness by distinguishing between ambiguous in-distribution samples and truly out-of-distribution (OOD) samples.

## Installation & Setup
### Running on Google Colab
To run this project on **Google Colab**, follow these steps:
1. Open Google Colab: [Google Colab](https://colab.research.google.com/)
2. Upload the `final_project_OSR.ipynb` notebook.
3. Run the notebook in Colab without needing a local Python setup.

Alternatively, follow the local setup instructions below.

### Prerequisites
- Python 3.x
- Jupyter Notebook
- PyTorch, NumPy, Matplotlib, Scikit-learn

### Local Setup Instructions
1. Clone the repository:
```sh
git clone https://github.com/your-username/OSR-Project.git
cd OSR-Project
```
2. Install dependencies:
```sh
pip install -r requirements.txt
```
3. Run the Jupyter notebook:
```sh
jupyter notebook final_project_OSR.ipynb
```

## Dataset Information
- We use the **MNIST dataset** for training and testing.
- The dataset is split into:
  - **Known classes:** Used for training.
  - **Unknown classes:** Used for evaluating OSR performance.
- The model was tested on additional OOD datasets, including **FashionMNIST, CIFAR-10, SVHN, USPS, and EMNIST**.
- To download the dataset manually, use:
```sh
wget http://example.com/dataset.zip
unzip dataset.zip
```

## Results & Visualizations
- **Accuracy on Known Classes (MNIST):** 95.32%
- **Recognition of Unknown Classes (OOD detection):**
  - FashionMNIST: 100%
  - CIFAR-10: 100%
  - SVHN: 100%
  - USPS: 98.5%
  - EMNIST: 81.5%
- **Final Validation Accuracy:** 95.58%
- **Final OSR Model Performance:** Achieved a balanced trade-off between classifying in-distribution digits correctly and rejecting OOD samples.

### Hyperparameter Configuration
Best settings obtained from grid search:
- **Embedding Dimension:** 64
- **Triplet Loss Margin:** 1.0
- **Probability Threshold:** 0.6
- **DBSCAN Parameters:**
  - `eps = 2.2`
  - `min_samples = 9`
  - `distance_factor = 1.0`



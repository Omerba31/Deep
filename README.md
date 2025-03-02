# Open Set Recognition (OSR) Project

## Overview
This project tackles the **Open Set Recognition (OSR)** problem, which occurs when a model encounters data it has never seen before. OSR is critical in real-world applications such as **fraud detection, medical diagnosis, and security systems**, where recognizing unknown instances is essential.

Our approach combines **embedding space analysis, triplet loss learning, probability thresholds, and density-based clustering** to improve classification accuracy and effectively identify unknown samples.

## How It Works
1️⃣ **Feature Extraction:** The model maps input data to an embedding space.  
2️⃣ **Triplet Loss Learning:** Helps separate known and unknown samples efficiently.  
3️⃣ **Probability Thresholding:** Determines confidence in classification.  
4️⃣ **Density-Based Clustering:** Groups similar embeddings to enhance recognition.  

## Installation & Setup
### Running on Google Colab
If you prefer running the project on **Google Colab**, follow these steps:
1️⃣ Open Google Colab: [Google Colab](https://colab.research.google.com/)
2️⃣ Upload the `final_project_OSR.ipynb` notebook.
3️⃣ Run the notebook in Colab without needing a local Python setup.

Alternatively, follow the instructions below to run it locally.
### Prerequisites
- Python 3.x
- Jupyter Notebook
- PyTorch, NumPy, Matplotlib, Scikit-learn

### Setup Instructions
1️⃣ Clone the repository:
```sh
git clone https://github.com/your-username/OSR-Project.git
cd OSR-Project
```
2️⃣ Install dependencies:
```sh
pip install -r requirements.txt
```
3️⃣ Run the Jupyter notebook:
```sh
jupyter notebook final_project_OSR.ipynb
```

## Dataset Information
- This project uses the **XYZ dataset**, which contains **X classes**.
- The dataset is split into:
  - **Known classes:** Used for training.
  - **Unknown classes:** Used for evaluating OSR performance.
- If applicable, download the dataset:
```sh
wget http://example.com/dataset.zip
unzip dataset.zip
```

## Results & Visualizations
- **Accuracy on Known Classes:** 98.5%
- **Recognition of Unknown Classes:** 85.2% (F1-score)

![Embedding Space Visualization](results/embedding_space.png)

## Project Structure
```
OSR-Project/
├── notebooks/         # Jupyter notebooks
│   ├── final_project_OSR.ipynb
├── src/               # Python scripts
│   ├── model.py
│   ├── train.py
├── results/           # Figures & experiment results
│   ├── embedding_space.png
├── data/              # Dataset (if needed)
├── README.md          # Project documentation
├── requirements.txt   # Dependencies
```

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m "Added feature"`)
4. Push branch (`git push origin feature-name`)
5. Submit a pull request

## License
[MIT License] - Free to use and modify.

## Author
Developed as part of a **Deep Learning Final Project**.

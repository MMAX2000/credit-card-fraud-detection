# Unsupervised Credit Card Fraud Detection

An **unsupervised deep learning project** built with **PyTorch** to detect fraudulent credit card transactions without relying on labeled fraud data.

## Project Overview

Credit card fraud costs financial institutions billions every year, and many traditional supervised models struggle with **previously unseen (zero-day) fraud patterns**.

This project addresses the problem using an **autoencoder neural network** trained exclusively on legitimate transactions. The model learns to accurately reconstruct normal spending behavior, while fraudulent transactions generate higher reconstruction errors and are flagged as anomalies.

## Key Results

- **Approach:** Unsupervised deep autoencoder (PyTorch)
- **Dataset:** European Credit Card Transactions Dataset (280k+ transactions)  
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Performance:**
  - **Recall:** ~76% (376 out of 492 frauds detected)
  - **False Positive Rate:** ~0.5% (281 false alarms across 56k+ legitimate transactions)
  - **Threshold Selection:** Optimized using F1-score analysis to balance precision and recall

## Tech Stack

- **Programming:** Python, PyTorch
- **Data Processing:** Pandas, NumPy, Scikit-Learn  
  - `RobustScaler` used for outlier-resistant feature scaling

## Methodology

1. **Preprocessing**  
   Transaction amounts are scaled using `RobustScaler` to reduce the influence of extreme outliers commonly found in financial data.

2. **Model Architecture**  
   A bottleneck autoencoder compresses the 29-feature input into a 3-dimensional latent space, forcing the model to learn the most relevant patterns of legitimate transactions.

3. **Anomaly Detection**  
   Transactions with a reconstruction mean squared error (MSE) above a selected threshold are classified as potentially fraudulent.

## Project Structure

- `fraud_detection_model.ipynb` — End-to-end training, evaluation, and analysis
- `requirements.txt` — Project dependencies

## Author

**Massimiliano Amato**  
Master’s Student in Applied Data Science for Banking & Finance

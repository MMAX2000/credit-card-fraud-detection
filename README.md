# Unsupervised Credit Card Fraud Detection 

A Deep Learning Anomaly Detection system built with **PyTorch** to identify fraudulent credit card transactions without needing labeled fraud data.

## Project Overview
Financial institutions lose billions to fraud annually. Traditional supervised models struggle to detect "Zero-Day" attacks (new fraud patterns they haven't seen before). 

This project implements an **Autoencoder Neural Network** that learns a compressed representation of *legitimate* transactions. By training only on normal data, the model learns to reconstruct valid transactions perfectly but fails to reconstruct anomalies (fraud), flagging them with high error rates.

## Key Results
* **Technique:** Unsupervised Deep Autoencoder (PyTorch)
* **Dataset:** European Credit Card Transactions (280k+ records) https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* **Performance:**
    * **Recall:** ~76% (Detected 376/492 frauds)
    * **Precision:** High precision maintenance with only ~0.5% False Positive Rate (281 false alarms out of 56k+ legitimate transactions).
    * **Stability:** Threshold optimized via F1-Score analysis.

## Tech Stack
* **Core:** Python, PyTorch
* **Data:** Pandas, NumPy, Scikit-Learn (RobustScaler for outlier management)
* **Visualization:** Matplotlib (Log-scale reconstruction error analysis)

## How It Works
1.  **Data Pipeline:** Transaction amounts are scaled using `RobustScaler` to handle extreme outliers common in financial data.
2.  **Architecture:** A bottleneck Autoencoder compresses the 29-feature input down to a 3-neuron latent space, forcing the model to learn the most important features of legitimate spending.
3.  **Inference:** New transactions are passed through the model. If `MSE(Input, Output) > Threshold`, the transaction is flagged.

## Project Structure
* `fraud_detection_model.ipynb`: The complete training and evaluation pipeline.
* `requirements.txt`: Dependencies needed to run the project.

## Author
[Massimiliano Amato] - Master's Student in Applied Data Science for Banking & Finance

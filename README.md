# Concept Drift Detection in Streaming Machine Learning Systems

This project demonstrates how concept drift can be detected in a deployed machine learning model by monitoring prediction error over time.

## Overview
- Simulates a non-stationary data stream with a distribution shift
- Trains a baseline classifier on pre-drift data
- Evaluates the model on streaming data without retraining
- Monitors prediction error using a sliding window
- Automatically detects concept drift when error exceeds a threshold
- Visualizes model degradation over time

## Tech Stack
- Python
- NumPy
- scikit-learn
- Matplotlib

## How It Works
1. Synthetic data is generated with a known drift point
2. A logistic regression model is trained on early data
3. The trained model is evaluated on incoming data
4. Prediction errors are tracked in real time
5. A sliding window computes recent error rates
6. Drift is detected when error exceeds a baseline threshold

## Running the Project
```bash
pip install -r requirements.txt
python main.py

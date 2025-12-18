# Concept Drift Detection in Streaming Machine Learning Systems

This project demonstrates how concept drift can be detected in a deployed machine learning model by monitoring prediction error over time.

## Motivation
In real-world ML systems, models are trained on historical data and deployed for long periods. Over time, the data distribution may change, causing silent performance degradation. This phenomenon is known as concept drift.

This project explores how drift can be detected by tracking prediction errors instead of relying on retraining or accuracy metrics alone.

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
1. Data Generation

A synthetic 1D data stream is generated where:
- Early data follows one distribution
- Later data follows a shifted distribution (concept drift)
- Labels are assigned using a fixed rule, ensuring that only the data distribution changes, not the     labeling logic.

2. Model Training

A simple logistic regression classifier is trained only on pre-drift data.
After training, the model is frozen to simulate a deployed system.

3. Streaming Evaluation

The trained model is used to make predictions on incoming data points.
Each prediction is compared against the true label, producing a binary error signal:
- 0 → correct prediction
- 1 → incorrect prediction

These errors are logged sequentially to form a time-ordered error stream.

4. Sliding-Window Drift Detection

A drift detector monitors the average error rate over a recent window of predictions.

Drift is detected when:
Current Error Rate > Baseline Error + Threshold

This approach ensures that detection is based on sustained degradation, not isolated mistakes.

5. Visualization

Prediction error over time is visualized, along with the known drift point (for reference).
This provides intuitive insight into how model performance degrades under distribution shift.

## Understanding Detection Delay in Concept Drift Monitoring

In this project, the detected drift point may occur later than the actual data distribution shift. This behavior is intentional and reflects how real-world ML monitoring systems operate.

Actual Drift vs. Detected Drift

Actual Drift refers to the moment when the underlying data distribution changes.

Detected Drift refers to the moment when the monitoring system becomes confident that model performance has degraded.

In practice, these two events rarely coincide.

Why Detection Is Delayed
1. Gradual Performance Degradation

After a distribution shift, model performance typically degrades gradually, producing intermittent errors rather than an immediate failure. The system must observe consistent degradation, not isolated mistakes.

2. Sliding Window Aggregation

The detector computes error statistics over a fixed-size sliding window. This smooths noisy error signals and prevents alerts caused by random fluctuations, but introduces natural detection latency.

3. Baseline + Threshold Logic

Drift is detected only when the recent error rate exceeds the baseline error by a configurable threshold. This conservative design prioritizes reliability over immediacy.

Why This Is Desirable?

Immediate detection of every fluctuation would result in:
- Frequent false positives
- Unnecessary retraining
- Unstable monitoring behavior

Delayed detection ensures that alerts correspond to meaningful and persistent model degradation, which is critical in production systems.

## Design Trade-offs

Detection behavior can be tuned by adjusting:
- Window size (smaller windows detect faster but are noisier)
- Threshold (lower thresholds increase sensitivity but risk false positives)

This project intentionally balances:
- Detection latency
- Robustness to noise
- Interpretability of alerts

## Running the Project
```bash
pip install -r requirements.txt
python main.py

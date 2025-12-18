from src.data_generator import generate_stream
from src.model import train_model
from src.visualize import plot_error
from src.drift_detector import DriftDetector

# 1. Generate data
X, y = generate_stream()

# 2. Train model on pre-drift data
X_train = X[:200]
y_train = y[:200]
model = train_model(X_train, y_train)

# 3. Initialize drift detector
detector = DriftDetector(window_size=20, threshold=0.05)

errors = []
drift_detected_at = None

# 4. Stream evaluation
for i in range(200, len(X)):
    x_current = X[i].reshape(1, -1)
    y_true = y[i]
    y_pred = model.predict(x_current)[0]

    error = int(y_pred != y_true)
    errors.append(error)

    if detector.update(error) and drift_detected_at is None:
        drift_detected_at = i
        print(f"⚠️ Drift detected at index {i}")

# 5. Visualize
plot_error(errors, drift_point=300)

if drift_detected_at:
    print("Drift was automatically detected.")
else:
    print("No drift detected.")

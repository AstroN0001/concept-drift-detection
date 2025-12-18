from src.data_generator import generate_stream
from src.model import train_model
from src.visualize import plot_error

# 1. Generate streaming data
X, y = generate_stream()

# 2. Train model on pre-drift data
X_train = X[:200]
y_train = y[:200]
model = train_model(X_train, y_train)

# 3. Evaluate model on streaming data
errors = []

for i in range(200, len(X)):
    x_current = X[i].reshape(1, -1)
    y_true = y[i]
    y_pred = model.predict(x_current)[0]

    error = int(y_pred != y_true)
    errors.append(error)

# 4. Visualize error over time
plot_error(errors, drift_point=300) #original drift is at 500 minus the 200 samples, thus 300


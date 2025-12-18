from src.data_generator import generate_stream
from src.model import train_model

# 1. Generate streaming data
X, y = generate_stream()

# 2. Train model on pre-drift data only
X_train = X[:200]
y_train = y[:200]
model = train_model(X_train, y_train)

# 3. Use model on incoming data and track errors
errors = []

for i in range(200, len(X)):
    x_current = X[i].reshape(1, -1)
    y_true = y[i]
    y_pred = model.predict(x_current)[0]

    error = int(y_pred != y_true)
    errors.append(error)

print("Total predictions made:", len(errors))
print("Total errors:", sum(errors))

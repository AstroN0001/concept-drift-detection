from src.data_generator import generate_stream
from src.model import train_model

# Generate streaming data
X, y = generate_stream()

# Train only on early (pre-drift) data
X_train = X[:200]
y_train = y[:200]

model = train_model(X_train, y_train)

print("Model trained on initial (pre-drift) data.")

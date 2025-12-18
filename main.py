from src.data_generator import generate_stream

X, y = generate_stream()

print("First 5 samples:")
print(X[:5], y[:5])

print("\nLast 5 samples:")
print(X[-5:], y[-5:])

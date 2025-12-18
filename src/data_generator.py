import numpy as np

def generate_stream(n_samples=1000, drift_point=500):
    """
    Generating a 1D data stream with concept drift.

    Before drift:
        Data comes from a normal distribution centered at 0
    After drift:
        Data comes from a normal distribution centered at 3

    Labels:
        1 if value > 1.5 else 0
    """

    X = []
    y = []

    for i in range(n_samples):
        # Before drift
        if i < drift_point:
            x = np.random.normal(loc=0.0, scale=1.0)
        # After drift
        else:
            x = np.random.normal(loc=3.0, scale=1.0)

        # Simple labeling rule
        label = 1 if x > 1.5 else 0

        X.append([x])   # model expects 2D input
        y.append(label)

    return np.array(X), np.array(y)

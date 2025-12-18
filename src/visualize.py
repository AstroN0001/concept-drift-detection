import matplotlib.pyplot as plt

def plot_error(errors, drift_point):
    """
    Plots prediction error over time.
    """

    plt.figure(figsize=(10, 4))
    plt.plot(errors, label="Prediction Error", alpha=0.7)

    plt.axvline(
        x=drift_point,
        color="red",
        linestyle="--",
        label="Actual Drift Point"
    )

    plt.xlabel("Time")
    plt.ylabel("Error (0 = correct, 1 = wrong)")
    plt.title("Model Error Over Time Under Concept Drift")
    plt.legend()
    plt.tight_layout()
    plt.show()

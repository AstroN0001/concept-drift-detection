from collections import deque
import numpy as np

class DriftDetector:
    def __init__(self, window_size=50, threshold=0.2):
        self.window_size = window_size
        self.threshold = threshold
        self.errors = deque(maxlen=window_size)
        self.baseline_error = None

    def update(self, error):
        """
        Add a new error (0 or 1) and check for drift.
        Returns True if drift is detected.
        """
        self.errors.append(error)

        # Not enough data yet
        if len(self.errors) < self.window_size:
            return False

        current_error = np.mean(self.errors)

        # Set baseline once
        if self.baseline_error is None:
            self.baseline_error = current_error
            return False

        # Drift condition
        if current_error > self.baseline_error + self.threshold:
            return True

        return False

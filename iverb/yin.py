import numpy as np

class Yin():

    LOW_PITCH_HZ = 50
    HIGH_PITCH_HZ = 600

    def __init__(self, window_size, sr):
        self._window_size = window_size
        self._sr = sr

        lower_tau = int((1 / self.HIGH_PITCH_HZ) * sr)
        upper_tau = int((1 / self.LOW_PITCH_HZ) * sr)

        self._tau_range = (lower_tau, upper_tau)
        self._num_taus = upper_tau - lower_tau

    def _difference_function(self, x):

        x_windowed = x[:self._window_size]
        shifted_x = np.empty_like(x_windowed)
        diff_func = np.empty_like(x_windowed)

        for tau in range(self._window_size):

            shifted_x[:] = np.roll(x_windowed, shift=tau)
            a = np.correlate(x_windowed, x_windowed, 'valid')[0]
            b = np.correlate(shifted_x, shifted_x, 'valid')[0]
            c = -2 * np.correlate(x_windowed, shifted_x, 'valid')[0]

            diff_func[tau] = a + b + c

        return diff_func

    def _normalized_difference(self, x):
        diff_func = self._difference_function(x)
        normalization = np.cumsum(diff_func) / np.arange(1, self._window_size + 1)
        normalized_diff = diff_func / normalization
        return normalized_diff

    def predict(self, x, threshold=0.15):
        normalized_diff = self._normalized_difference(x)
        candidates = normalized_diff[self._tau_range[0]:self._tau_range[1]]

        ideal_candidates = candidates <= threshold

        if np.any(ideal_candidates):
            best_idx = np.argmax(ideal_candidates)
        else:
            best_idx = np.argmin(candidates)

        best_tau = best_idx + self._tau_range[0]
        estimate_hz = self._sr / best_tau

        return estimate_hz
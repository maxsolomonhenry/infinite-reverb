import numpy as np
from .ola_buffer import OlaBuffer
from .util import db_to_mag, mag_to_db, rms

class PhaseFreezer(OlaBuffer):

    NUM_RAMP_SAMPLES = 5e5

    def __init__(self, frame_size, num_overlap, threshold_db, decay_seconds, sr):
        super().__init__(frame_size, num_overlap)

        self._sr = sr

        self._window = self._make_normalized_window(frame_size)

        self._do_freeze = False
        self._do_grab_frame_one = False
        self._do_grab_frame_two = False

        hN = frame_size // 2 + 1
        self._fft_buffer = np.zeros([hN, 2], dtype=complex)

        self._magnitude = np.zeros(hN)
        self._delta_phase = np.zeros(hN)
        self._phase = np.zeros(hN)

        self._threshold = db_to_mag(threshold_db)
        self._decay_ramp = self._make_decay_ramp(decay_seconds)
        self._p_decay = 0

    def request_freeze(self):
        self._request_freeze()

    def request_unfreeze(self):
        self._request_unfreeze()

    def set_threshold_db(self, x):
        self._threshold = db_to_mag(x)

    def _request_freeze(self):
        self._do_grab_frame_one = True
        self._do_grab_frame_two = False

    def _request_unfreeze(self):
        self._do_freeze = False

    def _make_decay_ramp(self, decay_seconds):
        # See thee mighty JOS:
        # https://ccrma.stanford.edu/~jos/st/Audio_Decay_Time_T60.html

        tau = decay_seconds / 6.91

        time = np.arange(self.NUM_RAMP_SAMPLES) / self._sr
        return np.exp(- time / tau)

    def _make_normalized_window(self, frame_size):
        window = np.hamming(frame_size)

        normalization = 0
        for n in range(0, frame_size, self._hop_size):
            normalization += window[n]

        return window / normalization

    def _processor(self, frame):

        # print(f"RMS:\t{mag_to_db(rms(frame)):.2f}dB")
        if rms(frame) >= self._threshold:
            self._request_freeze()
            self._p_decay = 0

        if self._do_grab_frame_two:
            self._fft_buffer[:, 1] = np.fft.rfft(frame)
            self._do_grab_frame_two = False

            self._init_freeze()
            self._do_freeze = True
            
        if self._do_grab_frame_one:
            self._fft_buffer[:, 0] = np.fft.rfft(frame)
            self._do_grab_frame_one = False
            self._do_grab_frame_two = True

        if self._do_freeze:
            self._phase += self._delta_phase
            frame = self._magnitude * np.exp(1j * self._phase)
            frame = np.fft.irfft(frame)

        frame *= self._window

        return frame
    
    def _post_processor(self, x):

        x *= self._decay_ramp[self._p_decay]

        if self._do_freeze:
            self._p_decay = min(self._p_decay + 1, self.NUM_RAMP_SAMPLES)
            x += self._dry_x

        return x

    def _init_freeze(self):
        self._magnitude = np.abs(self._fft_buffer[:, 1])

        one = np.angle(self._fft_buffer[:, 0])
        two = np.angle(self._fft_buffer[:, 1])
        self._delta_phase = (two - one) % (2 * np.pi)

        self._phase = two

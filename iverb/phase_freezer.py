import numpy as np
from .ola_buffer import OlaBuffer
from .util import db_to_mag, mag_to_db, rms

class PhaseFreezer(OlaBuffer):

    NUM_RAMP_SAMPLES = 5e6

    def __init__(self, frame_size, num_overlap, threshold_db, decay_seconds, bend_amount, sr):
        super().__init__(frame_size, num_overlap)

        self._sr = sr

        self._window = self._make_normalized_window(frame_size)

        self._do_freeze = False

        hN = frame_size // 2 + 1
        self._fft_buffer = np.zeros([hN, 2], dtype=complex)

        self._magnitude = np.zeros(hN)
        self._delta_phase = np.zeros(hN)
        self._phase = np.zeros(hN)
        
        self._phase_offset = np.arange(hN) / frame_size * (2 * np.pi)
        self._bend_amount = bend_amount

        self._threshold = db_to_mag(threshold_db)
        self._decay_ramp = self._make_decay_ramp(decay_seconds)
        self._p_decay = 0

        self._rms_alpha = 1.0 / (2 * frame_size)
        self._envelope = 0

    def set_threshold_db(self, x):
        self._threshold = db_to_mag(x)

    def _make_decay_ramp(self, decay_seconds):
        # See thee mighty JOS:
        # https://ccrma.stanford.edu/~jos/st/Audio_Decay_Time_T60.html

        tau = decay_seconds / 6.91

        time = np.arange(self.NUM_RAMP_SAMPLES) / self._sr
        ramp = np.exp(- time / tau)

        hN = self._frame_size // 2 + 1
        window = np.hamming(self._frame_size)
        ramp[:hN] *= window[:hN]
        
        return ramp

    def _make_normalized_window(self, frame_size):
        window = np.hamming(frame_size)

        normalization = 0
        for n in range(0, frame_size, self._hop_size):
            normalization += window[n]

        return window / normalization
    
    def _pre_processor(self, x):
        
        # Pseudo moving average for amplitude envelope.
        self._envelope = self._rms_alpha * np.abs(x) + (1 - self._rms_alpha) * self._envelope

        return x

    def _processor(self, frame):

        if self._envelope >= self._threshold:
            self._do_freeze = True
            self._p_decay = 0
            self._init_freeze()

        if self._do_freeze:
            self._phase += self._delta_phase
            self._delta_phase += (self._phase_offset * self._bend_amount)

            # Slowly morph magnitude.
            MAG_ALPHA = 0.01
            fft_buffer = np.fft.rfft(frame)
            new_magnitude = np.abs(fft_buffer)
            self._magnitude = MAG_ALPHA * new_magnitude + (1 - MAG_ALPHA) * self._magnitude

            frame = self._magnitude * np.exp(1j * self._phase)
            frame = np.fft.irfft(frame)

        frame *= self._window

        return frame
    
    def _post_processor(self, x):

        x *= self._decay_ramp[self._p_decay]

        if self._do_freeze:
            self._p_decay = min(self._p_decay + 1, self.NUM_RAMP_SAMPLES)
            
        # x += self._dry_x

        return x

    def _init_freeze(self):

        p_previous_frame = (self._p_newest_frame - 1) % self._num_overlap

        frame = self._clean_frame_buffers[:, self._p_newest_frame]
        last_frame = self._clean_frame_buffers[:, p_previous_frame]

        self._fft_buffer[:, 0] = np.fft.rfft(last_frame)
        self._fft_buffer[:, 1] = np.fft.rfft(frame)

        self._magnitude = np.abs(self._fft_buffer[:, 1])

        one = np.angle(self._fft_buffer[:, 0])
        two = np.angle(self._fft_buffer[:, 1])

        self._delta_phase = (two - one) % (2 * np.pi)

        self._phase = two
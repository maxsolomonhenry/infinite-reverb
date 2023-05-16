import numpy as np
from .ola_buffer import OlaBuffer
from .util import db_to_mag, mag_to_db, rms

class PhaseFreezer(OlaBuffer):

    def __init__(self, frame_size, num_overlap, threshold_db):
        super().__init__(frame_size, num_overlap)
        self._window = np.hamming(frame_size)

        self._do_freeze = False
        self._do_grab_frame_one = False
        self._do_grab_frame_two = False

        hN = frame_size // 2 + 1
        self._fft_buffer = np.zeros([hN, 2], dtype=complex)

        self._magnitude = np.zeros(hN)
        self._delta_phase = np.zeros(hN)
        self._phase = np.zeros(hN)

        self._threshold = db_to_mag(threshold_db)

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

    def _processor(self, frame):

        # print(f"RMS:\t{mag_to_db(rms(frame)):.2f}dB")
        if rms(frame) >= self._threshold:
            self._request_freeze()

        frame *= self._window

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
    
    def _init_freeze(self):
        self._magnitude = np.abs(self._fft_buffer[:, 1])

        one = np.angle(self._fft_buffer[:, 0])
        two = np.angle(self._fft_buffer[:, 1])
        self._delta_phase = (two - one) % (2 * np.pi)

        self._phase = two

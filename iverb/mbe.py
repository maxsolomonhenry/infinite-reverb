import numpy as np
from .ola_buffer import OlaBuffer
from .yin import Yin
from .util import db_to_power

import matplotlib.pyplot as plt

class Mbe(OlaBuffer):

    def __init__(self, frame_size, num_overlap, silence_db, sr):
        super().__init__(frame_size, num_overlap)

        window_size = frame_size // 2
        self._yin = Yin(window_size, sr)

        self._silence_threshold = db_to_power(silence_db)

        self._debug = []

    def _pre_processor(self, x):
        return x

    def _processor(self, frame):

        pitch_hz = self._yin.predict(frame)

        power = np.mean(frame ** 2)
        if power < self._silence_threshold:
            pitch_hz = 0

        self._debug.append(pitch_hz)

        return frame
    
    def _post_processor(self, x):
        return x
    
    def get_debug(self):
        return np.array(self._debug)
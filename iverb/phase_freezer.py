import numpy as np
from .ola_buffer import OlaBuffer

class PhaseFreezer(OlaBuffer):

    def __init__(self, frame_size, num_overlap):
        super().__init__(frame_size, num_overlap)
        self._window = np.hamming(frame_size)
        self._do_frame_drop = False

    def _processor(self, frame):

        tmp = frame.copy()

        if self._do_frame_drop:
            tmp *= 0
            self._do_frame_drop = False

        return tmp * self._window
    
    def request_drop(self):
        self._do_frame_drop = True
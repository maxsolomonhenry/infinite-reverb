from abc import ABC, abstractmethod
import numpy as np

class OlaBuffer(ABC):
    def __init__(self, frame_size, num_overlap):
        self._frame_size = frame_size
        self._num_overlap = num_overlap

        self._hop_size = int(np.ceil(frame_size / num_overlap))

        self._delay_buffer = np.zeros(frame_size)
        self._add_buffer = np.zeros(self._hop_size)
        self._frame_buffers = np.zeros((self._frame_size, num_overlap))
        self._clean_frame_buffers = np.zeros((self._frame_size, num_overlap))

        self._p_delay = 0
        self._p_add = 0
        self._p_newest_frame = 0

        self._dry_x = 0

    def process_block(self, block):
        
        num_samples = len(block)

        for n in range(num_samples):
            block[n] = self.process(block[n])

        return block

    def process(self, x):
        self._dry_x = self._delay_buffer[self._p_delay]
        self._delay_buffer[self._p_delay] = x

        is_new_hop = (self._p_delay % self._hop_size == 0)
        if is_new_hop:

            self._clean_frame_buffers[:, self._p_newest_frame] = self._fill_from_delay_buffer()

            self._frame_buffers[:, self._p_newest_frame] = self._processor(
                self._fill_from_delay_buffer()
            )
            self._fill_add_buffer()
            self._p_newest_frame = (self._p_newest_frame + 1) % self._num_overlap

        x = self._add_buffer[self._p_add]

        self._p_delay = (self._p_delay + 1) % self._frame_size
        self._p_add = (self._p_add + 1) % self._hop_size

        x = self._post_processor(x)

        return x
    
    @abstractmethod
    def _processor(self, frame):
        pass

    @abstractmethod
    def _post_processor(self, x):
        pass

    def _fill_add_buffer(self):
        self._add_buffer *= 0

        frame_order = self._get_frame_order()

        p_in = 0
        p_out = self._hop_size

        for f in frame_order:
            self._add_buffer += self._frame_buffers[p_in:p_out, f]

            p_in += self._hop_size
            p_out += self._hop_size

    def _fill_from_delay_buffer(self):
        frame = np.zeros(self._frame_size)

        p_read = self._p_delay

        for n in range(self._frame_size):
            frame[n] = self._delay_buffer[p_read]
            p_read = (p_read + 1) % self._frame_size
        
        return frame

    def _get_frame_order(self):

        # Find distance to newest frame index.
        order = (self._num_overlap - self._p_newest_frame)

        # Make reverse order starting from latest frame.
        order = (np.arange(self._num_overlap)[::-1] - order + 1) % self._num_overlap

        # Wrap to non-negative values.
        order = order + self._num_overlap % self._num_overlap

        return order
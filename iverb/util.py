from scipy.io import wavfile
import numpy as np

def audioread(fpath):
    sr, audio_data = wavfile.read(fpath)

    max_value = np.iinfo(audio_data.dtype).max
    audio_data = audio_data.astype(np.float32) / max_value
    return audio_data, sr
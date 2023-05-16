from scipy.io import wavfile
import numpy as np

def audioread(fpath):
    sr, audio_data = wavfile.read(fpath)

    max_value = np.iinfo(audio_data.dtype).max
    audio_data = audio_data.astype(np.float32) / max_value
    return audio_data, sr

def db_to_mag(x):
    return 10.0 ** (x / 20.0)

def mag_to_db(x):
    return 20.0 * np.log10(x)

def rms(x):
    return np.sqrt(np.mean(x ** 2))
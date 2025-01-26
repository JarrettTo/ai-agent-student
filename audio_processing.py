import numpy as np
from scipy.signal import butter, lfilter
import noisereduce as nr

def bandpass_filter(audio_data, rate, lowcut=300, highcut=3400):
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    filtered_audio = lfilter(b, a, audio_array)
    return filtered_audio.astype(np.int16).tobytes()

def reduce_noise(audio_data, rate):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    noise_profile = audio_array[:rate] if len(audio_array) > rate else audio_array
    reduced_audio = nr.reduce_noise(y=audio_array, sr=rate, y_noise=noise_profile)
    return reduced_audio.tobytes()

def preprocess_audio(audio_data, rate):
    audio_data = reduce_noise(audio_data, rate)
    audio_data = bandpass_filter(audio_data, rate)
    return audio_data

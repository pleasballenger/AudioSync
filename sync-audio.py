import numpy as np
import pandas as pd
import wave 

def read_wav(path):
    with wave.open(path, 'rb') as file:
        frames = file.getnframes()
        audio = file.readframes(frames)
        return audio

def compute_fft(audio):
    audio_array = np.frombuffer(audio, dtype=np.int16)
    fft = np.fft.fft(audio_array)
    return fft

def cross_corr(fft_1, fft_2):
    cross_correlation = np.correlate(fft_1, fft_2, mode='full')
    return cross_correlation

def max(cross):
    return np.argmax(np.abs(cross))

def time_shift(cross, fft_1):
    max_corr_index = max(cross)
    shift = max_corr_index / fft_1

    return shift

def synchronize(path1, path2):
    audio1 = read_wav(path1)
    audio2 = read_wav(path2)

    diff = len(audio2) - len(audio1)
    if diff > 0:
        audio1 = np.pad(audio1, (0, diff))
    else:
        audio2 = np.pad(audio2, (0, -diff))

    shifted_audio2 = np.roll(audio2, max(cross_corr(compute_fft(audio1), compute_fft(audio2))))

    return shifted_audio2
  

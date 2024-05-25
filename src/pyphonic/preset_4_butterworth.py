# butterworth
# Butterworth filter (High, Low, or Bandpass)

import pyphonic

import numpy as np
from scipy.signal import butter, sosfilt

order = 6
fs = None
sos = None
cutoff = 500
Zl, Zr = None, None

def process_npy(midi, audio):
    global Zl, Zr, fs, sos
    if fs is None:
        fs = pyphonic.getSampleRate()
        sos = butter(order, cutoff, fs=fs, btype='lowpass', analog=False, output='sos')
        Zl = np.zeros((sos.shape[0], 2))
        Zr = np.zeros((sos.shape[0], 2))
        return midi, audio
    left = audio[0]
    right = audio[1]
    left, Zl = sosfilt(sos, left, zi=Zl)
    right, Zr = sosfilt(sos, right, zi=Zr)
    return midi, np.stack((left, right))
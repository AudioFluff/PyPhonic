# beat_synced_noise
# Add some fizz to your chorus with noise. 1000 ticks per beat, so starting at 250 is a 16th after each beat.

import numpy as np

import pyphonic

NOISE_VOLUME = 0.2
NOISE_DENSITY = NOISE_VOLUME * 0.5
start_ticks, end_ticks = 250, 300

def process_npy(midi, audio):
    if start_ticks <= pyphonic.getTransport()["ticks"] < end_ticks:
        noise = np.random.normal(scale=NOISE_VOLUME, size=audio.shape)
        noise = (np.abs(np.random.normal(size=audio.shape)) < NOISE_DENSITY) * noise
        return midi, audio + noise.astype(np.float32)
    return midi, audio
# sc_compressor
# Fake sidechain compressor to duck audio on the kick

import numpy as np

import pyphonic

ATTENUATION = 0.5
start_ticks, end_ticks = 0, 250

def process_npy(midi, audio):
    if start_ticks <= pyphonic.getTransport()["ticks"] < end_ticks:
        return midi, audio * ATTENUATION
    return midi, audio
import torch

import pyphonic

# This adds oomph to EDM drums
# DEW_POINT = 0.9
# SHIFT = 0
# FOLDBACK_POINT = 0.8
# COMP_GAIN = 1.1

# This is a Brit guitar amp
DEW_POINT = 0.1
SHIFT = 0.005
FOLDBACK_POINT = 0.8
COMP_GAIN = 3.0

def process_torch(midi, audio):
    audio[audio < DEW_POINT * audio.min()] = DEW_POINT * audio.min() + SHIFT
    audio[audio > DEW_POINT * audio.max()] = DEW_POINT * audio.max() + SHIFT
    audio[audio > FOLDBACK_POINT] = FOLDBACK_POINT - audio[audio > FOLDBACK_POINT]
    audio[audio < -FOLDBACK_POINT] = -FOLDBACK_POINT + -audio[audio < -FOLDBACK_POINT]
    return midi, audio * COMP_GAIN
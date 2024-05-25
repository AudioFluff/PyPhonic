# Noise
# Just adds some configurable noise to the input audio

import torch

import pyphonic

NOISE_VOLUME = 0.1
# 0.01 is a vinyl crackle, 0.1 is a light rain, 0.5 is a heavy rain, 1 is a thunderstorm
NOISE_DENSITY = NOISE_VOLUME * 0.5

def process_torch(midi, audio):
    noise = torch.randn_like(audio) * NOISE_VOLUME
    noise = noise * (torch.rand_like(noise).abs() < NOISE_DENSITY)
    return midi, audio + noise
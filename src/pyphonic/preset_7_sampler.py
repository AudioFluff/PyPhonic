# time_stretching_sampler
# Stretches the duration of a sample without affecting pitch. Playable with MIDI; samples will start looping faster as they get shorter.

from pathlib import Path

import pyphonic
from pyphonic import MidiMessage
import numpy as np
import librosa

sample = np.load(pyphonic.getDataDir() / Path("glockenspiel.pkl"), allow_pickle=True)

def noteToFreq(midi_note):
    a = 440
    return (a / 32) * (2 ** ((midi_note - 9) / 12))

print("Building wavetable")
voices = {}
for note in range(31, 103): # G1 to G7
    print(f"Building wavetable {note}")
    if note == 60:
        ratio = 1 # Center on C3
    else:
        freq = noteToFreq(note)
        ratio = freq / 261.63
    left = librosa.effects.time_stretch(sample[0], rate=ratio)
    right = librosa.effects.time_stretch(sample[1], rate=ratio)
    joined = np.array([left, right])
    voices[note] = {"wave": joined, "position": 0, "playing": False, "velocity": 0}

def process_npy(midi, audio):

    num_channels, num_samples = audio.shape

    for msg in midi:
        if msg.note not in voices:
            continue
        if msg.type == "note_on":
            if voices[msg.note]["playing"]:
                voices[msg.note]["position"] = 0
            else:
                voices[msg.note]["playing"] = True
            voices[msg.note]["velocity"] = msg.velocity
        elif msg.type == "note_off":
            voices[msg.note]["position"] = 0
            voices[msg.note]["playing"] = False
    
    new_audio = np.zeros((num_samples, num_channels), dtype=np.float32)

    for voice, data in voices.items():
        if not data["playing"]:
            continue
        start_pos = data["position"] % data["wave"].shape[1]
        end_pos = (start_pos + num_samples)
        if end_pos >= data["wave"].shape[1]:
            end_pos = data["wave"].shape[1]
            new_audio[:, :end_pos - start_pos] += data["wave"][:, start_pos:end_pos]
            voices[voice]["position"] = 0
        else:
            new_audio += data["wave"][:, start_pos:end_pos] * (data["velocity"] / 127)
            voices[voice]["position"] += num_samples
    
    return midi, new_audio
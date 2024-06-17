# drum_synth
# Drum sample player. C3 is kick, D3 is snare, E3 is hat, F3 is open hat, G3 is perc.

from pathlib import Path

import pyphonic
import numpy as np

kick = np.load(pyphonic.getDataDir() / Path("kick.pkl"), allow_pickle=True)
snare = np.load(pyphonic.getDataDir() / Path("snare.pkl"), allow_pickle=True)
chh = np.load(pyphonic.getDataDir() / Path("chh.pkl"), allow_pickle=True)
ohh = np.load(pyphonic.getDataDir() / Path("ohh.pkl"), allow_pickle=True)
perc = np.load(pyphonic.getDataDir() / Path("perc.pkl"), allow_pickle=True)

voices = {}
for note in [60, 62, 64, 65, 67]: # C3 to G3
    switch = {
        60: kick,
        62: snare,
        64: chh,
        65: ohh,
        67: perc
    }
    voices[note] = {"wave": switch[note], "positions": [], "velocities": []}

def process_npy(midi, audio):

    num_channels, num_samples = audio.shape

    for msg in midi:
        if msg.note not in voices:
            continue
        if msg.type == "note_on":
            voices[msg.note]["positions"].append(0)
            voices[msg.note]["velocities"].append(msg.velocity)

    new_audio = np.zeros((num_channels, num_samples), dtype=np.float32)

    for voice, data in voices.items():
        if not data["positions"]:
            continue
        ended = []
        for i in range(len(data["positions"])):
            start_pos = data["positions"][i]
            if start_pos >= data["wave"].shape[1]:
                ended.append(i)
                continue
            end_pos = (start_pos + num_samples)
            if end_pos >= data["wave"].shape[1]:
                end_pos = data["wave"].shape[1]
                new_audio[:, :end_pos - start_pos] += data["wave"][:, start_pos:end_pos] * (data["velocities"][i] / 127)
                ended.append(i)
            else:
                new_audio += data["wave"][:, start_pos:end_pos] * (data["velocities"][i] / 127)
                voices[voice]["positions"][i] += num_samples
        for i in ended:
            try:
                voices[voice]["positions"].pop(i)
                voices[voice]["velocities"].pop(i)
            except:
                pass
    
    return [], new_audio
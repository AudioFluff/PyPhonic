# Generative drum machine
# Outputs MIDI (C3-G3) to pipe to a sampler. Hold one or all of those notes down, press play.

import random

import pyphonic
from pyphonic import MidiMessage

midi_notes = {x: False for x in range(128)}
current_pattern = {}
generated_patterns = {}
start = None
last_bar = -1
last_quarter = -1

instruments = {
    "kick": 60,
    "snare": 62,
    "hat": 64,
    "openhat": 65,
    "perc": 67
}
kick_patterns = [
    [(1, 2, 127), (5, 6, 127), (9, 10, 127), (13, 14, 127)],
    [(1, 4, 127), (8, 9, 127), (9, 13, 127)],
    [(1, 4, 127), (8, 9, 127), (9, 10, 127), (11, 11, 45)],
    [(1, 4, 127), (8, 9, 35), (9, 10, 127), (11, 11, 45)],
    [(1, 2, 127), (9, 10, 127)],
    [(1, 2, 127), (6, 6, 127), (9, 10, 127), (14, 14, 127)],
    [(1, 2, 127), (5, 6, 127), (9, 10, 127), (12, 12, 30), (13, 14, 110)],
]
snare_patterns = [
    [(5, 6, 127), (13, 15, 127)],
    [(5, 6, 127), (13, 15, 127), (16, 16, 127)],
    [(5, 6, 127), (13, 15, 127), (16, 16, 127)],
    [(5, 6, 127), (10, 10, 63), (13, 15, 127), (16, 16, 127)],
    [(2, 3, 127), (13, 14, 127)],
]
closed_hat_patterns = [
    [(1, 1, 127), (2, 2, 127), (3, 3, 127), (4, 4, 127), (5, 5, 127), (6, 6, 127), (7, 7, 127), (8, 8, 127), (9, 9, 127), (10, 10, 127), (11, 11, 127), (12, 12, 127), (13, 13, 127), (14, 14, 127), (15, 15, 127), (16, 16, 127)],
    [(3, 3, 127), (4, 4, 127), (7, 7, 127), (8, 8, 127), (11, 11, 127), (12, 12, 127), (15, 15, 127), (16, 16, 127)],
]
open_hat_patterns = [
    [(3, 3, 80), (7, 7, 80), (11, 11, 80), (15, 15, 80)],
    [(1, 2, 30), (5, 6, 30), (9, 10, 30), (13, 14, 60)],
]


def is_note_on(note):
    return midi_notes[note]
def process_midi(midi):
    global midi_notes
    for msg in midi:
        if msg.type == "note_on":
            midi_notes[msg.note] = True
        elif msg.type == "note_off":
            midi_notes[msg.note] = False
        elif msg.type == "channel_pressure":
            print(f"New pattern {msg.note}")
            generate_pattern(msg.note)
    return None
def generate_perc():
    i = 0
    perc = []
    while i < 16:
        if random.random() < 0.4:
            next_ = random.randint(0, 4)
            perc.append((i, i + next_, random.randint(1, 127)))
        else:
            next_ = 1
        i += next_
    return perc
def generate_pattern(num):
    global current_pattern
    if generated_patterns.get(num):
        current_pattern = generated_patterns[num]
    else:
        current_pattern = {
            "kick": random.choice(kick_patterns),
            "snare": random.choice(snare_patterns),
            "hat": random.choice(closed_hat_patterns),
            "openhat": random.choice(open_hat_patterns),
            "perc": generate_perc(),
        }
        generated_patterns[num] = current_pattern

    return current_pattern
def play_pattern(cur_16th):
    global current_pattern
    midi_out = []
    for hit in current_pattern:
        for start, stop, vel in current_pattern[hit]:
            if cur_16th == start and is_note_on(instruments[hit]):
                midi_out.append(MidiMessage("note_on", instruments[hit], vel, 0))
            if cur_16th == stop:
                midi_out.append(MidiMessage("note_on", instruments[hit], 0, 0))
    return midi_out

generate_pattern(0)

def get_16th():
    global start, last_bar, last_quarter
    transport = pyphonic.getTransport()
    if start is None:
        start = transport["sample_num"]
    elapsed = transport["sample_num"] - start
    bpm = pyphonic.getBPM()
    samples_per_beat = 60 * 44100 / bpm
    quarter = int((elapsed / 44100 * 8) % samples_per_beat)
    if transport["bar"] != last_bar:
        start = transport["sample_num"]
        last_bar = transport["bar"]
    if quarter != last_quarter:
        num_16 = quarter + 1
    else:
        num_16 = None
    last_quarter = quarter
    return num_16


def process(midi, audio):
    global start, last_bar, last_quarter
    
    process_midi(midi)

    num_16 = get_16th()
    
    midi_out = play_pattern(num_16)

    return midi_out, audio
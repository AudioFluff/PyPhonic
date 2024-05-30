# midi_gen_markovian
# Generates MIDI chords. Pipe it into a lush pad, *HIT PLAY*, survive the ambient and wait for the epic trance progression.

import random

import pyphonic
from pyphonic import MidiMessage

last_bar = -1
first_of_bar = True
last_chord = None
root = 60 # C4
chords = {
  "I": { 
    "notes": [root, root + 4, root + 7],
    "next": ["IV", "V"]
  },
  "V": {
    "notes": [root - 5, root - 1, root + 2],
    "next": ["vi"]
  },
  "vi": {
    "notes": [root - 3, root, root + 4, root + 9],
    "next": ["IV", "V", "viinv", "IVoct"]
  },
  "viinv": {
    "notes": [root - 3, root, root + 4, root + 9],
    "next": ["IV", "V", "I"]
  },
  "IV": {
    "notes": [root - 7, root - 3, root],
    "next": ["I", "vi", "V"]
  },
  "IVoct": {
    "notes": [root - 7, root - 3, root, root + 5],
    "next": ["I", "vi"]
  }
}

def gen_new_chord(last_chord):
    if last_chord is None:
        return "I"
    next_ = random.choice(chords[last_chord]["next"])
    return next_

def process(midi, audio):
    global last_chord, last_bar, first_of_bar
    current_bar = pyphonic.getTransport()["bar"] % 8

    if last_bar != current_bar:
        first_of_bar = True
        last_bar = current_bar
    else:
        first_of_bar = False
    
    if first_of_bar and current_bar in (0, 2, 4, 5, 6):
        if current_bar == 5:
            if random.random() < 0.7:
                return midi, audio

        if last_chord:
            for note in chords[last_chord]["notes"]:
                midi.append(MidiMessage("note_off", note, 127, 0))
        last_chord = gen_new_chord(last_chord)
        for note in chords[last_chord]["notes"]:
            midi.append(MidiMessage("note_on", note, 127, 0))

    return midi, audio
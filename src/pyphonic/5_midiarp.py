# midi_arp
# Simple MIDI minor triad arpeggiator

import pyphonic
from pyphonic import MidiMessage

timer = 0
midibuf = {}
initial_delay = 0
duration = 1
arp_delay = 7
num_samples = None

def add_note(beats_from_now, type, note, velocity, channel):
    global midibuf, timer, num_samples

    bpm = pyphonic.getBPM()
    
    bps = bpm / 60
    ticks_per_second = bps * 1000

    blocks_per_second = pyphonic.getSampleRate() / num_samples

    ticks_per_block = ticks_per_second / blocks_per_second

    tick_skip = 1000 * beats_from_now
    
    blocks_in_future = tick_skip / ticks_per_block / 4 # assume 4/4 time

    when = int(timer + blocks_in_future)

    midibuf[when] = midibuf.get(when, [])

    new_note = MidiMessage(type, note, velocity, channel)
    if type == "note_off":
        new_note.velocity = 0
    midibuf[when].append(new_note)

def process(midi, audio):
    global timer, midibuf, num_samples
    if num_samples is None:
        num_samples = len(audio[0])
    for msg in midi:
        if msg.type == "note_on":
            when = initial_delay
            add_note(when, "note_on", msg.note, msg.velocity, msg.channel)
            when += duration
            add_note(when, "note_off", msg.note, msg.velocity, msg.channel)
            when += arp_delay
            add_note(when, "note_on", msg.note + 3, msg.velocity, msg.channel)
            when += duration
            add_note(when, "note_off", msg.note + 3, msg.velocity, msg.channel)
            when += arp_delay
            add_note(when, "note_on", msg.note + 7, msg.velocity, msg.channel)
            when += duration
            add_note(when, "note_off", msg.note + 7, msg.velocity, msg.channel)
    
    for k in list(midibuf.keys()):
        if k < timer:
            del midibuf[k]

    timer += 1

    return midibuf.get(timer - 1, []), audio
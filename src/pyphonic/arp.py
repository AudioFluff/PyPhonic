import pyphonic
from pyphonic import MidiMessage

timer = 0
midibuf = {}
initial_delay = 1
duration = 20
arp_delay = 50

def add_note(when, type, note, velocity, channel):
    global midibuf
    midibuf[when] = midibuf.get(when, [])
    if ("note_off", note) in [(x.type, x.note) for x in midibuf[when]]:
        print(f"Skipping {type} {note} {velocity} {channel} at {when}")
        return
    new_note = MidiMessage(type, note, velocity, channel)
    if type == "note_off":
        new_note.velocity = 0
    midibuf[when].append(new_note)

def process(midi, audio):
    global timer, midibuf
    for msg in midi:
        if msg.type == "note_on":
            print(f"Note on: {msg.note} {msg.velocity} {msg.channel}")
            when = timer + initial_delay
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
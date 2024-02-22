import pyphonic
from pyphonic import MidiMessage

timer = 0
midibuf = {}
delay = 300
arp_delay = 10

def process(midi, audio):
    global timer, midibuf
    for msg in midi:
        midibuf[timer + delay] = midibuf.get(timer, [])
        new_note = MidiMessage(msg.type, msg.note, msg.velocity, msg.channel)  
        midibuf[timer + delay].append(new_note)
    timer += 1
    return midibuf.get(timer, []), audio
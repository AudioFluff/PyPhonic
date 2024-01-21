import random
on = False
def process_npy(midi_in, audio):
    global on
    midi = bytearray(b"")
    if any([x.type=="note_on" for x in midi_in]):
        on = True
    if any([x.type=="note_off" for x in midi_in]):
        on = False
    if random.randint(1, 100) > 98 and on:
        midi = bytearray(b"\x90\x4a\x5a")
    else:
        if random.randint(1, 1000) > 996:
            midi = bytearray(b"\x80\x4a\x5a")
    return midi, audio



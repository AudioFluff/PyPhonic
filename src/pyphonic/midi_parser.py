class MidiMessage:
    """A MIDI message.

    Attributes:
        type (str): The type of message: `note_on`, `note_off`, `control_change`, `pitch_wheel_change`.

        note (int): The note number (if note_on/off) or control number (if control_change). Else garbage.

        velocity (int): The velocity (if note_on/off) or control value/pitch wheel position (0-127).

        channel (int): The channel number.
        
        description (str): A plain English description of the message.
    """
    def __init__(self, type_, note, velocity, channel):
        self.type = type_
        self.note = note
        self.velocity = velocity
        self.channel = channel
        plain_type = type_.replace("_", " ").title()
        self.description = f"{plain_type} {get_note_info(note)} Velocity {velocity} Channel {channel}"

def parse_bytes_to_midi(midi_bytes):
    midi_messages = []
    
    while len(midi_bytes) >= 3:
        status_byte = midi_bytes[0] & 0xF0
        
        if status_byte == 0x00:
            message_type = "skip"
        elif status_byte == 0x80:
            message_type = "note_off"
        elif status_byte == 0x90:
            message_type = "note_on"
        elif status_byte == 0xA0:
            message_type = "polyphonic_aftertouch"
        elif status_byte == 0xB0:
            message_type = "control_change"
        elif status_byte == 0xC0:
            message_type = "program_change"
        elif status_byte == 0xD0:
            message_type = "channel_pressure"
        elif status_byte == 0xE0:
            message_type = "pitch_wheel_change"
        else:
            message_type = "unknown"
        
        if message_type != "skip":
            note = midi_bytes[1]
            velocity = midi_bytes[2]
            channel = midi_bytes[0] & 0x0F
            midi_message = MidiMessage(message_type, note, velocity, channel)
            midi_messages.append(midi_message)
        
        midi_bytes = midi_bytes[3:]
    
    return midi_messages


def get_note_info(midi_note):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_note // 12) - 1
    note_index = midi_note % 12
    note_name = notes[note_index]
    return f"{note_name}{octave}"


def parse_midi_to_bytes(midi_messages):
    midi_bytes = []
    
    for midi_message in midi_messages:
        message_type = midi_message.type
        note = midi_message.note
        velocity = midi_message.velocity
        channel = midi_message.channel
        
        if message_type == "note_off":
            status_byte = 0x80 | channel
        elif message_type == "note_on":
            status_byte = 0x90 | channel
        elif message_type == "polyphonic_aftertouch":
            status_byte = 0xA0 | channel
        elif message_type == "control_change":
            status_byte = 0xB0 | channel
        elif message_type == "program_change":
            status_byte = 0xC0 | channel
        elif message_type == "channel_aftertouch":
            status_byte = 0xD0 | channel
        elif message_type == "pitch_wheel_change":
            status_byte = 0xE0 | channel
        else:
            continue

        midi_bytes.append(status_byte)
        midi_bytes.append(note)
        midi_bytes.append(velocity)
    
    return bytes(midi_bytes)
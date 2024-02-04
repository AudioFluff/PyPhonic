
def parse_midi_bytes(midi_bytes):
    midi_messages = []
    
    while midi_bytes:
        status_byte = midi_bytes[0]
        
        # Check the status byte to determine the type of MIDI message
        if status_byte & 0xF0 == 0x80:
            message_type = "note_off"
        elif status_byte & 0xF0 == 0x90:
            message_type = "note_on"
        elif status_byte & 0xF0 == 0xA0:
            message_type = "polyphonic_aftertouch"
        elif status_byte & 0xF0 == 0xB0:
            message_type = "control_change"
        elif status_byte & 0xF0 == 0xC0:
            message_type = "program_change"
        elif status_byte & 0xF0 == 0xD0:
            message_type = "channel_aftertouch"
        elif status_byte & 0xF0 == 0xE0:
            message_type = "pitch_wheel_change"
        else:
            message_type = "unknown"
        
        # Extract the note, velocity, and channel from the MIDI bytes
        note = midi_bytes[1]
        velocity = midi_bytes[2]
        channel = status_byte & 0x0F
        
        # Create a MIDI message object with the extracted attributes
        midi_message = {
            "type": message_type,
            "note": note,
            "velocity": velocity,
            "channel": channel,
            "description": f"{message_type} - Note: {note}, Velocity: {velocity}, Channel: {channel}"
        }
        
        # Append the MIDI message to the list
        midi_messages.append(midi_message)
        
        # Remove the processed bytes from the MIDI bytes
        midi_bytes = midi_bytes[3:]
    
    return midi_messages


def get_note_info(midi_note):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_index = midi_note % 12
    note_name = notes[note_index]
    if len(note_name) == 2:
        return note_name[0], "sharp"
    else:
        return note_name, "flat"



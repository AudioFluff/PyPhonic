# Status: Python + Python-in-c++ code is identical! Except in Python they have to `from pyphonic import state`

# TODO do the flippin kubernetes thing
# Destruction test!
# Some kind of authentication on top of the magic packet, maybe just on initial connect
# Add top menu bar to quickly try out some presets, ideally loaded from git... saturator, "piano minus the note's fundamental frequency as a
# sine wave", etc.
# Set up git repo, cicd, pypi for pyphonic. (Working on it)
# Installer, packaging Python with it :(
# Does it work on mac :(
# Does it work in the DAW

# python -c "import pyphonic; from pyphonic.demo import process;  pyphonic.start(process, 8019)"

import socket
import threading
import math
import struct
import sys
import time

import mido

all_threads = []
should_stop = threading.Event()
safe_to_transmit = threading.Event()

in_buffer, out_buffer = [], []
seq_num = 0
block_size = 44100, 441

class State:
    sample_rate = 44100
    block_size = 441
    num_channels = 2
    bpm = 120
    sample_num = 0

state = State()

def transmit(socket, b: bytes):
    magic_num = 15
    header = magic_num.to_bytes(4, byteorder="little")
    length = len(b)
    length = length.to_bytes(4, byteorder="little")
    msg = header + length + b
    socket.sendall(msg)

def handle(socket_, addr):
    global cache
    global cacheidx
    global block_size

    next_recv_size = None
    empty_receives = 0
    while not should_stop.is_set():
        try:
            data = bytearray(socket_.recv(next_recv_size or 100).strip())
        except socket.timeout:
            should_stop.set()
            continue
        if not len(data):
            empty_receives += 1
            if empty_receives > 100:
                print("Other end disconnected.")
                should_stop.set()
            continue

        if b'EHLO' in data:
            transmit(socket_, b'ok\n')
            print(f"Received handshake from client.")
            safe_to_transmit.set()
        elif b'AUDIO' in data:
            global in_buffer
            global seq_num
            desired_length = 100000
            
            if not next_recv_size:
                while True:
                    chunk = socket_.recv(100).strip()
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) >= desired_length:
                        break
                    split_at = b'AUDIO'
                    headers, content = data.split(split_at)
                    hex_length = content[:4]

                    content_length = int.from_bytes(hex_length, byteorder="little")
                    (state.sample_rate, state.block_size, state.num_channels,
                        state.bpm, state.sample_num) = struct.unpack('<2iBfl', content[4:21])
                    print(state.sample_rate, state.block_size, state.num_channels, state.bpm, state.sample_num)
                    
                    desired_length = int.from_bytes(headers[-4:], byteorder='little') + len(headers)
                    next_recv_size = int.from_bytes(headers[-4:], byteorder='little') + len(headers)
                    print(f"Next recv size: {next_recv_size}")
                    if len(data) >= desired_length:
                        break
            # magic_num:total_msg_length:AUDIO:(content+midi)_length:samplerate:blocksize:numChannels:bpm:sample_num
            
            (state.sample_rate, state.block_size, state.num_channels,
                state.bpm, state.sample_num) = struct.unpack('<2iBfl', data[17:34])
            
            content = data[len(headers)+len(split_at)+4+4+4+1+4+4:]
            midi, audio = content[:100], content[100:]
            
            if len(audio) != content_length - 100:
                print("Not an error, just an early payload: ", len(audio), content_length, len(data))
                continue
            else:
                in_buffer.append((seq_num, audio, midi))
                seq_num += 1


def shuffler(process_fn):

    def wrapped_process_fn(midi_messages, audio):
        return process_fn(midi_messages, audio)

    while not should_stop.wait(0.0001):
        while len(in_buffer) and not should_stop.is_set():
            seq_num, audio_in, midi_in = in_buffer.pop(0)
            audio_in = struct.unpack(f"{state.block_size*state.num_channels}f", audio_in) # bytes -> floats
            p = mido.Parser()
            p.feed(midi_in)
            rendered_audio = wrapped_process_fn(p.messages, audio_in)
            try:
                rendered_audio = struct.pack(f"{state.block_size*state.num_channels}f", *rendered_audio)
            except struct.error:
                print("Audio length didn't match, returning silence this time.")
                rendered_audio = [0.0] * (state.block_size*state.num_channels)
                rendered_audio = struct.pack(f"{state.block_size*state.num_channels}f", *rendered_audio)

            out_buffer.append((seq_num, rendered_audio))

def responder(socket_):
    safe_to_transmit.wait(10)
    if not safe_to_transmit.is_set():
        print("Handshake didn't happen")
        sys.exit(1)
    while not should_stop.wait(0.0001):
        try:
            if not len(out_buffer):
                continue
            s, o = out_buffer.pop(0)
            transmit(socket_, o)
        except BrokenPipeError:
            print("Client disconnected, probably")
            should_stop.set()
            raise

def start(process_fn, port=8015):
    global in_buffer, out_buffer, seq_num
    while True:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(5)
        s.bind(('0.0.0.0', port))
        s.listen(1)
        all_threads = []
        safe_to_transmit.clear()
        in_buffer, out_buffer = [], []
        seq_num = 0
        should_stop.clear()
        shuffle_thread = threading.Thread(target=shuffler, args=(process_fn, ))
        shuffle_thread.start()
        all_threads.append(shuffle_thread)
        try:
            print(f"Listening on port {port}...")
            conn = None
            while not should_stop.is_set():
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                conn.settimeout(1)
                print(conn, addr)
                handler_thread = threading.Thread(target=handle, args=(conn, addr))
                handler_thread.start()
                all_threads.append(handler_thread)
                responder_thread = threading.Thread(target=responder, args=(conn, ))
                responder_thread.start()
                all_threads.append(responder_thread)
        except KeyboardInterrupt:
            print("User initiated stop.")
            should_stop.set()
        except BrokenPipeError:
            print("Other end disconnected.")
            should_stop.set()
        finally:
            print("Stopping listener...")
            if s:
                s.close()
            for t in all_threads:
                t.join()
            print("Restarting listener in 1s. Ctrl-C to quit.")
            time.sleep(1)
        
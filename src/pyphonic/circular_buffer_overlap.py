# circular_buffer_overlap
# Just a demo of how to make a circular buffer in NumPy that can also handle overlapped writes. First few seconds will be rough.

import numpy as np
from numpy.fft import rfft, irfft
from scipy import signal

import pyphonic

BUF_SIZE = 8820

stored_buffer_left = np.zeros((BUF_SIZE, ), dtype=np.float32)
stored_buffer_right = np.zeros((BUF_SIZE, ), dtype=np.float32)
ptr_left = 0
ptr_right = 0
started = False

def write_to(data, buffer, ptr):
    buffer[ptr:ptr + len(data)] = data
    ptr += len(data)
    return buffer, ptr
def read_from(num, buffer, ptr):
    ptr = max(0, ptr - num)
    retval = buffer[0:num].copy()
    buffer[0:num] *= 0.
    buffer = np.roll(buffer, -num)
    return retval, buffer, ptr
def overlapped_write(data, buffer, ptr, overlap=256):
    num_can_overlap = ptr - max(ptr - overlap, 0)
    buffer[max(ptr - overlap, 0):ptr] += data[:min(num_can_overlap, overlap)]
    buffer[ptr:ptr + len(data) - overlap] = data[overlap:]
    ptr += len(data) - overlap
    return buffer, ptr


def process_npy(midi, audio):
    global stored_buffer_left, stored_buffer_right, ptr_left, ptr_right, started
    stored_buffer_left, ptr_left = write_to(audio[0], stored_buffer_left, ptr_left)
    stored_buffer_right, ptr_right = write_to(audio[1], stored_buffer_right, ptr_right)

    if started or ptr_left > 1024:
        started = True
        left, stored_buffer_left, ptr_left = read_from(pyphonic.getBlockSize(), stored_buffer_left, ptr_left)
        right, stored_buffer_right, ptr_right = read_from(pyphonic.getBlockSize(), stored_buffer_right, ptr_right)
        
        return midi, np.stack([left, right])
    
    return midi, np.zeros((pyphonic.getNumChannels(), pyphonic.getBlockSize()))
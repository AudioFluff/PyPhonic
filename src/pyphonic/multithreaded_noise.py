# Multithreaded Noise
# Demo of using threads from Python. Important bit: set daemon = True otherwise it'll hang at exit.
import numpy as np
import threading

out = np.zeros([441,], dtype=np.float32)

def mythread():
    global out
    while True:
        out = np.random.randn(441)

mine = threading.Thread(target=mythread)

mine.daemon = True
mine.start()

def process_npy(midi, audio):
    return midi, np.concatenate([out, out])
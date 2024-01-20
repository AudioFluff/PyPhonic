import math
class Synth:
    def __init__(self, sample_rate=44100):
        self.angleDelta = 0.0
        self.currentAngle = 0.0
        self.tail = None
        self.stopped = True
        self.sample_rate = sample_rate # default, will be set by vst
        self.sin_cache = {}

    def start_note(self, note, vel):
        self.currentAngle = 0.0
        self.level = vel * 0.05
        self.tail = None
        self.stopped = False

        def noteToFreq(midi_note):
            a = 440
            return (a / 32) * (2 ** ((midi_note - 9) / 12))
        self.cyclesPerSecond = noteToFreq(note)
        self.cyclesPerSample = self.cyclesPerSecond / self.sample_rate
        self.angleDelta = self.cyclesPerSample * 2 * math.pi
    
    def sin(self, val):
        val = val % (2 * math.pi)
        val = round(val, 3)
        retval = self.sin_cache.get(val, None)
        if retval is not None:
            return retval
        self.sin_cache[val] = math.sin(val)
        return self.sin_cache[val]
    
    def stop_note(self):
        self.tail = 50000
    
    def renew(self):
        self.tail = None
    
    def is_active(self):
        if self.stopped:
            return False
        return True
    
    def render(self, num):
        is_stopping = self.tail is not None
        buf = []
        if abs(self.angleDelta) != 0 and not self.stopped:
            while num > 0:
                num -= 1
                cur = self.sin(self.currentAngle) * self.level
                if self.tail is not None:
                    cur *= self.tail / 50000
                    self.tail = self.tail - 1
                buf.append(cur)
                self.currentAngle += self.angleDelta
        else:
            buf = [0.0 for _ in range(num)]
        
        if is_stopping and self.tail <= 0:
            self.tail = None
            self.angleDelta = 0.0
            self.stopped = True

        return buf

class Poly:

    def __init__(self, sample_rate=44100, block_size=64):
        self.synths = {}
        self.delay_buf = [0.0] * 12000
        self.delay_position = 0
        self.sample_rate = sample_rate
        self.block_size = block_size

    def set_sample_rate_block_size(self, rate, block_size):
        self.sample_rate = rate
        self.block_size = block_size

    def start_note(self, note, vel):
        if note not in self.synths:
            self.synths[note] = Synth(sample_rate=self.sample_rate)
            self.synths[note].start_note(note, vel)
        elif not self.synths[note].is_active():
            self.synths[note].start_note(note, vel)
        else:
            self.synths[note].renew()
    
    def stop_note(self, note):
        if note in self.synths:
            self.synths[note].stop_note()
    
    def render(self):

        if not len(self.synths):
            cur = [0.0] * self.block_size
        else:
            bufs = []
            one_active = False
            for note, synth in self.synths.items():
                if synth.is_active():
                    one_active = True
                    bufs.append(synth.render(self.block_size))
            if not one_active:
                cur = [0.0] * self.block_size
            else:
                cur = [sum(x) for x in zip(*bufs)]

        for i in range(0, self.block_size):
            in_ = cur[i]
            cur[i] += self.delay_buf[self.delay_position]
            self.delay_buf[self.delay_position] = (self.delay_buf[self.delay_position] + in_) * 0.5
            self.delay_position += 1
            if self.delay_position >= len(self.delay_buf):
                self.delay_position = 0

        return cur * 2 # mono

poly = Poly()
import random
from pyphonic import state

def process(midi_messages, audio):
    newaudio = [0.0] * len(audio)
    if random.choice([0,1]) == 0:
        r = range(0, state.block_size)
    else:
        r = range(state.block_size, state.block_size*2)
    for i in r:
        newaudio[i] = audio[i]
    return newaudio






    return poly.render()
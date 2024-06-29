# binaural_beats_lfo
# Binaural beats, plus an LFO on top

import math
import pyphonic

LEFT = 440  # Hz
RIGHT = 444  # For a beat frequency of 4Hz
LFO_FREQ = 4  # Hz

class Synth:
    def __init__(self, sample_rate=44100):
        self.angleDelta = 0.0
        self.currentAngle = 0.0
        self.sample_rate = sample_rate # default, will be set by vst

    def start_note(self, freq, vel):
        self.currentAngle = 0.0
        self.level = vel
        self.cyclesPerSecond = freq
        self.cyclesPerSample = self.cyclesPerSecond / self.sample_rate
        self.angleDelta = self.cyclesPerSample * 2 * math.pi
    
    def render(self, num):
        buf = []
        while num > 0:
            num -= 1
            cur = math.sin(self.currentAngle) * self.level
            buf.append(cur)
            self.currentAngle += self.angleDelta
        return buf

class Poly:
    def __init__(self):
        self.synths = {}
        self.sample_rate = None
        self.block_size = None
        self.not_yet = []

    def set_sample_rate_block_size(self, rate, block_size):
        if self.sample_rate is not None:
            return
        self.sample_rate = rate
        self.block_size = block_size
        for tone, vel, pan in self.not_yet:
            self.start_tone(tone, vel, pan)

    def start_tone(self, tone, vel, pan):
        if self.sample_rate is None:
            self.not_yet.append((tone, vel, pan))
            return
        self.synths[pan] = Synth(sample_rate=self.sample_rate)
        self.synths[pan].start_note(tone, vel)
    
    def render(self):
        bufs = []
        bufs.append(self.synths[0].render(self.block_size))
        bufs.append(self.synths[1].render(self.block_size))
        lfo = self.synths[2].render(self.block_size)
        for channel in bufs:
            for i in range(len(channel)):
                channel[i] *= 1 + lfo[i]
        return bufs

poly = Poly()
poly.start_tone(LEFT, 0.1, 0)
poly.start_tone(RIGHT, 0.1, 1)
poly.start_tone(LFO_FREQ, 0.2, 2)

def process(midi_messages, audio):
    num_samples = len(audio[0])
    poly.set_sample_rate_block_size(pyphonic.getSampleRate(), num_samples)
    return midi_messages, poly.render()
# polysine
# A simple polyphonic sine wave synthesizer

import math
import numpy as np
import pyphonic
import librosa

class ADSR:
    def __init__(self, attack=20, decay=20, sustain=0.9, release=20):
        self.attack = attack
        self.decay = decay
        self.sustain = max(0, min(sustain, 1))
        self.release = max(release, 1)
        self.current = 0
        self.is_on = True
        self.release_done = False
    def start_release(self):
        self.is_on = False
        self.current = self.sustain
        self.release_done = False
    def render(self, num):
        buf = []
        if self.is_on:
            while num > 0:
                num -= 1
                if self.current < self.attack:
                    self.current += 1
                    cur = self.current / self.attack
                elif self.current < self.attack + self.decay:
                    self.current += 1
                    cur = 1 - (1 - self.sustain) * (self.current - self.attack) / self.decay
                else:
                    cur = self.sustain
                buf.append(cur)
        else:
            while num > 0:
                num -= 1
                if self.current > 0:
                    self.current -= (self.sustain / self.release)
                else:
                    self.current = 0
                    self.release_done = True
                buf.append(self.current)
        return np.array(buf, dtype=np.float32)

class Synth:
    def __init__(self, sample_rate=44100, block_size=448, detune_coarse=0, detune=0, op="sin",
                 phase=0, rel_vel=1.0, attack=0, decay=0, sustain=0, release=0,
                 delay_seconds=0.0, delay_length=0, delay_feedback=0, delay_mix=0,
                 lfo=None, lfo_freq=None, lfo_velocity=None):
        self.angleDelta = 0.0
        self.currentAngle = 0.0
        self.stopped = True
        self.sample_rate = sample_rate # default, will be set by vst
        self.block_size = block_size

        self.detune_coarse = detune_coarse
        self.detune = detune
        self.op = op
        self.phase = phase
        self.rel_vel = rel_vel
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.delay_buf = np.zeros(max(int(delay_seconds * self.sample_rate), self.block_size), dtype=np.float32)
        self.delay_feedback = delay_feedback
        self.delay_mix = delay_mix
        self.no_delay = self.delay_feedback == 0 and self.delay_mix == 0
        self.one_cycle = None
        self.delay_length = delay_length  # If there is delay, we do a min of 1 iteration of that delay (e.g. 1sec if delay_seconds==1) OR this many. AKA, how many echos
        self.delay_ends_in = None  # NOT PUBLIC after note stops, how long til we can just return 0s

        self.lfo = lfo  # Can add any other Synth object as an LFO
        if self.lfo is not None:
            assert isinstance(lfo_freq, (int, float)), "If adding an LFO, supply its frequency as lfo_freq"
            assert isinstance(lfo_velocity, float), "If adding an LFO, supply its velocity as lfo_velocity between 0.0 and 1.0"
            assert self.detune_coarse == 0, "LFO is not compatible with detune_coarse"
            assert self.detune == 0, "LFO is not compatible with detune"
            self.lfo.start_freq(lfo_freq, lfo_velocity)
    
    def start_freq(self, freq, vel):
        self.level = vel
        self.stopped = False
        self.delay_ends_in = None
        self.envelope = ADSR(self.attack, self.decay, self.sustain, self.release)
        self.hz = freq
        self.waveform = self.func(self.hz, type_=self.op) * self.rel_vel
        self.waveform = np.roll(self.waveform, -int((self.phase/360) * self.one_cycle))

    def start_note(self, note, vel):
        self.level = vel / 10
        self.stopped = False
        self.delay_ends_in = None
        self.envelope = ADSR(self.attack, self.decay, self.sustain, self.release)

        def noteToFreq(midi_note):
            a = 440
            freq = (a / 32) * (2 ** ((midi_note - 9) / 12))
            return round(freq, 2)
        self.hz = noteToFreq(note + self.detune_coarse) + self.detune
        
        self.waveform = self.func(self.hz, type_=self.op) * self.rel_vel
        self.waveform = np.roll(self.waveform, -int((self.phase/360) * self.one_cycle))
    
    def func(self, hz, type_="sin"):
        fs = self.sample_rate
        if type_ == "square":
            reps = int(fs // hz)
            ramp = np.linspace(-1, 1, reps)
            ramp[ramp >= 0] = 1
            ramp[ramp < 0] = -1
            val = np.tile(ramp, min(reps, 10))
        elif type_ == "saw":
            reps = int(fs / hz)
            ramp = np.linspace(-1, 1, reps)
            val = np.tile(ramp, min(reps, 10))
        elif type_ == "tri":
            reps = int(fs / hz)
            ramp = np.linspace(-1, 1, reps // 2)
            ramp = np.append(ramp, np.linspace(1, -1, reps // 2))
            val = np.tile(ramp, min(reps, 10))
        elif type_ == "sin":
            ramp = np.linspace(0, 2 * np.pi, fs)
            val = np.sin(ramp * hz)
        elif type_ == "laser":
            ramp = np.logspace(0, 2 * np.pi, fs)
            val = np.sin(ramp * hz)
        
        #val = librosa.resample(val, orig_sr=self.sample_rate, target_sr=self.sample_rate * 0.5)
        #hz = hz / 0.5
        # then use this
        #max_size = int(int(hz) * self.one_cycle) # working one... at least without freq shift
        # if self.one_cycle < 50:
        #     reps = 30
        # elif self.one_cycle < 150:
        #     reps = 20
        # else:
        #     reps = 10
        # max_size = math.ceil(reps * self.one_cycle)
        # val = val[:max_size]

        # Find integer periods
        self.one_cycle = fs / hz
        
        if type_ in ("sin", "laser"):
            max_size = int(int(hz) * self.one_cycle) # working one... at least without freq shift
            val = val[:max_size]
        else:
            max_size = int(reps * self.one_cycle)
            if max_size > self.sample_rate:
                val = np.tile(val[:int(self.one_cycle)], 2)
            else:
                val = val[:max_size]

        while val.shape[0] < self.block_size:
            val = np.append(val, val)
        
        return val
    
    def stop_note(self):
        self.envelope.start_release()
        min_delay = self.delay_buf.size // self.block_size
        self.delay_ends_in = self.delay_length * min_delay
    
    def is_active(self):
        if self.stopped:
            return False
        return True
    
    def renew(self):
        self.envelope = ADSR(self.attack, self.decay, self.sustain, self.release)
    
    def render(self):
        is_stopped = self.envelope.release_done
        if not is_stopped:
            buf = self.waveform[:self.block_size] * self.level
            self.waveform = np.roll(self.waveform, -self.block_size)
            buf *= self.envelope.render(self.block_size)
            if self.lfo is not None:
                lfo = self.lfo.render()
                buf *= 1 + lfo
        else:
            buf = np.zeros(self.block_size, dtype=np.float32)

        if not self.no_delay:
            if self.delay_ends_in is None or self.delay_ends_in > 0:
                self.delay_buf = np.roll(self.delay_buf, -self.block_size)
                buf += self.delay_buf[:self.block_size] * self.delay_feedback
                    
                buf += self.delay_buf[:self.block_size] * self.delay_mix
                if self.delay_ends_in is not None:
                    self.delay_ends_in -= 1
                self.delay_buf[-self.block_size:] = buf

        return buf

class Poly:

    def __init__(self, sample_rate=44100, block_size=448):
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
            self.synths[note] = {"synths": [
                Synth(sample_rate=self.sample_rate, block_size=self.block_size, detune_coarse=0, detune=0, op="sin", rel_vel=0.5, attack=20, decay=2000, sustain=0.4, release=20, delay_feedback=0, delay_mix=0),
                # Synth(sample_rate=self.sample_rate, block_size=self.block_size, detune_coarse=0, detune=0, op="sin", rel_vel=0.5, attack=20, decay=2000, sustain=0.4, release=20, delay_seconds=0.5, delay_length=5, delay_feedback=0.5, delay_mix=0,
                #       lfo=Synth(sample_rate=self.sample_rate, block_size=self.block_size, op="tri", rel_vel=0.9, attack=20, decay=2000, sustain=1.0, release=20, delay_seconds=0.2, phase=90, delay_length=2, delay_feedback=0.3, delay_mix=0.2), lfo_freq=13, lfo_velocity=1.0,
                #       ),
                # Synth(sample_rate=self.sample_rate, block_size=self.block_size, detune_coarse=-24, op="sin", rel_vel=0.5, attack=20, decay=2000, sustain=0.7, release=20),
                # Synth(sample_rate=self.sample_rate, block_size=self.block_size, detune_coarse=-12, op="saw", rel_vel=0.5, attack=200, decay=200, sustain=0.5, release=20000),
                # Synth(sample_rate=self.sample_rate, block_size=self.block_size, detune_coarse=0, detune=-1, op="saw", phase=-10, attack=200, decay=200, sustain=0.5, release=100),
                # Synth(sample_rate=self.sample_rate, block_size=self.block_size, detune=0, op="saw", attack=200, decay=200, sustain=0.5, release=2000),
                # Synth(sample_rate=self.sample_rate, block_size=self.block_size, detune_coarse=0, detune=1, op="saw", phase=10, attack=200, decay=200, sustain=0.5, release=10000)
            ]}
            for synth in self.synths[note]["synths"]:
                synth.start_note(note, vel)
        else:
            for synth in self.synths[note]["synths"]:
                synth.renew()
    
    def stop_note(self, note):
        if note in self.synths:
            for synth in self.synths[note]["synths"]:
                synth.stop_note()
    
    def render(self):
        buf = np.zeros(self.block_size, dtype=np.float32)
        for _, synths in self.synths.items():
            for synth in synths["synths"]:
                this = synth.render()
                if this is not None:
                    buf += this
        return buf

poly = Poly()
started = None

def process_npy(midi_messages, audio):
    global started
    num_samples = len(audio[0])
    if started is None:
        poly.set_sample_rate_block_size(pyphonic.getSampleRate(), num_samples)
        started = True
    for m in midi_messages:
        if m.type == "note_on":
            if m.note < 20:
                continue
            poly.start_note(m.note, m.velocity/10)
        elif m.type == "note_off":
            poly.stop_note(m.note)
        else:
            print(m)
    
    render = poly.render()
    out = np.stack((render, render), axis=0)
    return midi_messages, out

# Synth
# Full-featured wavetable and FM with unlimited voices of different kinds, ADSR envelopes, delay, LFOs that can be synths themselves modulating frequency and amplitude...

import math
import random

import numpy as np
import pyphonic

from typing import Callable
from scipy.signal import savgol_filter


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

    @classmethod
    def square(cls, pos, extra_params=None):
        if pos < math.pi:
            return 1
        return -1
    @classmethod
    def tri(cls, pos, extra_params=None):
        return 0.5 - abs(pos - math.pi) / math.pi
    @classmethod
    def saw(cls, pos, extra_params=None):
        return (pos / (2 * math.pi)) - 0.5
    @classmethod
    def noise(cls, pos, extra_params=None):
        return random.random() * 2 - 1
    @classmethod
    def sin(cls, pos, extra_params=None):
        return math.sin(pos)
    
    def __init__(self, detune_coarse=0, detune=0, op="sin", gen_op=None, op_extra_params={},
                 phase=0, rel_vel=1.0, attack=0, decay=0, sustain=0.0, release=0,
                 delay_seconds=0.0, delay_length=0, delay_feedback=0, delay_mix=0,
                 lfo=None, lfo_freq=None, lfo_velocity=None,
                 freq_mod=None, freq_mod_freq=None, freq_mod_velocity=None,
                 ):
        self.stopped = True

        # Set later, in the loop
        self.sample_rate = None
        self.block_size = None
        self.delay_buf = None

        self.detune_coarse: int = detune_coarse
        self.detune: int = detune
        self.op: str = op
        self.gen_op: Callable = gen_op  # e.g. lambda x: math.sin(x)
        if self.op == "gen":
            assert self.gen_op is not None, "If using generative operators, you must supply a function, e.g. Synth.sin or Synth.noise, or any lambda that transforms a position in [0, 2*pi] to a float in [-1, 1]"
            assert isinstance(self.gen_op, Callable), "If using generative operators, you must supply a function, e.g. Synth.sin or Synth.noise, or any lambda that transforms a position in [0, 2*pi] to a float in [-1, 1]"
        self.op_extra_params = op_extra_params  # e.g. {std, filter_size, filter_poly_degree} for random walk
        
        self.phase: int = phase  # Phase shift in degrees, from 0 to 360
        self.rel_vel: float = rel_vel  # If voices are stacked, relative velocity of this one

        self.envelope = None
        self.attack = attack  # Time to go from no sound to full output
        self.decay = decay  # Time after attack time to drop to sustain level
        self.sustain: float = sustain  # Amplitude of held note, in the middle
        self.release = release  # Time to go from sustain to no sound
        
        self.delay_seconds = delay_seconds
        self.delay_feedback = delay_feedback
        self.delay_mix = delay_mix
        self.no_delay = self.delay_feedback == 0 and self.delay_mix == 0
        self.delay_length = delay_length  # If there is delay, we do a min of 1 iteration of that delay (e.g. 1sec if delay_seconds==1) OR this many. AKA, how many echos
        self._delay_ends_in = None  # After note stops, how long til we can just return 0s
        
        self.one_cycle = None  # Set when note starts playing - the period
        self.position = None  # Initially set when note starts playing

        self.lfo = lfo  # Can add any other Synth object as an LFO
        self.lfo_freq = lfo_freq
        self.lfo_velocity = lfo_velocity
        if self.lfo is not None:
            assert isinstance(lfo_freq, (int, float)), "If adding an LFO, supply its frequency as lfo_freq"
            assert isinstance(lfo_velocity, float), "If adding an LFO, supply its velocity as lfo_velocity between 0.0 and 1.0"
            assert self.detune_coarse == 0, "LFO is not compatible with detune_coarse"
            assert self.detune == 0, "LFO is not compatible with detune"
        
        self.freq_mod = freq_mod
        self.freq_mod_freq = freq_mod_freq
        self.freq_mod_velocity = freq_mod_velocity
        if self.freq_mod is not None:
            assert isinstance(freq_mod_freq, (int, float)), "If adding a frequency LFO, supply its frequency as lfo_freq"
            assert isinstance(freq_mod_velocity, float), "If adding a frequency LFO, supply its velocity as lfo_velocity between 0.0 and 1.0"
            assert self.op.startswith("gen"), "Frequency modulation is only compatible with generative operators"
            assert self.freq_mod.op.startswith("gen"), "Frequency modulation is only compatible with generative operators"
            assert self.detune_coarse == 0, "LFO is not compatible with detune_coarse"
            assert self.detune == 0, "LFO is not compatible with detune"
    
    def set_sample_rate_block_size(self, rate, block_size):
        self.sample_rate = rate
        self.block_size = block_size
        self.delay_buf = np.zeros(max(int(self.delay_seconds * self.sample_rate), self.block_size), dtype=np.float32)
        if self.lfo is not None:
            self.lfo.set_sample_rate_block_size(self.sample_rate, self.block_size)
            self.lfo.start_freq(self.lfo_freq, self.lfo_velocity)
        if self.freq_mod is not None:
            self.freq_mod.set_sample_rate_block_size(self.sample_rate, self.block_size)
            self.freq_mod.start_freq(self.freq_mod_freq, self.freq_mod_velocity)

    def start_freq(self, freq, vel):
        self.level = vel
        self.stopped = False
        self._delay_ends_in = None
        self.envelope = ADSR(self.attack * self.sample_rate, self.decay * self.sample_rate, self.sustain, self.release * self.sample_rate)
        self.hz = freq
        if self.op == "gen":
            self.waveform = None
            self.one_cycle = self.sample_rate / self.hz
            self.position = int((self.phase/360) * self.one_cycle)
        else:
            self.waveform = self.func(self.hz, type_=self.op) * self.rel_vel
            self.waveform = np.roll(self.waveform, -int((self.phase/360) * self.one_cycle))

    def start_note(self, note, vel):
        self.level = vel / 10
        self.stopped = False
        self._delay_ends_in = None
        self.envelope = ADSR(self.attack * self.sample_rate, self.decay * self.sample_rate, self.sustain, self.release * self.sample_rate)

        def noteToFreq(midi_note):
            a = 440
            freq = (a / 32) * (2 ** ((midi_note - 9) / 12))
            return round(freq, 2)
        self.hz = noteToFreq(note + self.detune_coarse) + self.detune
        
        if self.op == "gen":
            self.waveform = None
            self.one_cycle = self.sample_rate / self.hz
            self.position = int((self.phase/360) * self.one_cycle)
        else:
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
        elif type_ == "randomwalk":
            reps = int(fs // hz)
            randomData = bounded_random_walk(reps, lower_bound=-1, upper_bound=1, start=0, end=0, std=self.op_extra_params.get("std", 1))
            if self.op_extra_params.get("filter_size"):
                assert self.op_extra_params.get("filter_poly_degree"), "If you want to filter the random walk, you must supply the polynomial degree as filter_poly_degree e.g. 1, 2, 3..."
                if randomData.size >= self.op_extra_params.get("filter_size"):
                    filter_size = self.op_extra_params.get("filter_size")
                else:
                    filter_size = randomData.size
                smooth = savgol_filter(randomData, filter_size, min(filter_size, self.op_extra_params.get("filter_poly_degree")))
            else:
                smooth = randomData
            val = np.tile(smooth, min(reps, 10))

        # Find integer periods
        self.one_cycle = fs / hz
        
        if type_ in ("sin", "laser"):
            if int(hz) != 0:
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
        self._delay_ends_in = self.delay_length * min_delay
    
    def is_active(self):
        if self.stopped:
            return False
        return True
    
    def renew(self):
        self.envelope = ADSR(self.attack * self.sample_rate, self.decay * self.sample_rate, self.sustain, self.release * self.sample_rate)
    
    def render(self):
        is_stopped = self.envelope.release_done
        if not is_stopped:
            if self.waveform is None:
                num = self.block_size
                cyclesPerSample = 1 / self.one_cycle
                angleDelta = cyclesPerSample * 2 * math.pi
                buf = []

                if self.freq_mod is not None:
                    angle_deltas = self.freq_mod.render()
                while num > 0:
                    num -= 1
                    cur = self.gen_op(self.position, self.op_extra_params) * self.level
                    buf.append(cur)
                    self.position += angleDelta
                    self.position %= 2 * math.pi
                    if self.freq_mod is not None:
                        angleDelta += angle_deltas[self.block_size - num - 1]
                        angleDelta %= 2 * math.pi
                buf = np.array(buf)
            else:
                buf = self.waveform[:self.block_size] * self.level
                self.waveform = np.roll(self.waveform, -self.block_size)
            buf *= self.envelope.render(self.block_size)
            if self.lfo is not None:
                lfo = self.lfo.render()
                buf *= 1 + lfo
        else:
            buf = np.zeros(self.block_size, dtype=np.float32)

        if not self.no_delay:
            if self._delay_ends_in is None or self._delay_ends_in > 0:
                self.delay_buf = np.roll(self.delay_buf, -self.block_size)
                buf += self.delay_buf[:self.block_size] * self.delay_feedback
                    
                buf += self.delay_buf[:self.block_size] * self.delay_mix
                if self._delay_ends_in is not None:
                    self._delay_ends_in -= 1
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
                # Synth(detune_coarse=0, detune=0, op="sin", rel_vel=0.5, attack=20, decay=2000, sustain=0.4, release=20, delay_feedback=0, delay_mix=0),
                # Synth(detune_coarse=0, detune=0, op="gen", gen_op=Synth.square, rel_vel=1.0, attack=1, decay=3, sustain=0.6, release=1.1, delay_feedback=0, delay_mix=0,
                #       freq_mod=Synth(detune_coarse=0, detune=0, op="gen", gen_op=Synth.square, rel_vel=0.5, attack=1, decay=1, sustain=0.4, release=0.0001, delay_feedback=0, delay_mix=0), freq_mod_freq=2, freq_mod_velocity=0.00005),
                Synth(detune_coarse=0, detune=0, op="randomwalk", rel_vel=0.3, attack=0.1, decay=0.3, phase=90, sustain=0.5, release=2, delay_seconds=0.05, delay_length=2, delay_feedback=0.0, delay_mix=0,
                      op_extra_params={"std": 2, "filter_size": 20, "filter_poly_degree": 5}),
                # Synth(detune_coarse=0, detune=0, op="sin", rel_vel=0.5, attack=20, decay=2000, sustain=0.4, release=20, delay_seconds=0.5, delay_length=5, delay_feedback=0.5, delay_mix=0,
                #       lfo=Synth(sample_rate=self.sample_rate, block_size=self.block_size, op="tri", rel_vel=0.9, attack=20, decay=2000, sustain=1.0, release=20, delay_seconds=0.2, phase=90, delay_length=2, delay_feedback=0.3, delay_mix=0.2), lfo_freq=13, lfo_velocity=1.0,
                #       ),
                # Synth(detune_coarse=-24, op="sin", rel_vel=0.5, attack=20, decay=2000, sustain=0.7, release=20),
                # Synth(detune_coarse=-12, op="saw", rel_vel=0.5, attack=200, decay=200, sustain=0.5, release=20000),
                # Synth(detune_coarse=0, detune=-1, op="saw", phase=-10, attack=200, decay=200, sustain=0.5, release=100),
                # Synth(detune=0, op="saw", attack=200, decay=200, sustain=0.5, release=2000),
                # Synth(detune_coarse=0, detune=1, op="saw", phase=10, attack=200, decay=200, sustain=0.5, release=10000)
            ]}
            for synth in self.synths[note]["synths"]:
                synth.set_sample_rate_block_size(self.sample_rate, self.block_size)
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
    
    render = poly.render()
    out = np.stack((render, render), axis=0)
    return midi_messages, out


def bounded_random_walk(length, lower_bound,  upper_bound, start, end, std):
    assert (lower_bound <= start and lower_bound <= end)
    assert (start <= upper_bound and end <= upper_bound)

    bounds = upper_bound - lower_bound

    rand = (std * (np.random.random(length) - 0.5)).cumsum()
    rand_trend = np.linspace(rand[0], rand[-1], length)
    rand_deltas = (rand - rand_trend)
    rand_deltas /= np.max([1, (rand_deltas.max()-rand_deltas.min())/bounds])

    trend_line = np.linspace(start, end, length)
    upper_bound_delta = upper_bound - trend_line
    lower_bound_delta = lower_bound - trend_line

    upper_slips_mask = (rand_deltas-upper_bound_delta) >= 0
    upper_deltas =  rand_deltas - upper_bound_delta
    rand_deltas[upper_slips_mask] = (upper_bound_delta - upper_deltas)[upper_slips_mask]

    lower_slips_mask = (lower_bound_delta-rand_deltas) >= 0
    lower_deltas =  lower_bound_delta - rand_deltas
    rand_deltas[lower_slips_mask] = (lower_bound_delta + lower_deltas)[lower_slips_mask]

    return trend_line + rand_deltas
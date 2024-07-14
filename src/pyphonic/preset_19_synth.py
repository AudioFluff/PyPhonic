# Synth
# Full-featured wavetable and FM with unlimited voices of different kinds, ADSR envelopes, delay, LFOs that can be synths themselves modulating frequency and amplitude...

import math
import random
from collections import defaultdict
from copy import deepcopy

import librosa
import numpy as np
import pyphonic

from typing import Callable
from scipy.signal import butter, iirpeak, savgol_filter, sosfilt, tf2sos


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
        if not self.is_on:
            return
        self.is_on = False
        self.current = self.sustain
        self.release_done = False
    def reset(self):
        self.is_on = True
        self.current = 0
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

class Filter:
    def __init__(self, **args):
        """
        drywet: 1.0 is completely dry, 0.0 is completely wet
        """
        for k, v in args.items():
            setattr(self, k, v)

    def __call__(self, buf):
        output, self.Z = sosfilt(self.sos, buf, zi=self.Z)
        return buf * self.drywet + (output * (1 - self.drywet))
    
    def set_sample_rate_block_size(self, rate, block_size, idx):
        self.sample_rate = rate
        self.block_size = block_size
        if self.type.startswith("band"):
            self.sos = butter(self.order, [self.low, self.high], fs=self.sample_rate, btype=self.type, analog=False, output='sos')            
            self.Z = np.zeros((self.sos.shape[0], 2))
        elif self.type == "resonant":
            b, a = iirpeak(self.w, self.Q, fs=self.sample_rate)
            self.sos = tf2sos(b, a)
            self.Z = np.zeros((self.sos.shape[0], 2))
        else:
            self.sos = butter(self.order, self.cutoff, fs=self.sample_rate, btype=self.type, analog=False, output='sos')
            self.Z = np.zeros((self.sos.shape[0], 2))
    
    @classmethod
    def lowpass(cls, cutoff=500, order=6, drywet=0.0):
        return cls(cutoff=cutoff, order=order, type="lowpass", drywet=drywet)
    
    @classmethod
    def highpass(cls, cutoff=500, order=6, drywet=0.0):
        return cls(cutoff=cutoff, order=order, type="highpass", drywet=drywet)
    
    @classmethod
    def bandpass(cls, low=500, high=600, order=6, drywet=0.0):
        return cls(low=low, high=high, order=order, type="bandpass", drywet=drywet)
    
    @classmethod
    def bandstop(cls, low=500, high=600, order=6, drywet=0.0):
        return cls(low=low, high=high, order=order, type="bandstop", drywet=drywet)

    @classmethod
    def resonant(cls, w, Q, drywet=0.0):
        return cls(w=w, Q=Q, type="resonant", drywet=drywet)

wavetables = defaultdict()

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
                 phase=0, rel_vel=1.0, attack=0, decay=0, sustain=1.0, release=0,
                 delay_seconds=0.0, delay_length=0, delay_feedback=0, delay_mix=0,
                 lfo=None, lfo_freq=None, lfo_velocity=None,
                 freq_mod=None, freq_mod_freq=None, freq_mod_velocity=None,
                 filter=None
                 ):
        self.stopped = True

        # Set later, in the loop
        self.sample_rate = None
        self.block_size = None
        self.idx: int = None  # Index of the synth in the stack, to preserve data across different notes' synths
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
        
        self.filter = filter
        if self.filter is not None:
            assert isinstance(self.filter, Filter), "If adding a filter, supply a Filter object"
    
    def set_sample_rate_block_size(self, rate, block_size, idx):
        self.sample_rate = rate
        self.block_size = block_size
        self.idx = idx
        self.delay_buf = np.zeros(max(int(self.delay_seconds * self.sample_rate), self.block_size), dtype=np.float32)
        if self.lfo is not None:
            self.lfo.set_sample_rate_block_size(self.sample_rate, self.block_size, str(self.idx) + "_lfo")
            self.lfo.start_freq(self.lfo_freq, self.lfo_velocity)
        if self.freq_mod is not None:
            self.freq_mod.set_sample_rate_block_size(self.sample_rate, self.block_size, str(self.idx) + "_freq_mod")
            self.freq_mod.start_freq(self.freq_mod_freq, self.freq_mod_velocity)
        if self.filter is not None:
            self.filter.set_sample_rate_block_size(self.sample_rate, self.block_size, str(self.idx) + "_filter")

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
            
        else:
            self.waveform = self.func(self.hz, type_=self.op) * self.rel_vel
            self.waveform = np.roll(self.waveform, -int((self.phase/360) * self.one_cycle))

        self.position = int((self.phase/360) * self.one_cycle)
    
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
        elif type_ == "noise":
            val = np.random.random(fs) * 2 - 1
        elif type_ == "wavetable": 
            reps = int(fs // hz)
            self.position = 0
            
            if self.idx in wavetables:
                sample, orig_freq = wavetables[self.idx]

                if abs(orig_freq - hz) > 1.0:
                    print(f"{self.sample_rate * orig_freq / hz} is what were resampling to")
                    sample = np.tile(sample, 10)
                    sample = librosa.resample(sample, orig_sr=self.sample_rate, target_sr=int(self.sample_rate * orig_freq / hz))
            else:
                sample = np.load(self.op_extra_params["sample"], allow_pickle=True)
                if sample.shape[0] > 1:
                    sample = sample[0]
                if self.op_extra_params.get("start"):
                    start = int(self.op_extra_params["start"] * sample.size)
                else:
                    start = 0
                if self.op_extra_params.get("end"):
                    if self.op_extra_params.get("find_closest"):
                        l = int(self.op_extra_params.get("end") * sample.size)
                        r = l
                        target = sample[start]
                        best = float("inf")
                        best_idx = None
                        while l > start and r < sample.size:
                            if abs(sample[l] - target) < best:
                                best = abs(sample[l] - target)
                                best_idx = l
                            if abs(sample[r] - target) < best:
                                best = abs(sample[r] - target)
                                best_idx = r
                            l -= 1
                            r += 1
                        end = best_idx
                    else:
                        end = int(self.op_extra_params["end"] * sample.size)
                else:
                    end = sample.size
                sample = sample[start:end]
                sample /= np.max(np.abs(sample))
                orig_freq = self.op_extra_params.get("orig_freq", 440)
                if abs(orig_freq - hz) > 1.0:
                    print(f"{self.sample_rate * orig_freq / hz} is what were resampling to")
                    sample = np.tile(sample, 10)
                    sample = librosa.resample(sample, orig_sr=self.sample_rate, target_sr=int(self.sample_rate * orig_freq / hz))
                wavetables[self.idx] = (sample, hz)
            self.original_wavetable_length = sample.size
            val = np.tile(sample, min(reps, 10))

        elif type_ == "randomwalk":
            reps = int(fs // hz)
            if self.idx in wavetables:
                smooth, orig_freq = wavetables[self.idx]
                if orig_freq != hz:
                    smooth = librosa.resample(smooth, orig_sr=self.sample_rate, target_sr=self.sample_rate * orig_freq / hz)
            else:
                randomData = bounded_random_walk(reps, lower_bound=-1, upper_bound=1, start=0, end=0, std=self.op_extra_params.get("std", 1))
                if self.op_extra_params.get("filter_size"):
                    assert self.op_extra_params.get("filter_poly_degree"), "If you want to smooth the random walk, you must supply the polynomial degree as filter_poly_degree e.g. 1, 2, 3..."
                    if randomData.size >= self.op_extra_params.get("filter_size"):
                        filter_size = self.op_extra_params.get("filter_size")
                    else:
                        filter_size = randomData.size
                    smooth = savgol_filter(randomData, filter_size, min(filter_size, self.op_extra_params.get("filter_poly_degree")))
                else:
                    smooth = randomData
                wavetables[self.idx] = (smooth, hz)
            val = np.tile(smooth, min(reps, 10))

        # Find integer periods
        self.one_cycle = fs / hz
        
        if type_ in ("sin", "laser", "noise"):
            if int(hz) != 0:
                max_size = int(int(hz) * self.one_cycle) # working one... at least without freq shift
                val = val[:max_size]
        else:
            max_size = int(reps * self.one_cycle)
            if self.op != "wavetable":
                if max_size > self.sample_rate:
                    val = np.tile(val[:int(self.one_cycle)], 2)
                else:
                    val = val[:max_size]
            else:
                max_size = int(self.one_cycle * self.original_wavetable_length / hz)
                print(self.one_cycle, max_size, self.original_wavetable_length, reps)
                self.original_wavetable_length = max_size
                val = val[:min(max_size, self.original_wavetable_length)]
                # if max_size > self.original_wavetable_length:
                #     val = val[:self.original_wavetable_length]
                # else:
                #     val = val[:max_size]

        while val.shape[0] < self.block_size:
            val = np.append(val, val)
        
        return val
    
    def stop_note(self):
        if not self.envelope.release_done:
            self.envelope.start_release()
            min_delay = self.delay_buf.size // self.block_size
            self._delay_ends_in = self.delay_length * min_delay
        self.stopped = True
    
    def is_active(self):
        if self.stopped:
            return False
        return True
    
    def renew(self):
        self.stopped = False
        self.envelope = ADSR(self.attack * self.sample_rate, self.decay * self.sample_rate, self.sustain, self.release * self.sample_rate)
        self.position = int((self.phase/360) * self.one_cycle)
        if self.op == "wavetable":  # restart, from the beginning (modified by phase)
            self.waveform = self.func(self.hz, type_=self.op) * self.rel_vel
            self.waveform = np.roll(self.waveform, -int((self.phase/360) * self.one_cycle))
    
    def render(self):
        is_stopped = self.envelope.release_done or self.stopped
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
                if self.position is not None:
                    self.position += self.block_size
                
                if self.op == "wavetable" and self.position >= self.original_wavetable_length:
                    if self.op_extra_params.get("one_shot"):
                        buf[self.position - self.original_wavetable_length:] = 0
                        self.stop_note()
                        self.stopped = True
                    else:
                        if not self.stopped:
                            self.position = 0
                            self.envelope.reset()
            
            buf *= self.envelope.render(self.block_size)
            if self.lfo is not None:
                lfo = self.lfo.render()
                lfo = 1 + lfo
                buf *= lfo

        else:
            buf = np.zeros(self.block_size, dtype=np.float32)
        

        if self.filter is not None:
            buf = self.filter(buf)

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

    def __init__(self, stack=[],
                 filters=[], lfos=[],
                 delay_seconds=0.0, delay_length=0, delay_feedback=0,
                 sample_rate=44100, block_size=448):
        self.synths = {}
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.synth_stack = stack
        self.filters = filters
        self.lfos = lfos

        self.delay_seconds = delay_seconds
        self.delay_feedback = delay_feedback
        self.delay_length = delay_length  # If there is delay, we do a min of 1 iteration of that delay (e.g. 1sec if delay_seconds==1) OR this many. AKA, how many echos

    def set_sample_rate_block_size(self, rate, block_size):
        self.sample_rate = rate
        self.block_size = block_size
        self.delay_buf = np.zeros(max(int(self.delay_seconds * self.sample_rate), self.block_size), dtype=np.float32)

    def start_note(self, note, vel):
        if note not in self.synths:
            self.synths[note] = [deepcopy(synth) for synth in self.synth_stack]

                #Synth(op="sin", rel_vel=1.0, filter=Filter.bandpass()),

                # Synth(detune_coarse=0, detune=0, op="sin", rel_vel=0.5, attack=20, decay=2000, sustain=0.4, release=20, delay_feedback=0, delay_mix=0),
                # Synth(detune_coarse=0, detune=0, op="gen", gen_op=Synth.square, rel_vel=1.0, attack=1, decay=3, sustain=0.6, release=1.1, delay_feedback=0, delay_mix=0,
                #       freq_mod=Synth(detune_coarse=0, detune=0, op="gen", gen_op=Synth.square, rel_vel=0.5, attack=1, decay=1, sustain=0.4, release=0.0001, delay_feedback=0, delay_mix=0), freq_mod_freq=2, freq_mod_velocity=0.00005),
                
                # Kind of a train
                # Synth(op="saw", rel_vel=0.2, attack=0.001, decay=0.1, sustain=0.2, release=0.1, lfo=Synth(op="square", rel_vel=1.0), lfo_freq=5, lfo_velocity=1.0),
                # Synth(op="noise", rel_vel=0.5, attack=0.3, decay=0.5, sustain=0.5, release=0.3, lfo=Synth(op="square", rel_vel=1.0), lfo_freq=5, lfo_velocity=0.8),
                # Synth(op="gen", gen_op=Synth.noise, rel_vel=0.05, attack=0.01, decay=0.2, phase=45, sustain=0.3, release=0.1),

                
                
                # Synth(detune_coarse=0, detune=0, op="sin", rel_vel=0.5, attack=20, decay=2000, sustain=0.4, release=20, delay_seconds=0.5, delay_length=5, delay_feedback=0.5, delay_mix=0,
                #       lfo=Synth(op="tri", rel_vel=0.9, attack=20, decay=2000, sustain=1.0, release=20, delay_seconds=0.2, phase=90, delay_length=2, delay_feedback=0.3, delay_mix=0.2), lfo_freq=13, lfo_velocity=1.0,
                #       ),
                # Synth(detune_coarse=-24, op="sin", rel_vel=0.5, attack=20, decay=2000, sustain=0.7, release=20),
                # Synth(detune_coarse=-12, op="saw", rel_vel=0.5, attack=200, decay=200, sustain=0.5, release=20000),
                # Synth(detune_coarse=0, detune=-1, op="saw", phase=-10, attack=200, decay=200, sustain=0.5, release=100),
                # Synth(detune=0, op="saw", attack=200, decay=200, sustain=0.5, release=2000),
                # Synth(detune_coarse=0, detune=1, op="saw", phase=10, attack=200, decay=200, sustain=0.5, release=10000)
            for i, synth in enumerate(self.synths[note]):
                synth.set_sample_rate_block_size(self.sample_rate, self.block_size, i)
                synth.start_note(note, vel)
        else:
            for synth in self.synths[note]:
                synth.renew()
    
    def stop_note(self, note):
        if note in self.synths:
            for synth in self.synths[note]:
                synth.stop_note()
    
    def render(self):
        buf = np.zeros(self.block_size, dtype=np.float32)
        for _, synths in self.synths.items():
            for synth in synths:
                this = synth.render()
                if this is not None:
                    buf += this
        
        for lfo, _, _ in self.lfos:
            lfo = lfo.render()
            lfo = 1 + lfo
            buf *= lfo
        
        for filter in self.filters:
            buf = filter(buf)
        
        if self.delay_feedback > 0:
            self.delay_buf = np.roll(self.delay_buf, -self.block_size)
            buf += self.delay_buf[:self.block_size] * self.delay_feedback
            self.delay_buf[-self.block_size:] = buf

        return buf

presets = {
    "basic": {
        "stack": [(Synth, {"op": "sin", "rel_vel": 1.0})],
        "filters": [(Filter.resonant, {"w": 200, "Q": 50, "drywet": 0.5})],
        "lfos": [ 
            {"synth": (Synth, {"op": "sin", "rel_vel": 1.0}), "freq": 4, "vel": 0.2},
            {"synth": (Synth, {"op": "tri", "rel_vel": 1.0}), "freq": 10, "vel": 0.4},
        ]
    },
    "cello": {
        "stack": [
            (Synth, {"detune_coarse": 0, "detune": 0, "op": "randomwalk", "rel_vel": 0.9, "attack": 0.1, "decay": 0.3, "phase": 90, "sustain": 0.5, "release": 2, "delay_seconds": 0.05, "delay_length": 2, "delay_feedback": 0.0, "delay_mix": 0,
                    "op_extra_params": {"std": 2, "filter_size": 20, "filter_poly_degree": 5}}),
            (Synth, {"detune_coarse": 0, "detune": -10, "op": "randomwalk", "rel_vel": 0.6, "attack": 0.05, "decay": 0.1, "phase": 0, "sustain": 0.2, "release": 0.01, "delay_seconds": 0.05, "delay_length": 2, "delay_feedback": 0.0, "delay_mix": 0,
                    "op_extra_params": {"std": 0.1}}),
            (Synth, {"detune_coarse": 0, "detune": 11, "op": "randomwalk", "rel_vel": 0.3, "attack": 0.01, "decay": 0.2, "phase": 45, "sustain": 0.3, "release": 0.1, "delay_seconds": 0.05, "delay_length": 2, "delay_feedback": 0.0, "delay_mix": 0,
                    "op_extra_params": {"std": 0.2}})
        ],
        "filters": [(Filter.lowpass, {"cutoff": 4000, "order": 4, "drywet": 0.5})], "lfos": [{"synth": (Synth, {"op": "sin", "rel_vel": 1.0}), "freq": 10, "vel": 0.1}],
        "delay_seconds": 0.15, "delay_length": 2, "delay_feedback": 0.12
    }
}

def get_preset(name):
    preset = presets[name]
    synth_stack = []
    for stack_item in preset["stack"]:
        cls, params = stack_item
        synth_stack.append(cls(**params))
    filter_stack = []
    for filter in preset["filters"]:
        cls, params = filter
        filter_stack.append(cls(**params))
    lfo_stack = []
    for lfo in preset["lfos"]:
        synth, params = lfo["synth"]
        lfo_stack.append((synth(**params), lfo["freq"], lfo["vel"]))
    preset.pop("stack")
    preset.pop("filters")
    preset.pop("lfos")
    return {
        "stack": synth_stack,
        "filters": filter_stack,
        "lfos": lfo_stack,
        **preset
    }

# poly = Poly(**get_preset("cello"))
poly = Poly(
    stack=[Synth(op="wavetable", rel_vel=1.0, attack=0, decay=0, sustain=1.0, release=0,
                 op_extra_params={"one_shot": False, "start": 0.1, "end": 0.9,
                                  "find_closest": False, "orig_freq": 100, "sample": pyphonic.getDataDir() + "/glockenspiel.pkl"})],
    filters=[]#Filter.lowpass(cutoff=4000, order=4, drywet=0.5)]
    )

# Or, to define one from scratch:
# poly = Poly(
#     stack=[Synth(op="sin", rel_vel=1.0)],
#     filters=[Filter.resonant(w=200, Q=50, drywet=0.5)],
#     lfos=[(Synth(op="sin", rel_vel=1.0), 4, 0.1),
#          (Synth(op="tri", rel_vel=1.0), 10, 0.4)],
# )

started = None

def process_npy(midi_messages, audio):
    global started
    num_samples = len(audio[0])

    if started is None:
        poly.set_sample_rate_block_size(pyphonic.getSampleRate(), num_samples)
        started = True
        for i, filter in enumerate(poly.filters):
            filter.set_sample_rate_block_size(pyphonic.getSampleRate(), num_samples, i)
        for i, lfo in enumerate(poly.lfos):
            lfo_synth, freq, vel = lfo
            lfo_synth.set_sample_rate_block_size(pyphonic.getSampleRate(), num_samples, i)
            lfo_synth.start_freq(freq, vel)

    for m in midi_messages:
        if m.type == "note_on":
            if m.note < 20:
                continue
            poly.start_note(m.note, m.velocity/10)
        elif m.type == "note_off":
            poly.stop_note(m.note)
    
    render = poly.render()
    out = np.stack((render, render), axis=0, dtype=np.float32)
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
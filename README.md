# Pyphonic

![Publish workflow](https://github.com/tomgrek/pyphonic/actions/workflows/python-publish.yml/badge.svg) ![Docs workflow](https://github.com/tomgrek/pyphonic/actions/workflows/pages-publish.yml/badge.svg) 

This is the Python library for the Pyphonic VST plugin.

Docs: https://tomgrek.github.io/pyphonic/

The VST streams audio and midi to some server; the server responds with some processed audio.

This library is (one implementation of) the server component.

## Quickstart

##### Super Quick Demo

```bash
python -c "import pyphonic; from pyphonic.demo import process;  pyphonic.start(process, 8020)"
```

Or if you want to use `numpy` (recommended):

```bash
python -c "import pyphonic; from pyphonic.demo_numpy import process_npy;  pyphonic.start(process_npy, 8020)"
```

Enter `127.0.0.1:8020` in the VST and you'll hear synthesized tones for each MIDI note you press.

##### Here's an example that simply echoes back the audio received from the server:

```python
import pyphonic

def process(midi, audio):
    return midi, audio

PORT = 8020
pyphonic.start(process, PORT)
```

Enter the url http://127.0.0.1:8020 in the VST and voila, you have a perfectly useless plugin.

##### Here's an example of a dynamic gain plugin

```python
import pyphonic

def process(midi, audio):
    if pyphonic.getBPM() > 140:
        return [x * 1.1 for x in audio]
    return midi, [0.0] * len(audio)

PORT = 8020
pyphonic.start(process, PORT)
```

## Next Steps

YMMV with network audio, particularly if you're running this server on a different computer than the VST.

That's why the Pyphonic VST also provides the ability to then take the exact same Python code and run it _within the VST_. In other words, the remote setup is great for POC and debugging, the next step is to run it in the VST itself. (Optional third step is then to translate the Python code to C++ but that's on you).

Remotely, you can use any third party Python lib installed in your environment (e.g. `PyTorch`). In the VST, currently, `numpy`, `scipy`, and `librosa` (which includes `numba` and `scikit-learn`) are offered.

## Docs

## Included Demos

1. `pyphonic.demo` / `pyphonic.demo_numpy` - a simple sine wave synth
2. `pyphonic.arp` - a MIDI arpeggiator
3. `pyphonic.butterworth` - a configurable high/low/bandpass filter
4. `pyphonic.sampler` - a wavetable synth or "ROMpler", demonstrating pitch shifting
5. `pyphonic.stretcher` - a time stretching wavetable synth
6. (TODO) `pyphonic.saturator` - a saturator/distortion effect
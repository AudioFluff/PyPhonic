# Pyphonic

![Publish workflow](https://github.com/tomgrek/pyphonic/actions/workflows/python-publish.yml/badge.svg) ![Docs workflow](https://github.com/tomgrek/pyphonic/actions/workflows/pages-publish.yml/badge.svg) 

This is the Python library for the Pyphonic VST plugin.

![Plugin screenshot](docs/plugin_standalone.png)

Docs: https://tomgrek.github.io/pyphonic/

The VST streams audio and midi to some server; the server responds with some processed audio.

This library is (one implementation of) the server component.

## Where do I get the VST?

The VST is not yet released. It's in the final stages of development with release expected in May 2024; you can sign up at https://audiofluff.com to get notified when it's released.

## Quickstart

##### Super Quick Demo

```bash
python -c "import pyphonic; from pyphonic.demo import process;  pyphonic.start(process, 8020)"
```

An example using NumPy (recommended over basic Python):

```bash
python -c "import pyphonic; from pyphonic.sampler import process_npy;  pyphonic.start(process_npy, 8020)"
```

An example using PyTorch:

```bash
python -c "import pyphonic; from pyphonic.torch_saturator import process_torch;  pyphonic.start(process_torch, 8020)"

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
    left, right = audio[0], audio[1]
    if pyphonic.getBPM() > 140:
        return midi, [[x * 1.1 for x in left], [x * 1.1 for x in right]]
    return midi, [[0.0 for _ in left], [0.0 for _ in right]]

PORT = 8020
pyphonic.start(process, PORT)
```

## Next Steps

YMMV with network audio, particularly if you're running this server on a different computer than the VST.

That's why the Pyphonic VST also provides the ability to then take the **exact same Python code** and run it _within the VST_. In other words, the remote setup is great for POC and debugging, the next step is to run it in the VST itself. (Optional third step is then to translate the Python code to C++ but that's on you).

Remotely, you can use any third party Python lib installed in your environment (e.g. `PyTorch`). In the VST, currently, `numpy`, `scipy`, `torch` and `librosa` (which includes `numba` and `scikit-learn`) are offered.

## Docs

## Included Demos

1. `pyphonic.demo` - a simple polyphonic sine wave synth
2. `pyphonic.arp` - a beat sync'd minor triad MIDI arpeggiator
3. `pyphonic.butterworth` - a configurable high/low/bandpass filter
4. `pyphonic.sampler` - a wavetable synth or "ROMpler", demonstrating pitch shifting
5. `pyphonic.stretcher` - a time stretching wavetable synth
6. `pyphonic.torch_noise` - to be charitable, it adds a vinyl crackle or tape hiss to the audio
7. `pyphonic.torch_saturator` - a saturator/distortion effect, nice warmth on EDM drums
8. (TODO) `pyphonic.reverb` - a reverb effect
9. (TODO) `pyphonic.delay` - a beat sync'd delay effect
10. (TODO) `pyphonic.pitcher` - a realtime pitch shift and time stretch effect for audio
11. (TODO) `pyphonic.deverb` - A deep learning model trained to remove reverb
12. (TODO) `pyphonic.source_separation` - A [deep learning model](https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html#sphx-glr-tutorials-hybrid-demucs-tutorial-py) trained to separate music into drums, bass, vocals and other

## Status

The VST is fully functional, rarely crashes, and stays quite performant even under heavy load. Before releasing it (hopefully to the KVR community), I need to 1. make a few more presets that push performance boundaries and check edge cases - dogfooding, basically - 2. write more docs and 3. make a Windows installer. A Mac installer will follow (and Linux if there's interest).

As at early May 2024 I reckon this should be this month.

## Development

Contributions welcome!

In particular, the VST plugin automatically pulls `presets.json` and uses that to populate the presets dropdown. If you come up with a great preset and want to share it, please consider making a pull request.

### How to add to presets.json

```
import json
f = open('my_python_file.py').read()
json.dump(f, open("ready_to_copy_paste_into_presets.json", "w"))
```
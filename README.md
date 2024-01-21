# Pyphonic

![Publish workflow](https://github.com/tomgrek/pyphonic/actions/workflows/python-publish.yml/badge.svg) ![Docs workflow](https://github.com/tomgrek/pyphonic/actions/workflows/pages-publish.yml/badge.svg) 

This is the Python library for the Pyphonic VST plugin.

Docs: https://tomgrek.github.io/pyphonic/

The VST streams audio and midi to some server; the server responds with some processed audio.

This library is (one implementation of) the server component.

## Quickstart

##### Here's an example that simply echoes back the audio received from the server:

```python
import pyphonic
from pyphonic import state  # this is a dynamic object with e.g. `state.bpm`

def process(midi, audio):
    return audio

PORT = 8020
pyphonic.start(process, PORT)
```

Enter the url http://127.0.0.1:8080 in the VST and voila, you have a perfectly useless plugin.

##### Here's an example of a dynamic gain plugin

```python
import pyphonic
from pyphonic import state

def process(midi, audio):
    if state.bpm > 140:
        return [x * 1.1 for x in audio]
    return [0.0] * len(audio)

PORT = 8020
pyphonic.start(process, PORT)
```

## Next Steps

YMMV with network audio, particularly if you're running this server on a different computer than the VST.

That's why the Pyphonic VST also provides the ability to then take the exact same Python code and run it _within the VST_. In other words, the remote setup is great for POC and debugging, the next step is to run it in the VST itself. (Optional third step is then to translate the Python code to C++ but that's on you).

Remotely, you can use any third party Python lib installed in your environment (e.g. `PyTorch`). In the VST, currently, `numpy` and `scipy` are offered.

## History

There's been a 3 year gap between Pyphonic 0.9.1 and now. This time, it's much cleaner, and now the setup is reversed so the VST is no longer a server, instead it's a client transmitting to the Python server. Performance is much improved.
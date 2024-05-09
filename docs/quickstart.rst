QuickStart
==========

Whether you're network streaming or running Python locally inside the VST, your Python code is the same. A ``process`` function must exist, accepting the args ``midi`` and ``audio``. (That function
can alternatively be called ``process_npy`` if you want audio as a numpy array or ``process_torch`` for a PyTorch tensor, where the shape of the array/tensor is (num_channels, num_samples).) The
process function must return a tuple of (midi, audio) where midi is a list of any length, and audio is the same shape/format as the input audio.

Here's a simple example that just passes audio through:

.. code-block:: python

   def process(midi, audio):
       return midi, audio

Here's one that applies gain:

.. code-block:: python

   def process(midi, audio):
       for channel in range(len(audio)):
           for sample in range(len(audio[channel])):
               audio[channel][sample] *= 0.5
       return midi, audio

Here's the same using NumPy:

.. code-block:: python

   def process_npy(midi, audio):
       audio *= 0.5
       return midi, audio

It does not have to be the same array:

.. code-block:: python

   def process_npy(midi, audio):
       new_audio = np.zeros_like(audio)
       new_audio += audio
       return midi, new_audio * 0.5

Neither does it have to even use the incoming audio or MIDI:

.. code-block:: python

   def process_torch(midi, audio):
       return [], torch.randn_like(audio) * 0.1

You can get the transport position and tempo:

.. code-block:: python

   import pyphonic

   def process_npy(midi, audio):
       if pyphonic.getTransport()["beat"] == 1 and pyphonic.getBPM() > 130:
           print("Moar kick")
           audio *= 1.1
       return midi, audio

Note that:

1. Print statements will show in your terminal if you're running it networked, and in the VST if you're running Python natively there,

2. If you want to do something like, boost the first 10ms only of the first beat, and only when it's the first bar, ``getTransport()`` returns bar and ticks too. Obviously beats per bar depends on your time signature, whereas a tick is always a 1000th of a beat. See the `function docstring <http://localhost:8000/reference.html#pyphonic.getTransport>`__.

And you can modify the MIDI as you like:

.. code-block:: python

   def process(midi, audio):
       for msg in midi:
            if msg.type == "note_on":
                msg.velocity = 127
       return midi, audio

Note that the MIDI messages returned by your code are displayed in the VST.

Installation
------------

The Windows installer takes care of everything, just load up your DAW and a new VST3 will be listed. The libraries `librosa`, `numpy`, `torch`, `torchaudio`, `transformers`, and `scipy` are automatically installed into the Python environment as part of the installation. `librosa` also pulls in `numba` and `scikit-learn`. The `pyphonic` library is also there too, it's a native part of the plugin. If you want to use other libraries, you can install them using `pip`, but be sure to ``python -m pip`` using the Python included in the distribution (it's in the folder `python312` in the installation directory).

To stream over the network and back to the VST, ``pip install pyphonic``.

Usage
-----

Networked
^^^^^^^^^

.. code-block:: shell

   python -c "import pyphonic; from my_file import process; pyphonic.start(process, 8888)"

Where:

`8888` is the port you're running on

`my_file` is the name of your Python file containing the `process` function.

This will start a long-lived process which streams and buffers audio and MIDI between the VST and your function.

In the VST, enter `127.0.0.1:8888` in the address box and hit Connect. You'll get feedback about a successful connection both in the terminal where Python's running, and in the VST itself, and you're good to go. `Ctrl-C` interrupts the Python process and disconnects the VST; press it again quickly if you actually want to quit. The `pyphonic.start` function listens for changes to `my_file` and reloads it automatically, but you'll need to reconnect on the VST side.

Native
^^^^^^

Enter Python code directly into the VST's text box and hit Run. If there's an error in your code it'll pop up a message with a trace, you'll need to press Stop, fix it, then run again.
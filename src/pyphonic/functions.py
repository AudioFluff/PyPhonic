class _State:
    """State of the audio engine"""
    sample_rate = 44100
    block_size = 441
    num_channels = 2
    bpm = 120
    sample_num = 0
    bar = 1
    beat = 0
    ticks = 0
    is_playing = False
    min = 0.0
    max = 0.0
    rms = 0.0

_state = _State()

def getSampleRate():
    """
    Returns the sample rate of the audio stream
    """
    return _state.sample_rate

def getBlockSize():
    """
    Returns the block size of the audio stream. This is the number of samples
    per block of audio - each call to the process() function will contain this
    many samples (per channel).
    """
    return _state.block_size

def getNumChannels():
    """
    Returns the number of channels in the audio stream. This is typically 1 for
    mono audio and 2 for stereo audio.
    """
    return _state.num_channels

def getBPM():
    """
    Returns the current BPM (beats per minute) of the incoming data. This is
    set in the DAW, or defaults to 120 in the standalone plugin.
    """
    return _state.bpm

def getTransport():
    """
    Returns the current transport state of the DAW. Properties are:
    - sample_num: the current sample number
    - bar: the current bar number
    - beat: the current beat number
    - ticks: the current tick number
    - is_playing: whether the DAW is currently playing
    """
    return {
        "sample_num": _state.sample_num,
        "bar": _state.bar,
        "beat": _state.beat,
        "ticks": _state.ticks,
        "is_playing": _state.is_playing
    }

def getSignalStats():
    """
    Returns the current signal statistics of the incoming audio. Properties are:
    - min: the minimum value of the audio signal
    - max: the maximum value of the audio signal
    - rms: the root mean square value of the audio signal

    Note that stats are calculated over the audio as a whole, not per-channel -
    in other words, the max is the highest value whether left or right. The
    stats are only true for the current block of audio, there's no sliding windows.

    Also note that the stats are calculated on the incoming audio before any
    processing/synthesis by Pyphonic.
    """
    return {
        "min": _state.min,
        "max": _state.max,
        "rms": _state.rms
    }
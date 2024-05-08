import os
from pathlib import Path
import platform
import platformdirs

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

def getBPM():
    """
    Returns the current BPM (beats per minute) of the incoming data. This is
    set in the DAW, or defaults to 120 in the standalone plugin.
    """
    return _state.bpm

def getTransport():
    """
    Returns the current transport state of the DAW.

    Attributes of the returned dict: 
        ``sample_num``: the current sample number

        ``bar``: the current bar number

        ``beat``: the current beat number

        ``ticks``: the current tick number

        ``is_playing``: whether the DAW is currently playing

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
    Returns the current signal statistics of the incoming audio.
    
    Attributes of the returned dict:
        ``min``: the minimum value of the audio signal

        ``max``: the maximum value of the audio signal

        ``rms``: the root mean square value of the audio signal

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

def getDataDir():
    """
    Returns the directory where Pyphonic can store data. This is a directory
    that is guaranteed to be writable by the plugin, and is unique to the
    current user. This is useful for storing samples, presets, and other
    user-specific data.

    On Windows this is typically ``C:/Users/<username>/AppData/Roaming/AudioFluff/PyPhonic``
    """
    dir_ = Path(platformdirs.user_data_dir("PyPhonic", "AudioFluff", roaming=True))
    if "WSL" in platform.platform():
        paths = os.environ.get("PATH", "").split(":")
        for path in paths:
            if "AppData" in path:
                path = Path(path.split("AppData")[0])
                dir_ = path / "AppData" / "Roaming" / "AudioFluff" / "PyPhonic"
                dir_ = dir_.resolve()
    return str(dir_)

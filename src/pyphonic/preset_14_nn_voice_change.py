# speecht5_voice_change
# SpeechT5 model converts voice to another speaker.
# This will take a long time to initialize at first run (downloading weights).
# Output will start when input has been quiet for WAIT_FOR_QUIET blocks.
# As is, *THIS NEEDS CUDA*. You can modify it to run on CPU.
# Consider increasing your DAW's latency/buffer size for better results.

import threading
from pathlib import Path

import librosa
import pyphonic
import numpy as np
import torch

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan

MALE = "cmu_us_bdl_arctic-wav-arctic_a0009.npy"
FEMALE = "cmu_us_slt_arctic-wav-arctic_a0508.npy"
WAIT_FOR_QUIET = 20

checkpoint = "microsoft/speecht5_vc"
preprocessor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForSpeechToSpeech.from_pretrained(checkpoint).cuda()
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").cuda()

block_num = 0
blocks = [[]]
output_block_num = 0
output_block_ptr = 0
output_blocks = [[]]
quiet_for = 0
event = threading.Event()
ready_to_trigger = True

speaker_embedding = np.load(Path(pyphonic.getDataDir()) / FEMALE)
speaker_embedding = torch.tensor(speaker_embedding, requires_grad=False).unsqueeze(0).cuda()

def process_audio(sampling_rate, waveform):
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform.T)
    if sampling_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16000)
    waveform = torch.tensor(waveform, requires_grad=False)
    return waveform

def predict(audio):
    waveform = process_audio(44100, audio)
    inputs = preprocessor(audio=waveform, sampling_rate=16000, return_tensors="pt")
    if inputs["input_values"].shape[1] < 1024:
        return
    inputs = inputs["input_values"].contiguous().cuda()
    print(inputs.device)
    with torch.no_grad():
        speech = model.generate_speech(inputs, speaker_embedding, vocoder=vocoder)
    speech = speech.contiguous().cpu().numpy().astype(np.float32)
    return speech

def processor_thread():
    global block_num, blocks, event
    global output_block_num
    while True:
        event.wait()
        if not len(blocks[block_num]):
            continue
        
        signal_to_do_processing_on = np.concatenate(blocks[block_num])
        
        wave = predict(signal_to_do_processing_on)
        if wave is None:
            continue
        wave = librosa.resample(wave, orig_sr=16000, target_sr=44100)

        output_blocks[output_block_num] = wave
        block_num += 1
        blocks.append([])
        event.clear()
        print(block_num)
        blocks[block_num - 1] = None

processor = threading.Thread(target=processor_thread)
processor.daemon = True
processor.start()

def process_npy(midi, audio):
    global quiet_for, output_block_num, output_block_ptr, output_blocks
    global event, block_num, ready_to_trigger

    num_channels, num_samples = audio.shape

    if num_channels == 2:
        ready_audio = (audio[0] + audio[1]) / 2.0
    else:
        ready_audio = audio[0]
    
    if pyphonic.getSignalStats()["max"] < 0.01:
        quiet_for += 1
        if quiet_for > WAIT_FOR_QUIET and ready_to_trigger:
            quiet_for = 0
            event.set()
    else:
        blocks[block_num].append(ready_audio)
        quiet_for = 0
    
    if len(output_blocks[output_block_num]):
        if output_block_ptr + num_samples >= len(output_blocks[output_block_num]):
            output_block_num += 1
            output_blocks.append([])
            output_block_ptr = 0
            retval = output_blocks[output_block_num][output_block_ptr:]
            pad = num_samples - len(retval)
            retval = np.concatenate([retval, np.zeros([pad,])], dtype=np.float32)
            output_blocks[output_block_num - 1] = None
            ready_to_trigger = True
        else:
            retval = output_blocks[output_block_num][output_block_ptr:output_block_ptr+num_samples]
            output_block_ptr += num_samples
            ready_to_trigger = False

        if num_channels == 2:
            return midi, np.stack([retval, retval])
        else:
            return midi, np.expand_dims(retval, 0)

    return midi, np.zeros_like(audio)

# Credits:
# Matthijs Hollemans https://huggingface.co/spaces/Matthijs/speecht5-vc-demo
# Ao, Wang et al https://arxiv.org/abs/2110.07205, https://huggingface.co/mechanicalsea/speecht5-vc
# Microsoft https://github.com/microsoft/SpeechT5
# CMU Festvox http://www.festvox.org/cmu_arctic/
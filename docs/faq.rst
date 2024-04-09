FAQ
=====

1. Can it use Python threads or asyncio?

Yes to both. Performance gains might be limited because of the GIL but this can be improved by using NumPy and PyTorch.

2. Can it use Python multiprocessing?

Yes.

3. Can it use CUDA?

Yes, CUDA is supported. By default, tensors passed in to the `process_torch(midi, audio)` function will be on the CPU and must be
returned on the CPU. But you can put them to the GPU and process them there.

4. Does NumPy/Torch use _real_ threads?

Yes they do and can really run in parallel.

5. What is the format for network audio?

It streams 32 bit floats. We found that the performance gains from int16 were negligible when running on localhost.

6. Does it support PyTorch JIT?

Yes, but you'll have to write the Python wrapper code. We thought about having first class support in the VST for drag-n-dropping checkpoints,
but it seems PyTorch is moving more towards `torch.compile`.

7. Does it support `torch.compile`?

Unfortunately, no. PyTorch 2.2.0 doesn't support `torch.compile` with Python 3.12 (which is used by the VST) yet. You can try, but
your code will throw an error. You can always upgrade the PyTorch version used by the VST using `python.exe -m pip` from a command
prompt window, if they release an update. We will try to keep the VST up-to-date with the latest PyTorch version and are keenly watching
developments here.
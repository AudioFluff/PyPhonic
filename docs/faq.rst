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
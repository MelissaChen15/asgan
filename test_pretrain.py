import torch
from torch import Tensor

model = torch.hub.load('RF5/simple-asgan', 'asgan_hubert_sc09_6')
model = model.eval()
# The below returns a batch of (4, 16000) one second waveforms 
# that you can directly save as .wav files.
audio = model.unconditional_generate(4)
print(audio.shape)

import torch
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print('device is:', device)
import torch
print(torch.xpu.is_available())   # should print True
print(torch.__version__)

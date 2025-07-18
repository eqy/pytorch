import torch

for i in range(1024):
    s = i + 1
    a = torch.randn(max(1, s  % 32), s, max(1, s % 512), max(1, s % 512), device='cuda', dtype=torch.half)
    torch.nn.functional.conv2d(a, a)
    print(i)

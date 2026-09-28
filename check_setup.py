import platform
import torch

print("Machine:", platform.machine())
print("Python/PyTorch setup")
print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

x = torch.tensor([1.0, 2.0, 3.0], device=device)

print("Original:", x)
print("Times two:", x * 2)
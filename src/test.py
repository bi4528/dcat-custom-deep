import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
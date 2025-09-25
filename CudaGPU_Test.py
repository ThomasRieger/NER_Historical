# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
import torch
print(torch.version.cuda)        # Should print CUDA version
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print "NVIDIA GeForce RTX 3060"

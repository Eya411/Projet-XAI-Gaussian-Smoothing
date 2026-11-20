
import sys
import torch
import torchvision
import numpy
import scipy
import pandas
import matplotlib
import skimage
import statsmodels

print("=" * 55)
print("VERIFICATION DE L'ENVIRONNEMENT XAI")
print("=" * 55)

print("Python :", sys.version)
print("Executable :", sys.executable)
print("PyTorch :", torch.__version__)
print("TorchVision :", torchvision.__version__)
print("NumPy :", numpy.__version__)
print("SciPy :", scipy.__version__)
print("Pandas :", pandas.__version__)
print("Matplotlib :", matplotlib.__version__)
print("Scikit-image :", skimage.__version__)
print("Statsmodels :", statsmodels.__version__)

print("\nCUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
    print("CUDA PyTorch :", torch.version.cuda)

    x = torch.rand(1000, 1000, device="cuda")
    y = x @ x

    print("Test calcul GPU :", y.mean().item())
    print("Test GPU : REUSSI")
else:
    print("ATTENTION : GPU CUDA non disponible")

print("=" * 55)

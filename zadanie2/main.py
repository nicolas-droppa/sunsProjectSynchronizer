import torch

def checkForCuda():
    print("Torch version:", torch.__version__)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")


checkForCuda()


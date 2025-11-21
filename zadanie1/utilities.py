import torch


def checkForCuda():
    """
    Checks for torch version and whether CUDA is available
    """
    print("Torch version:", torch.__version__)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
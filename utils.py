import numpy as np
import torch

def to_tensor(array: np.ndarray):
    return torch.FloatTensor(array)

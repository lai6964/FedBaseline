import os
import torch.nn as nn
import torch.nn.functional as F

def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


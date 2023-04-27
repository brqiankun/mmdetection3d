import torch.nn as nn

from ..builder import VOXEL_ENCODERS

@VOXEL_ENCODERS.register_module()
class BR_HardVFE(nn.Module):
    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass


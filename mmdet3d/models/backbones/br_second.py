import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import BACKBONES

@BACKBONES.register_module()
class BR_SECOND(BaseModule):
    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  #should return a tuple
        pass
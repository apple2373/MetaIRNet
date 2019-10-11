import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Mixup(nn.Module):
    def __init__(self):
        '''
        mixup baseline to swap ImageFusionNet. 
        '''
        super(Mixup, self).__init__()
        
    def forward(self, original,generated):
        #took from https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py
        weight = torch.tensor(np.random.beta(1.0, 1.0),dtype=original.dtype,device=original.device)
        mixed = weight*original + (1-weight)*generated
        return mixed
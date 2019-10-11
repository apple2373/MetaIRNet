import torch
import torch.nn as nn
import numpy as np

class ImageMixer(nn.Module):
    def __init__(self,fusenet):
        super(ImageMixer, self).__init__()
        self.fusenet = fusenet

    def forward(self, support_inputs, support_labels, support_gens):
        '''
        support_inputs:  
        support_labels:  
        support_gens: shape is (nshot,num_gen,channel,h,w)
        '''
        nshot,num_gen,channel,h,w = support_gens.shape
        #num_gen means number of gallery images per original
        support_augmented = []
        for i in range(num_gen):
            gallerys = support_gens[:, i, :, :, :].view(nshot, channel, h, w)
            gallerys = self.fusenet(support_inputs,gallerys)
            support_augmented.append(gallerys)
        
        support_augmented = torch.cat(support_augmented)
        support_labels_augmented = torch.cat([support_labels]*num_gen)
                
        return support_augmented,support_labels_augmented
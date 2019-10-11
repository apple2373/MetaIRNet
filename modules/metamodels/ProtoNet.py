import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import pairwise_distances

class ProtoNet(nn.Module):
    def __init__(self,distance="l2"):
        super(ProtoNet, self).__init__()
        self.distance = distance
            
    def forward(self,support_embs, support_labels, query_embs):
        '''
        support_embs: (nway*nshot,feature_dim) features of support set
        support_labels: (nway*nshot,) labels of support
        query_embs: (nway*nshot,feature_dim) features of query set
        ''' 
        #compute prototype
        #this is just to take average of n_shot example per class
        nway = support_labels.unique().size(0)
        nshot = int(support_embs.size(0)/nway)
        #note that this computation assume some structure in the dataset
        support_embs = support_embs[support_labels.sort()[1]]#.sort()[1] is indices of sorted 
        prototypes = support_embs.view(nway,nshot,-1).mean(dim=1)
        #feature_dim = support_embs.size(1)
        #assert prototypes.shape == (nshot,feature_dim)
        
        distances = pairwise_distances(query_embs, prototypes, self.distance)
        scores = -distances
        
        return scores
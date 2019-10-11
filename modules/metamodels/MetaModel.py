# This is taken and modified from https://github.com/wyharveychen/CloserLookFewShot/blob/bac8d1c6ed5f5afbf638255ab43e3da99a2d992f/methods/protonet.py
#also https://github.com/oscarknagg/few-shot/blob/master/few_shot/proto.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils import pairwise_distances
import pandas as pd

class MetaModel(nn.Module):
    def __init__(self,feature,classifier,mixer=None,normalize_embs=False):
        super(MetaModel, self).__init__()
        '''
        feature: feature extractir
        classifier: fewshot meta-leaninig classifier
        mixer: image fusione net
        normalize_embs: normalize the feature or not
        '''
        self.feature = feature
        self.classifier = classifier
        self.mixer = mixer
        self.normalize_embs = normalize_embs
        
    def forward(self,inputs,labels,nway,nshot,nquery,inputs_generated=None,augmentations=[],print_final_nshot=False):
        #reindex the labels from 0 as in-place operation
        self.reindex_labels_(labels)
        
        #separate input and labels into support and query set
        support_inputs = inputs[:nway*nshot]
        query_inputs = inputs[nway*nshot:]
        support_labels = labels[:nway*nshot]
        query_labels = labels[nway*nshot:]
        support_inputs_generated = None
        if inputs_generated is not None:
            support_inputs_generated = inputs_generated[:nway*nshot]
            assert len(support_inputs_generated) == len(support_inputs)
            
        #peform data augmentation for inputs
        if self.mixer is not None:
            support_aug,support_aug_labels = self.mixer(support_inputs,support_labels,support_inputs_generated)
            support_inputs,support_labels = self.reorganize_support_set(support_inputs,support_labels,support_aug,support_aug_labels)  
        elif "generated" in augmentations:
            #flip data augmentation
            assert support_inputs_generated is not None
            assert support_inputs_generated.size(1) == 1#only naug=1 is implmeneted.
            support_aug = support_inputs_generated[:,0,:,:,:]
            support_aug_labels = support_labels
            support_inputs,support_labels = self.reorganize_support_set(support_inputs,support_labels,support_aug,support_aug_labels)
        if "flip" in augmentations:
            #flip data augmentation
            support_aug = torch.flip(support_inputs,(3,))
            support_aug_labels = support_labels
            support_inputs,support_labels = self.reorganize_support_set(support_inputs,support_labels,support_aug,support_aug_labels)

        #extract feature for support and query set
        embeddings = self.embed_samples(torch.cat([support_inputs,query_inputs]))
        support_emb = embeddings[:support_inputs.size(0)]
        query_emb = embeddings[support_inputs.size(0):]

        #peform data augmentation for embeddings/feature
        if "gaussian" in augmentations:
            support_aug_emb = support_emb + 0.01*torch.randn(support_emb.shape,device=inputs.device)
            support_aug_labels = support_labels
            support_emb,support_labels = self.reorganize_support_set(support_emb,support_labels,support_aug_emb,support_aug_labels)

        #finally classify query set using augmented support set
        outputs = self.classifier(support_emb,support_labels,query_emb)

        if print_final_nshot:
            print("final nshot: %d"%(support_emb.size(0)/nway))
            
        return outputs,query_labels

    def embed_samples(self,inputs, lengths= None):
        '''
        inputs: Input samples of few shot classification task. 
            shape is (samples, chanel, frames, h, w)
        lengths: list of number of frames of each video.  this is  only for evaluation time where we use multple chuncks from a video clip.
        '''         
        # Embed all samples
        #if the length is not given, only one clip is extracted from video.
        if lengths is None:
            embeddings =  self.feature(inputs)
        #but if length is given, multiple clips are extracted, so we have to average it
        else:
            #average the feature vectors per video
            total_len = 0
            embeddings = []
            for clip_len in lengths:
                embeddings_clip = self.feature(inputs[total_len:total_len+clip_len]).mean(dim=0)
                embeddings.append(embeddings_clip)
                total_len += clip_len
            #assert total_len == x.shape[0]
            embeddings = torch.stack(embeddings)
            #assert embeddings.shape[0] == len(lengths)
        
        if self.normalize_embs:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings
    
    def reindex_labels_(self,labels):
        """
        labels has to be int
        [2,1,3,3,5,1,2] into [1, 0, 2, 2, 3, 0, 1] 
        however, mapping could be different based on torch.unique behavior
        this is inplace opearation
        """
        #label_map = {idx.item():i for i,idx in enumerate(torch.unique(labels))}
        label_map = {idx:i for i,idx in enumerate(pd.unique(labels.cpu().numpy()))}
        for i in range(labels.size(0)):
            labels[i].fill_(label_map[labels[i].item()])
            
    def reorganize_support_set(self,support,support_labels,support_aug,support_aug_labels,sort_labels=False):
        #add augmented support into original support set and order it again 
        #support can be either support_input or support_emb
        if support is not None:
            support = torch.cat([support,support_aug])
            support_labels = torch.cat([support_labels,support_aug_labels])
        else:
            support = support_aug
            support_labels = support_aug_labels
        if sort_labels:
            support_labels_sorted_values,support_labels_sorted_indices = support_labels.sort()
            support = support[support_labels_sorted_indices]
            support_labels = support_labels_sorted_values
        return support,support_labels
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torchvision
import random

class WrapImagePandasDataset(Dataset):
    '''
    Wrap the ImagePandasDataset so that it can return generated images too
    '''
    def __init__(self, dataset, imgname2genroot,num_gen = 1,
                 max_num_gen=10,independent_transform=False,transform_gen = None):
        self.dataset = dataset
        assert num_gen <= max_num_gen
        self.n_aug = num_gen
        self.max_n_aug = max_num_gen
        self.imgname2genroot = imgname2genroot
        self.random_hflip = False
        self.num_classes = dataset.num_classes
        
        #this transform will crop at the same place and apply random horizontal flip for both
        transforms_new= []
        for transform in dataset.transform.transforms:
            tname = transform.__class__.__name__
            if tname == "RandomCrop":
                print("currently our implementation generates images with center \
                                        crop in advancece, so we cannot do this!")
                assert False
            elif tname == "RandomHorizontalFlip":
                #we need to do this complex way because random flip has to be applied syncronized for original and generated image
                self.random_hflip = True
            else:
                transforms_new.append(transform)
        self.transform = torchvision.transforms.Compose(transforms_new)


        if transform_gen is None:
            self.transform_gen = transform
        else:
            self.transform_gen = transform_gen
    
    def __getitem__(self, i):
        img_path = self.dataset.get_img_path(i)
        label_idx = self.dataset.get_label_idx(i)
        img = self.dataset.loader(img_path)
        img = self.transform(img)
        # img.shape = (3,h,w)
        
        img_ori_name = img_path.split("/")[-1]
        img_gen_paths = self.sample_gen_img_path(img_ori_name)
        img_gens = torch.stack([self.transform_gen(self.dataset.loader(path)) for path in  img_gen_paths])
        
        if self.random_hflip:
            if random.random() < 0.5:
                img = torch.flip(img,(2,))
                img_gens = torch.flip(img_gens,(3,))
        
        return {"input":img,"label":label_idx,"generated":img_gens}
    
    def sample_gen_img_path(self,source_img_name):
        gen_img_root = self.imgname2genroot[source_img_name]
        return [os.path.join(gen_img_root,"img_iter499_batch%d.jpg"%i) \
                for i in np.random.permutation(self.max_n_aug)[0:self.n_aug]]
    
    def __len__(self):
        return len(self.dataset)
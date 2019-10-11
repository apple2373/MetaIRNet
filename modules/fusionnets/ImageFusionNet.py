import torch
import torch.nn as nn
from  modules.layers.UpsampleDeterministic import upsample_deterministic

class ImageFusionNet(nn.Module):
    '''
    Class to mix a pair of (original, generated) images. 
    '''
    def __init__(self, img_encoder, img_gen_encoder, feature_dim, block_size=3):
        '''
        img_encoder: cnn to encode original input image
        img_gen_encoder: cnn to encode generated image
        feature_dim: feature dimension after concat.
        '''
        super(ImageFusionNet, self).__init__()
        self.img_encoder = img_encoder
        self.img_gen_encoder = img_gen_encoder
        self.fc = nn.Linear(feature_dim,block_size*block_size)
        
    def compute_mix_weight(self, x, x_gen):
        '''
        takes two images whose shape is (batch,3,hight,width)
        then outputs (batch,3,3) weights for fusion
        '''
        x = self.img_encoder(x)
        x_gen = self.img_gen_encoder(x_gen)
        
        x = x.view(x.size(0), -1)
        x_gen = x_gen.view(x_gen.size(0), -1)

        #output shape is (batch,3,3)
        h = torch.cat([x, x_gen], dim=1)
        h = self.fc(h).view(-1,3,3)
        
        return h
    
    def upsample(self,weight,size):
        '''
        Args:
            weight: weight to combine. Shape is (batch,3,3)
            size: output resolution. 
        Return:
            weight_upsampled: shape is (batch,1,size,size)
        '''
        if size[0]==224 and size[0]==224 and weight.size(1) == 3 and weight.size(2)==3:
            pass
        else:
            raise NotImplementedError("currently it's hard-coded for image size == 224 x 224 and weight == 3 x 3")

        return  upsample_deterministic(weight.unsqueeze(1),upscale=75)[:,:,0:224,0:224]
        
    def forward(self, x, x_gen):
        #assert x_probe.shape == x_gallery.shape
        weight_ = self.compute_mix_weight(x, x_gen)
        size = (x.size(2), x.size(3))#(h,w) tuple
        
        # broadcastable for channels
        weight = self.upsample(weight_,size)
        mixed = weight * x + (1 - weight) * x_gen
                    
        return  mixed
    
#         visualize=True
#         if visualize:
#             assert weight.size(1)==1
#             #from IPython import embed;embed()
#             import os
#             import numpy as np
#             import seaborn as sns
#             import matplotlib.pyplot as plt
#             import torchvision
#             from datetime import datetime
            
#             saveroot = "./results/"
#             if not os.path.exists(saveroot):
#                 os.mkdir(saveroot)
#             time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            
#             def denormalize(img):
#                 #img.shape should be (c,h,w)
#                 mean = [0.485, 0.456, 0.406]
#                 std = [0.229, 0.224, 0.225]
#                 for i in range(img.shape[0]):
#                     img[i] = (img[i]*std[i] + mean[i])
#                 return img
            
#             def covert2rgb(img,option="clip"):
#                 #img.shape should be (3,h,w) ranging 0 to 255
#                 for i in range(3):
#                     if img[i].min() < 0 or img[i].max() > 255:
#                         img[i] = np.clip(img[i],0,255)
#                 return img.astype(np.uint8)
            
#             for i in range(len(x)):
#                 img_ori = covert2rgb(255*denormalize(x[i].detach().cpu().numpy()))
#                 plt.imsave(saveroot+time+"%d-original.png"%i, img_ori.transpose([1,2,0]))
#                 img_gen = covert2rgb(255*denormalize(x_gen[i].detach().cpu().numpy()))
#                 plt.imsave(saveroot+time+"%d-generated.png"%i, img_gen.transpose([1,2,0]))
#                 img_w = weight[i].detach().cpu().numpy()
#                 img_w = denormalize(img_w)[0]
#                 sns.heatmap(img_w,center=0,cmap='RdBu_r',yticklabels=False,xticklabels=False)
#                 plt.savefig(saveroot+time+"%d-weight.png"%i,dpi=72, bbox_inches='tight', pad_inches = 0)
                
#                 img = img_w*img_ori + (1-img_w)*img_gen
#                 from IPython import embed;embed()
#                 img = covert2rgb(img)
#                 plt.imsave(saveroot+time+"%d-mixed.png"%i, img.transpose([1,2,0]))
                

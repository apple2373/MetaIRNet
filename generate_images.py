import sys
import os
import time
import argparse
import random
import uuid
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
try:
    from tqdm import tqdm
    tqdm = tqdm
except:
    print("can't import tqdm. progress bar is disabled")
    tqdm = lambda x: x

import torch
import torchvision

##setup imagebackend
from torchvision import get_image_backend,set_image_backend
try:
    import accimage
    set_image_backend("accimage")
except:
    print("accimage is not available")
print("image backend: %s"%get_image_backend())

from torchvision.datasets.folder import default_loader as img_loader
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

# imports from my own script
import utils
from modules.gans.AdaBIGGANLoss import AdaBIGGANLoss
from modules.gans.biggan128config import biggan128config
import modules.gans.biggan as biggan
from modules.gans.AdaBIGGAN import AdaBIGGAN

#for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def argparse_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help = "gpu id")
    parser.add_argument('--dataset', type=str,choices=["cub","nab","miniimagenet"], default="cub", help="dataset name")
    parser.add_argument('--dataset-root', type=str, default=None, help="Default is None, and ../data/<datasetname> is used.")
    parser.add_argument('--save-suffix', type=str, default="-generated", help="suffix to add the name of root dir")
    parser.add_argument('--saveroot',  default = "./data", help='Root directory to make the output directory')
    parser.add_argument('--model',  default = "biggan128-ada", help='biggan128-ada|biggan32-ada')
    parser.add_argument('--biggan-pretrained',  default = "./data/G_ema.pth", help='path to the biggan pretrained model')
    #so, save root will be ../data/<dataset-name>-generated/ in default
    return parser.parse_args()

def setup_model(name,dataset_size,resume=None,biggan_imagenet_pretrained_model_path="./data/G_ema.pth",img_init="zero",class_init="mean",trained_n_classes=1000):
    print("model name:",name)
    if name=="biggan128-ada":
        print("finetune BigGAN128")
        biggan128config['n_classes'] = trained_n_classes
        G = biggan.Generator(**biggan128config)
        G.load_state_dict(torch.load(biggan_imagenet_pretrained_model_path,map_location=lambda storage, loc: storage))
        model = AdaBIGGAN(G,dataset_size=dataset_size,embedding_init=img_init,embedding_class_init=class_init)
    elif name=="biggan32-ada":
        print("finetune BigGAN32")
        biggan128config['G_ch'] = 32
        biggan128config['D_ch'] = 32
        biggan128config['n_classes'] = trained_n_classes
        G = biggan.Generator(**biggan128config)
        G.load_state_dict(torch.load(biggan_imagenet_pretrained_model_path,map_location=lambda storage, loc: storage))
        model = AdaBIGGAN(G,dataset_size=dataset_size,embedding_init=img_init,embedding_class_init=class_init)
    else:
        raise NotImplementedError("%s (model name) is not defined"%name)
    if resume is not None:
        print("resuming trained weights from %s"%resume)
        checkpoint_dict = torch.load(resume)
        model.load_state_dict(checkpoint_dict["model"])
    return model

def setup_optimizer(model,lr_g_batch_stat,lr_g_linear,lr_bsa_linear,lr_embed,lr_class_cond_embed,step,step_facter=0.1):
    #group parameters by lr
    params = []
    params.append({"params":list(model.batch_stat_gen_params().values()), "lr":lr_g_batch_stat})
    if lr_g_linear > 0:
        params.append({"params":list(model.linear_gen_params().values()), "lr":lr_g_linear })
    else:
        for p in model.linear_gen_params().values():
            p.requires_grad = False   
    params.append({"params":list(model.bsa_linear_params().values()), "lr":lr_bsa_linear })
    params.append({"params":list(model.emebeddings_params().values()), "lr": lr_embed })
    params.append({"params":list(model.calss_conditional_embeddings_params().values()), "lr":lr_class_cond_embed})
    
    #setup optimizer
    optimizer = optim.Adam(params, lr=0)#0 is okay because sepcific lr is set by `params`
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=step_facter)
    return optimizer,scheduler

def save_img(img_tensor,save_path):
    ndarr = img_tensor.add_(1.0).div_(2).mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(save_path)
    print("saved",save_path)
    
if __name__=='__main__':
    #fix seed for reproducibility
    np.random.seed(123)
    torch.manual_seed(123)
    random.seed(123)
        
    args = argparse_setup()
    max_iter = 500
    num_gen = 30
    save_gen = 10
    truc = 0.4
    device = utils.setup_device(args.gpu)
    savedir = os.path.join(args.saveroot,args.dataset+args.save_suffix)

    #setup dataset as pandas data frame
    dataset = getattr(__import__("datasets.%s"%args.dataset),args.dataset)
    dataset_root = "./data/%s"%args.dataset
    if args.dataset_root is not None:
        dataset_root = args.dataset_root
    df_dict =  dataset.setup_df(dataset_root)
    dataset_df = pd.concat(df_dict.values()).sort_values("path")
    transform = transforms.Compose([
            transforms.Resize(146),
            transforms.CenterCrop((128,128)),
            transforms.ToTensor(),
        ])
    
    #setup model and loss
    model_ori = setup_model(args.model,
                       dataset_size=1,
                       resume=None,
                        biggan_imagenet_pretrained_model_path=args.biggan_pretrained,
                       img_init="zero",
                       class_init="mean",
                        )
    model_ori.eval()
    criterion = AdaBIGGANLoss(
                scale_per=0.1,
                scale_emd=0.1,
                scale_reg=0,
                normalize_img = 0,
                normalize_per = 0,
                dist_per = "l2",
                dist_img = "l1",
            )
    model_ori = model_ori.to(device)
    criterion = criterion.to(device)
    indices = torch.LongTensor([0]).to(device)

    for i in tqdm(range(0,len(dataset_df))):
        #fix seed for reproducibility
        np.random.seed(123)
        torch.manual_seed(123)
        random.seed(123)
        
        model = deepcopy(model_ori)
        optimizer,scheduler = setup_optimizer(model,
                    lr_g_batch_stat=0.0005,
                    lr_g_linear=0,
                    lr_bsa_linear=0.0005,
                    lr_embed=0.01,
                    lr_class_cond_embed=0.03,
                    step=500,
                    step_facter=0.1)

        img_path = dataset_df.iloc[i]["path"]
        new_img_name = img_path.split("/")[-1]+"."+str(uuid.uuid4())
        img_save_dir = os.path.join(savedir,new_img_name)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
            print("made dir:",img_save_dir)
        img = transform(img_loader(img_path)).unsqueeze(0)
        img = img.to(device)

        model.eval()
        #this has to be eval() even if it's training time
        #because we want to fix batchnorm running mean and var
        #note that we still change batchnrom scale and bias that is generated by linear layer in biggan

        #start trainig loop
        start = time.time()
        for iteration in range(max_iter):
            scheduler.step()

            #embeddings (i.e. z) + noise (i.e. epsilon) 
            embeddings = model.embeddings(indices)
            embeddings_eps = torch.randn(embeddings.size(),device=device)*0.05
            embeddings +=embeddings_eps 

            #forward
            img_generated = model(embeddings)
            loss = criterion(img_generated,img,embeddings,model.linear.weight)

            #compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #fix seed for reproducibility
        np.random.seed(123)
        torch.manual_seed(123)
        random.seed(123)
        with torch.no_grad():
            embeddings = model.embeddings(indices)
            embeddings = embeddings*torch.randint(2,size=(num_gen,120),dtype=embeddings.dtype,device=device)
            embeddings_eps = torch.randn((num_gen,120),device=device)*0.2
            embeddings +=embeddings_eps 
            embeddings = torch.clamp(embeddings,-truc,truc)
            
            #forward
            for i in range(save_gen):
                img_generated = model(embeddings[i].unsqueeze(0))
                img_generated = img_generated[0].cpu()
                save_path = os.path.join(img_save_dir,"img_iter%d_batch%d.jpg"%(iteration,i))
                save_img(img_generated,save_path)

        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
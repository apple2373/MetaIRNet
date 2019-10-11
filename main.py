import os
import sys
import time
from datetime import datetime
import argparse
from copy import deepcopy
import glob
import pandas as pd

try:
    if not os.environ.get("DISABLE_TQDM"):
        from tqdm import tqdm
        tqdm = tqdm
    else:
        print("progress bar is disabled")
except:
    print("can't import tqdm. progress bar is disabled")
    tqdm = lambda x: x

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

##setup imagebackend
from torchvision import get_image_backend,set_image_backend
try:
    import accimage
    set_image_backend("accimage")
except:
    print("accimage is not available")
print("image backend: %s"%get_image_backend())

# imports from my own script
import utils
utils.make_deterministic(123)
from dataloaders.ImagePandasDataset import ImagePandasDataset 
from dataloaders.NShotTaskSampler import NShotTaskSampler
from dataloaders.WrapImagePandasDataset import WrapImagePandasDataset
from metrics.AverageMeter import AverageMeter
from metrics.accuracy import accuracy
from modules.layers.Flatten import Flatten
from modules.layers.Identity import Identity
from modules.metamodels.Baselines import Baselines
from modules.metamodels.ProtoNet import ProtoNet
from modules.metamodels.MetaModel import MetaModel
from modules.fusionnets.ImageFusionNet import ImageFusionNet
from modules.fusionnets.ImageMixer import ImageMixer
from modules.fusionnets.Mixup import Mixup
from modules.backbones.Conv4 import Conv4

import numpy as np
import random
import json

def setup_args():
    parser = argparse.ArgumentParser(description="MetaIRNet")
    parser.add_argument('--dataset', type=str, default="cub", help = "dataset")
    parser.add_argument('--dataset-root', type=str, default=None, help="Default is None, and ../data/<datasetname> is used")
    parser.add_argument('--workers', type=int, default=8, help="number of processes to make batch worker.")
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--nway', default=5, type=int,  help='class num to classify for training. this has to be more than 1 and maximum is the total number of classes')
    parser.add_argument('--nway-eval', default=5, type=int,  help='class num to classify for evaluation. this has to be more than 1 and maximum is the total number of classes')
    parser.add_argument('--nshot'  , default=1, type=int,  help='number of labeled data in each class, same as nsupport') 
    parser.add_argument('--nquery' , default=16, type=int,  help='number of query point per class') 

    parser.add_argument('--resume', type=str, default=None, help="metamodel checkpoint to resume")
    parser.add_argument('--resume-optimizer', type=str, default=None, help="optimizer checkpoint to resume")

    parser.add_argument('--episodes-train', type=int, default=1000,help = "number of episodes per epoch for train" )
    parser.add_argument('--episodes-val', type=int, default=100,help = "number of episodes for val" )
    parser.add_argument('--episodes-test', type=int, default=1000,help = "number of episodes for test" )
    parser.add_argument('--eval-freq', type=int, default=1,help = "evaluate every this epochs" )

    parser.add_argument('--lr', type=float, default=1e-3, help = "learning rate. default  is 0.001")
    parser.add_argument('--steps', default=[5], nargs='+', type=int, help='decrease lr at this point')
    parser.add_argument('--step-facter', type=float, default=0.1, help="facter to decrease learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="number of epochs. if 0, evaluation only mode")

    parser.add_argument('--backbone', type=str,default = "resnet18",choices = ["resnet18","conv4"], help = "feature extraction cnn")
    parser.add_argument('--backbone-pretrained', type=int,default = 1, help = "use pretrained model or not for feature extraction cnn")
    parser.add_argument('--classifier', type=str,default = "protonet",choices = ["protonet","nearest","logistic","softmax"], help = "few-shot classification model")
    parser.add_argument('--augmentations', type=str,default = [None],nargs='+',choices=["generated","flip","gaussian"],help = "baseline static data augmentations")
    parser.add_argument('--mixer', type=str,default = None,choices=["fusion","mixup",None],help = "how to combine original and generated images")
    parser.add_argument('--fusion-pretrained', type=int,default = 1, help = "use pretrained model or not for fusion net")
    parser.add_argument('--naug', type=int,default = 1, help = "number of generated images to use for fusion") 
    
    parser.add_argument('--saveroot',  default = "./experiments/", help='Root directory to make the output directory')
    parser.add_argument('--saveprefix',  default = "log", help='prefix to append to the name of log directory')
    parser.add_argument('--saveargs',default = ["dataset","nway","nshot","classifier","backbone","mixer","augmentations"]
                        ,nargs='+', help='args to append to the name of log directory')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
    return parser.parse_args()

def setup_dataset(args):
    #setup dataset as pandas data frame
    dataset = getattr(__import__("datasets.%s"%args.dataset),args.dataset)
    dataset_root = "./data/%s"%args.dataset
    if args.dataset_root is not None:
        dataset_root = args.dataset_root
    df_dict =  dataset.setup_df(dataset_root)  
    
    dataset_dict = {}
    #key is train/val/test and the value is corresponding pytorch dataset
    for split,df in df_dict.items():
        target_transform = {label:i for i,label in enumerate(sorted(df["label"].unique()))}
        #target_transform is mapping from category name to category idx start from 0
        if split=="train":
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        dataset_dict[split] = ImagePandasDataset(df=df,
                                            img_key="path",
                                            label_key = "label",
                                            transform = transform,
                                            target_transform = target_transform,
                                            )
    return dataset_dict

def setup_dataloader(args,dataset_dic):
    dataloader_dict = {}
    episodes_dict = {"train":args.episodes_train,"val":args.episodes_val,"test":args.episodes_test}
    for split,dataset in dataset_dic.items():
        episodes = episodes_dict[split]
        if split == "train" and args.classifier in ["nearest","logistic","softmax"]:
            #if supervised baseline, don't use nway-kshot sampler
            dataloader_dict[split] = DataLoader(dataset,
                                              batch_size=32, 
                                              shuffle=True,
                                              num_workers=args.workers		 
                                             )
        else:
            if split == "train":
                nway = args.nway
            else:
                nway = args.nway_eval
            dataloader_dict[split] = DataLoader(
                            dataset,
                            batch_sampler=NShotTaskSampler(
                                dataset, 
                                episodes, 
                                args.nshot,
                                nway, 
                                args.nquery,
                            ),
                            num_workers=args.workers,
                        )
            
    #if we need to use generated images, wrap the dataloader to load them too
    if args.mixer is not None or "generated" in args.augmentations:
        glob_path = "./data/%s-generated/*/*.jpg"%args.dataset
#         if args.scratch_gen:
#             glob_path = "../data/%s-gnerated-scratch/*/*.jpg"%(args.dataset)
        imgname2genroot = organize_generated_images(glob_path)
        transfrom_gen = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
        for split,dataset in dataset_dic.items():
            dataloader_dict[split].dataset = WrapImagePandasDataset(dataset,imgname2genroot
                                            ,num_gen=args.naug,transform_gen=transfrom_gen)

    return dataloader_dict

def setup_backbone(name,pretrained=True):
    if name == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)    
        model.fc = Flatten()
        return model
    elif name=="conv4":
        assert pretrained==False
        return Conv4()
    else:
        raise NotImplementedError("this option is not defined")

def setup_classifier(name,feature_dim=None,num_train_classes=None):
    if name == "protonet":
        return ProtoNet()
    elif name in ["nearest","logistic","softmax"]:
        assert feature_dim is not None
        assert num_train_classes is not None
        return Baselines(name,feature_dim,num_train_classes)
    else:
        raise NotImplementedError("this option is not defined")
    return model

def setup_image_mixer(name,pretrained=True):
    if name == None:
        return None
    if name == "fusion":
        print("image fusion net is used")
        img_encoder = torchvision.models.resnet18(pretrained=pretrained)
        img_encoder.fc = Identity()#remove fc layer
        img_gen_encoder = torchvision.models.resnet18(pretrained=pretrained)
        img_gen_encoder.fc = Identity()#remove fc layer
        feature_dim =  512 + 512#notice that this is hard-coded
        fusenet = ImageFusionNet(img_encoder,img_gen_encoder,feature_dim)
        model = ImageMixer(fusenet=fusenet)
        return model
    elif name == "mixup":
        print("mixup is used")
        mixup = Mixup()
        model = ImageMixer(fusenet=mixup)
        return model
    else:
        raise NotImplementedError("this option is not defined",name)            

def organize_generated_images(glob_path):
    df_all = []
    print("loading",glob_path)
    for path in glob.glob(glob_path):
        entry = {}
        entry["img_path"] = path
        entry["img_name"] = path.split("/")[-1]
        entry["source_img_name"] = ".".join(path.split("/")[-2].split(".")[0:-1])
        df_all.append(entry)
    df_all = pd.DataFrame(df_all).sort_values(["img_path"])
    
    df = df_all.query("img_name=='img_iter499_batch0.jpg'")
    
    imgname2genroot = {}
    for i,row in df.iterrows():
        source_img_name = row["source_img_name"]
        gen_img_root = os.path.split(row["img_path"])[0]
        imgname2genroot[source_img_name] = gen_img_root
    return imgname2genroot

def train_one_epoch(dataloader,model,criterion,optimizer,accuracy=accuracy,device=None,print_freq=100,random_seed=None):
    if random_seed is not None:
        #be careful to use this!
        #it's okay to fix seed every time we call evaluate() because we want to have exactly same order of test images
        #HOWEVER, for training time, we want to have different orders of training images for each epoch.
        #to do this, we can set the seed as epoch, for example.
        utils.make_deterministic(random_seed)
        
    since = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train() # Set model to training mode
    
    losses = AverageMeter()
    accs = AverageMeter()

    suprevised_baseline= False
    if hasattr(dataloader.batch_sampler,"n_way"):
        nway = dataloader.batch_sampler.n_way 
        nshot = dataloader.batch_sampler.n_shot
        nquery = dataloader.batch_sampler.n_query
    else:
        suprevised_baseline = True
        
    for i,data in enumerate(tqdm(dataloader)):
        inputs = data["input"].to(device)
        labels = data["label"].to(device)

        if suprevised_baseline:
            #this is a baseline without meta-learning
            inputs = model.embed_samples(inputs)
            outputs = model.classifier(inputs)
            query_labels = labels
        else:
            inputs_generated = None
            if model.mixer is not None:
                inputs_generated = data["generated"].to(device)
            print_final_nshot = False
            if i == 0:
                print_final_nshot = True
            outputs,query_labels = model(inputs,labels,nway,nshot,nquery,
                                         inputs_generated=inputs_generated,print_final_nshot=print_final_nshot)

        loss = criterion(outputs, query_labels)
        acc = accuracy(outputs, query_labels)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure acc and record loss
        losses.update(loss.item(), outputs.size(0))
        accs.update(acc.item(),outputs.size(0))

        if i % print_freq == 0 or i == len(dataloader)-1:
            temp = "current loss: %0.5f "%loss.item()
            temp += "acc %0.5f "%acc.item()
            temp += "| running average loss %0.5f "%losses.avg
            temp += "acc %0.5f "%accs.avg
            print(i,temp)

    time_elapsed = time.time() - since
    print('this epoch took {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return float(losses.avg),float(accs.avg)

def evaluate(dataloader,model,criterion,accuracy,static_augmentations=[],device=None,random_seed=123):
    print("evaluating...")
    if random_seed is not None:
        utils.make_deterministic(random_seed)
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    losses = AverageMeter()
    accs = []
    nway = dataloader.batch_sampler.n_way 
    nshot = dataloader.batch_sampler.n_shot
    nquery = dataloader.batch_sampler.n_query
    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader)): 
            inputs = data["input"].to(device)
            labels = data["label"].to(device)
            inputs_generated = None
            if model.mixer is not None or "generated" in static_augmentations:
                inputs_generated = data["generated"].to(device)
            print_final_nshot = False
            if i == 0:
                print_final_nshot = True
            outputs,query_labels = model(inputs,labels,nway,nshot,nquery,
                                         inputs_generated=inputs_generated,
                                         print_final_nshot=print_final_nshot,
                                         augmentations= static_augmentations)
            loss = criterion(outputs, query_labels)
            acc = accuracy(outputs, query_labels)
            
            losses.update(loss.item(), outputs.size(0))
            accs.append(acc.item())
        
    print("eval loss: %0.5f "%losses.avg)
    acc = float(np.mean(accs))
    conf = float(1.96* np.std(accs)/np.sqrt(len(accs)))
    print("eval acc :%0.5f +- %0.5f"%(acc,conf))
    return float(losses.avg),acc,conf

def main(args):
    since = time.time()
    print(args)
        
    #setup the directory to save the experiment log and trained models
    log_dir =  utils.setup_savedir(prefix=args.saveprefix,basedir=args.saveroot,args=args,
                                   append_args = args.saveargs)
    #save args
    utils.save_args(log_dir,args)
    
    #setup gpu
    device = utils.setup_device(args.gpu)
    
    #setup dataset and dataloaders
    dataset_dict = setup_dataset(args)
    dataloader_dict = setup_dataloader(args,dataset_dict)

    #setup backbone cnn
    backbone = setup_backbone(args.backbone,pretrained = args.backbone_pretrained)
    #setup fewshot classification
    feature_dim = 64 if args.backbone=="conv4" else 512
    num_train_classes = dataset_dict["train"].num_classes
    classifier = setup_classifier(args.classifier,feature_dim,num_train_classes)
    #setup data augmentation model
    mixer = setup_image_mixer(args.mixer,pretrained = args.fusion_pretrained)
    #setup meta-learning model
    model = MetaModel(feature=backbone,classifier=classifier,mixer=mixer)
    #resume model if needed
    if args.resume is not None:
        model = utils.resume_model(model,args.resume,state_dict_key = "model")

    #setup loss
    criterion = torch.nn.CrossEntropyLoss()
        
    #setup optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,amsgrad=True)
    if args.resume_optimizer is not None:
        optimizer = utils.resume_model(optimizer,args.resume_optimizer,state_dict_key = "optimizer")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.step_facter)
    
    #main training
    log = {}
    log["git"] = utils.check_gitstatus()
    log["timestamp"] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log["train"] = []
    log["val"] = []
    log_save_path = os.path.join(log_dir,"log.json")
    utils.save_json(log,log_save_path)
    valacc = 0
    best_val_acc = 0
    bestmodel = model
    for epoch in range(args.epochs):
        print("epoch: %d --start from 0 and end at %d"%(epoch,args.epochs-1))
        lr_scheduler.step()
        loss,acc = train_one_epoch(dataloader_dict["train"],model,criterion,
                        optimizer,accuracy=accuracy,
                        device=device,print_freq=args.print_freq,random_seed=epoch)
        log["train"].append({'epoch':epoch,"loss":loss,"acc":acc})
        
        if epoch%args.eval_freq == 0:
            valloss,valacc,conf = evaluate(dataloader_dict["val"],model,criterion,
                              accuracy=accuracy,device=device)
            log["val"].append({'epoch':epoch,"loss":valloss,"acc":valacc,"95conf":conf})
        
        #if this is the best model so far, keep it on cpu and save it
        if valacc > best_val_acc:
            best_val_acc = valacc
            bestmodel = deepcopy(model)
            bestmodel.cpu()
            save_path = os.path.join(log_dir,"bestmodel.pth")
            utils.save_checkpoint(save_path,bestmodel,key="model")
            save_path = os.path.join(log_dir,"bestmodel_optimizer.pth")
            utils.save_checkpoint(save_path,optimizer,key="optimizer")
            log["best_epoch"] = epoch
            log["best_acc"] = best_val_acc
        
        utils.save_json(log,log_save_path)
            
    #use the best model to evaluate on test set
    loss,acc,conf  = evaluate(dataloader_dict["test"],bestmodel,criterion,accuracy=accuracy,static_augmentations=args.augmentations,device=device)
    log["test"] = {"loss":loss,"acc":acc,"95conf":conf}
    
    time_elapsed = time.time() - since
    log["time_elapsed"] = time_elapsed
    #save the final log
    utils.save_json(log,log_save_path)
    
if __name__ == '__main__':
    args = setup_args()
    main(args)
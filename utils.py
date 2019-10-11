import os
import sys
import json
import random
import numpy as np
import torch

def setup_device(gpu_id):
    #set up GPUS
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ""    
    gpu_id = int(gpu_id)
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print("set CUDA_VISIBLE_DEVICES=%s"%gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device %s"%device)
    return device

def setup_savedir(prefix="",basedir="./experiments",args=None,append_args=[]):
    savedir = prefix
    if len(append_args) > 0 and args is not None: 
        for arg_opt in append_args:
            arg_value = getattr(args, arg_opt)
            savedir +="_"+arg_opt+"-"+str(arg_value)
    else:
        savedir += "exp"
        
    savedir = savedir.replace(" ","").replace("'","").replace('"','')
    savedir = os.path.join(basedir,savedir)
    
    #if exists, append _num-[num]
    i = 1
    savedir_ori = savedir
    while True:
        try:
            os.makedirs(savedir)
            break
        except FileExistsError as e:
            savedir = savedir_ori+"_num-%d"%i
            i+=1
            
    print("made the log directory",savedir)
    return savedir

def save_args(savedir,args,name="args.json"):
    #save args as "args.json" in the savedir
    path = os.path.join(savedir,name)
    with open(path, 'w') as f:
        json.dump( vars(args), f, sort_keys=True, indent=4)
    print("args saved as %s"%path)
        
def save_json(dict,path):
    with open(path, 'w') as f:
        json.dump( dict, f, sort_keys=True, indent=4)
        print("log saved at %s"%path)
        
def resume_model(model,resume,state_dict_key = "model"):
    '''
    model:pytorch model
    resume: path to the resume file
    state_dict_key: dict key 
    '''
    print("resuming trained weights from %s"%resume)
    
    checkpoint = torch.load(resume,map_location='cpu')
    if state_dict_key is not None:
        pretrained_dict = checkpoint[state_dict_key]
    else:
        pretrained_dict = checkpoint
        
    try:
        model.load_state_dict(pretrained_dict)
    except RuntimeError as e:
        print(e)
        print("can't load the all weights due to error above, trying to load part of them!")
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict_use = {}
        pretrained_dict_ignored = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                pretrained_dict_use[k] = v
            else:
                pretrained_dict_ignored[k] = v
        pretrained_dict =pretrained_dict_use
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print("resumed only",pretrained_dict.keys())
        print("ignored:",pretrained_dict_ignored.keys())
        
    return model

def save_checkpoint(path,model,key="model"):
    #save model state dict
    checkpoint = {}
    checkpoint[key] = model.state_dict()
    torch.save(checkpoint, path)
    print("checkpoint saved at",path)
    
def check_gitstatus():
    try:
        import git
    except:
        print("cannot import gitpython ; try pip install gitpython")
        return None
    #from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    #from https://stackoverflow.com/questions/31540449/how-to-check-if-a-git-repo-has-uncommitted-changes-using-python
    #from https://stackoverflow.com/questions/33733453/get-changed-files-using-gitpython/42792158
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        untracked = repo.untracked_files
        changed = [ item.a_path for item in repo.index.diff(None) ]
    except Exception as e:
        print(e)
        return str(e)
        
    return {"hash":sha,"changed":changed,"untracked":untracked}


def make_deterministic(seed,strict=False):
    #https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if strict:
        #https://github.com/pytorch/pytorch/issues/7068#issuecomment-515728600
        torch.backends.cudnn.enabled = False
        print("strict reproducability required! cudnn disabled. make sure to set num_workers=0 too!")
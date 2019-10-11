#this class is trying to do the same thing as the author's implementation
# https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L242-L294 

import torch
import torchvision
import torch.nn as nn


class AdaBIGGAN(nn.Module):
    def __init__(self,generator, dataset_size, embed_dim=120, shared_embed_dim = 128,cond_embed_dim = 20,embedding_init="zero",embedding_class_init="mean"):
        '''
        generator: original big gan generator
        dataset_size: (small) number of training images. It should be less than 100. If more than 100, it's better to fine tune using normal adverserial training
        shared_embed_dim: class shared embedding dim. 
        cond_embed_dim: class conditional embedding dim
        See Generator row 2 in table 4 in the BigGAN paper (1809.11096v2) where Linear(20+129), which means Linear(cond_embed_dim+shared_embed_dim) 
        '''
        super(AdaBIGGAN,self).__init__()
        self.generator = generator
            
        #same as z in the chainer implementation
        self.embeddings = nn.Embedding(dataset_size, embed_dim)
        print("model embedding_init",embedding_init)
        if embedding_init == "zero":
            self.embeddings.from_pretrained(torch.zeros(dataset_size,embed_dim),freeze=False)
        elif embedding_init == "random":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        in_channels = self.generator.blocks[0][0].conv1.in_channels
        self.bsa_linear_scale = torch.nn.Parameter(torch.ones(in_channels,))
        self.bsa_linear_bias = torch.nn.Parameter(torch.zeros(in_channels,))
        
        self.linear = nn.Linear(1, shared_embed_dim, bias=False)
        print("model embedding_class_init",embedding_class_init)
        if embedding_class_init =="mean":
            init_weight = generator.shared.weight.mean(dim=0,keepdim=True).transpose(1,0)
            assert self.linear.weight.data.shape == init_weight.shape
            self.linear.weight.data  = init_weight
            del generator.shared
        elif embedding_class_init == "zero":
            self.linear.weight.data  = torch.zeros(self.linear.weight.data.shape)
        elif  embedding_class_init =="random":
            #https://github.com/nogu-atsu/small-dataset-image-generation/blob/078a26a1489a835595bf87d16168349783e61582/gen_models/ada_generator.py#L262
            #https://docs.chainer.org/en/stable/reference/generated/chainer.initializers.HeNormal.html#chainer.initializers.HeNormal
            #https://discuss.pytorch.org/t/weight-initilzation/157/11
            scale = 0.001 ** 0.5
            fan_out = self.linear.weight.size()[0]
            fan_in = self.linear.weight.size()[1]
            import numpy as np
            std = scale * np.sqrt(2. / fan_in)
            dtype = self.linear.weight.data.dtype
            self.linear.weight.data = torch.tensor(np.random.normal(loc=0.0,scale=std,size=(fan_out,fan_in)),dtype=dtype)
        else:
            raise NotImplementedError()
            
        self.set_training_parameters()
                
    def forward(self, z):
        '''
        z: tensor whose shape is (batch_size, shared_embed_dim) . in the training time noise (`epsilon` in the original paper) should be added. 
        '''
        #originally copied from the biggan repo
        #https://github.com/ajbrock/BigGAN-PyTorch/blob/ba3d05754120e9d3b68313ec7b0f9833fc5ee8bc/BigGAN.py#L226-L251
        #then modified to do the same job in chainer smallgan repo
        #https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L278-L294

        #original note, as original one use `forward(self, z, y)` (notice y)
        ##Note on this forward function: we pass in a y vector which has
        ##already been passed through G.shared to enable easy class-wise
        ##interpolation later. If we passed in the one-hot and then ran it through
        ##G.shared in this forward function, it would be harder to handle.

        #my note
        #here, we *do* make `y` inside forwad function
        #`y` is equivalent to `c` in chainer smallgan repo

        y = torch.ones((z.shape[0], 1),dtype=torch.float32,device=z.device)#z.shape[0] is batch size
        y = self.linear(y)

        # If hierarchical (i.e. use different z per layer), concatenate zs and ys
        if self.generator.hier:
            zs = torch.split(z, self.generator.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            raise NotImplementedError("I don't implement this case")
            ys = [y] * len(self.generator.blocks)

        # First linear layer
        h = self.generator.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.generator.bottom_width, self.generator.bottom_width)
        
        #Do scale and bias (i.e. apply newly intoroduced statistic parameters) for the first linear layer
        h = h*self.bsa_linear_scale.view(1,-1,1,1) + self.bsa_linear_bias.view(1,-1,1,1) 
        
        # Loop over blocks
        for index, blocklist in enumerate(self.generator.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.generator.output_layer(h))
    

    
    def set_training_parameters(self):
        '''
        set requires_grad=True only for parameters to be updated, requires_grad=False for others.
        '''
        #set all parameters requires_grad=False first
        for param in self.parameters():
            param.requires_grad = False
            
        named_params_requires_grad = {}
        named_params_requires_grad.update(self.batch_stat_gen_params())
        named_params_requires_grad.update(self.linear_gen_params())
        named_params_requires_grad.update(self.bsa_linear_params())
        named_params_requires_grad.update(self.calss_conditional_embeddings_params())
        named_params_requires_grad.update(self.emebeddings_params())
        
        for name,param in named_params_requires_grad.items():
            param.requires_grad = True
            
    def batch_stat_gen_params(self):
        '''
        get named parameters to generate batch statistics
        Weight corresponding to "Hyper" in Chainer implementation 
        ```
            blocks.0.0.bn1.gain.weight torch.Size([1536, 148])
            blocks.0.0.bn1.bias.weight torch.Size([1536, 148])
            blocks.0.0.bn2.gain.weight torch.Size([1536, 148])
            blocks.0.0.bn2.bias.weight torch.Size([1536, 148])
            blocks.1.0.bn1.gain.weight torch.Size([1536, 148])
            blocks.1.0.bn1.bias.weight torch.Size([1536, 148])
            blocks.1.0.bn2.gain.weight torch.Size([768, 148])
            blocks.1.0.bn2.bias.weight torch.Size([768, 148])
            blocks.2.0.bn1.gain.weight torch.Size([768, 148])
            blocks.2.0.bn1.bias.weight torch.Size([768, 148])
            blocks.2.0.bn2.gain.weight torch.Size([384, 148])
            blocks.2.0.bn2.bias.weight torch.Size([384, 148])
            blocks.3.0.bn1.gain.weight torch.Size([384, 148])
            blocks.3.0.bn1.bias.weight torch.Size([384, 148])
            blocks.3.0.bn2.gain.weight torch.Size([192, 148])
            blocks.3.0.bn2.bias.weight torch.Size([192, 148])
            blocks.4.0.bn1.gain.weight torch.Size([192, 148])
            blocks.4.0.bn1.bias.weight torch.Size([192, 148])
            blocks.4.0.bn2.gain.weight torch.Size([96, 148])
            blocks.4.0.bn2.bias.weight torch.Size([96, 148])
        ```
        '''
        named_params = {}
        for name,value in self.named_modules():
            if name.split(".")[-1] in ["gain","bias"]:
                for name2,value2 in  value.named_parameters():
                    name = name+"."+name2
                    params = value2
                    named_params[name] = params
                    
        return named_params
       
    def linear_gen_params(self):
        '''
        Fully connected weights in generator
        finetune with very small learning rate
        ```
            linear.weight torch.Size([24576, 20])
            linear.bias torch.Size([24576])
        ```
        '''
        return {"generator.linear.weight":self.generator.linear.weight,
                       "generator.linear.bias":self.generator.linear.bias}

    def bsa_linear_params(self):
        '''
        Statistics parameter (scale and bias) after lienar layer
        This is a newly intoroduced training parameters that did not exist in the original generator
        '''
        return {"bsa_linear_scale":self.bsa_linear_scale,"bsa_linear_bias":self.bsa_linear_bias}

    def calss_conditional_embeddings_params(self):
        '''
        128 dim input as the conditional noise (?)
        '''
        return {"linear.weight":self.linear.weight}


    def emebeddings_params(self):
        '''
        initialized with zero but added with random epsilon for training time
        this is 120 in the BigGAN 128 x 128 while 140 in 256 x 256
        '''
        return  {"embeddings.weight":self.embeddings.weight}

    
if __name__ == "__main__":
    import sys
    sys.path.append("../official_biggan_pytorch/")
    sys.path.append("../")
    from official_biggan_pytorch import utils
    
    import torch
    import torchvision

    parser = utils.prepare_parser()
    parser = utils.add_sample_parser(parser)
    config = vars(parser.parse_args(args=[]))
    
    # taken from https://github.com/ajbrock/BigGAN-PyTorch/issues/8
    config["resolution"] = utils.imsize_dict["I128_hdf5"]
    config["n_classes"] = utils.nclass_dict["I128_hdf5"]
    config["G_activation"] = utils.activation_dict["inplace_relu"]
    config["D_activation"] = utils.activation_dict["inplace_relu"]
    config["G_attn"] = "64"
    config["D_attn"] = "64"
    config["G_ch"] = 96
    config["D_ch"] = 96
    config["hier"] = True
    config["dim_z"] = 120
    config["shared_dim"] = 128
    config["G_shared"] = True
    config = utils.update_config_roots(config)
    config["skip_init"] = True
    config["no_optim"] = True
    config["device"] = "cuda"

    # Seed RNG.
    utils.seed_rng(config["seed"])

    # Set up cudnn.benchmark for free speed.
    torch.backends.cudnn.benchmark = True

    # Import the model.
    model = __import__(config["model"])
    experiment_name = utils.name_from_config(config)
    G = model.Generator(**config).to(config["device"])
    utils.count_parameters(G)

    # Load weights.
    weights_path = "../official_biggan_pytorch/data/G_ema.pth"  # Change this.
    # weights_path = "./data/G.pth"  # Change this.
    G.load_state_dict(torch.load(weights_path))

    model = AdaBIGGAN(G,dataset_size=42)
    model = model.cuda()
    
    batch_size = 4
    
    z = torch.ones((batch_size,140)).cuda()
    
    output = model(z)
    
    assert output.shape == (batch_size,3,128,128)
    
    print("simple test pased!")
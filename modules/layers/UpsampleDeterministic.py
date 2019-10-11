import torch.nn as nn

def upsample_deterministic(x,upscale):
    '''
    x: 4-dim tensor. shape is (batch,channel,h,w)
    output: 4-dim tensor. shape is (batch,channel,self. upscale*h,self. upscale*w)
    '''
    return x[:, :, :, None, :, None].expand(-1, -1, -1, upscale, -1, upscale).reshape(x.size(0), x.size(1), x.size(2)*upscale, x.size(3)*upscale)

class UpsampleDeterministic(nn.Module):
    def __init__(self,upscale=2):
        '''
        Upsampling in pytorch is not deterministic (at least for the version of 1.0.1)
        see https://github.com/pytorch/pytorch/issues/12207
        '''
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        '''
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,upscale*h,upscale*w)
        '''
        return upsample_deterministic(x,self.upscale)
    
    
if __name__ == '__main__':
    #test
    import torch
    upsample_layer = UpsampleDeterministic(upscale=3)
    x = torch.tensor([[1,2],[3,4]]).unsqueeze(0).unsqueeze(0) #(batch,channel,h,w)
    x = upsample_layer(x)
    x_ = torch.tensor([[[[1, 1, 1, 2, 2, 2],
              [1, 1, 1, 2, 2, 2],
              [1, 1, 1, 2, 2, 2],
              [3, 3, 3, 4, 4, 4],
              [3, 3, 3, 4, 4, 4],
              [3, 3, 3, 4, 4, 4]]]])
    x == x_
    assert torch.all(torch.eq(x,x_))
    print("simple test passed!")
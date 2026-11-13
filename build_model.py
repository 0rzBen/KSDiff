import torch
import torch.nn as nn




class LeFuser(nn.Module): # for stage 1
    def __init__(self, LE_model=None, fuse_model=None):
        super(LeFuser, self).__init__()
        self.LE_model = LE_model
        self.fuse_model = fuse_model
        

    def forward(self, lms=None, gt=None, pan=None, cond=None, learn_res=False):  

        prior = self.LE_model(lms=lms, gt=gt, pan=pan, cond=cond, learn_res=learn_res)
        output = self.fuse_model(lms, pan, prior, learn_res=learn_res)


        return output, prior

class LDMFuser(nn.Module): # for stage 2
    def __init__(self, denoise_fn=None, LE_model_2=None, fuse_model=None):
        super(LDMFuser, self).__init__()
        
        self.denoise_fn = denoise_fn
        self.LE_model_2 = LE_model_2
        self.fuse_model = fuse_model
        

    
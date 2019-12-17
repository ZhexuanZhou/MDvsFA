import torch
from torch import nn
import numpy as np

def init_conv2_weights(m):
    if type(m) == torch.nn.Conv2d and m.kernel_size != (1,1):
        shape =m.weight.shape
        array = np.random.normal(0,0.0025,size=shape)
        cx, cy = shape[0]//2, shape[1]//2
        c1 = shape[2]
        c2 = shape[3]
        if c1==c2 :
           for i in range(c1):
               array[cx, cy, i, i] = 1
        else :
           if c1>c2:
              cmax = c1
              for j in range(c2):
                 array[cx, cy, np.int16(j*c1/c2), j] = cmax/c2
           else :
              cmax = c2 
              for i in range(c1):
                 array[cx, cy, i, np.int16(i*c2/c1)] = cmax/c2
        m.weight.data.copy_(torch.from_numpy(array))

def init_linear_weights(tensor, mean=0, std=0.1):
    if type(tensor) == nn.Linear:
        # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/21
            # the implement is same with scipy
        # http://sofasofa.io/forum_main_post.php?postid=1004424
            # the difference between scipy and tf trunced_norm
        size = tensor.weight.shape
        tmp = tensor.weight.new_empty(size + (4,)).normal_()
        valid = (tmp < 2*std) & (tmp > -2*std)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.weight.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.weight.data.mul_(std).add_(mean)
        # tensor.require_grad = True
        tensor.bias.data.fill_(0.1)
import torch
import torch.nn as nn

class DFEU(nn.Module):
    def __init__(self, dim, n_div=4, drop=0.):
        super().__init__()
        self.dim = dim
        self.n_div = n_div  # Parameter for channel splitting ratio restoration
        self.split_dim = dim // n_div  # Calculate the number of channels to process (1/n_div)
        
        # Path 1: Feature extraction only on 1/n_div channels
        self.pconv = nn.Conv2d(self.split_dim, self.split_dim, kernel_size=1)  # Only process partial channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(self.split_dim, self.split_dim, kernel_size=1)
        
        # Path 2: Attention weight generation (only for processed channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(self.split_dim, self.split_dim),
            nn.Sigmoid()
        )
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        F_in = x  # Original input

        x_process, x_remain = torch.split(x, [self.split_dim, self.dim - self.split_dim], dim=1)
        
        # Step 2: Feature extraction on 1/n_div channels
        F = self.pconv(x_process)
        F = self.relu(F)
        F = self.conv1x1(F)
        F = self.drop(F)
        
        a = self.gap(x_process).flatten(1)
        weight = self.linear(a).unsqueeze(-1).unsqueeze(-1)
        
        F_processed = x_process + F * weight  
        F_out = torch.cat([F_processed, x_remain], dim=1)  
        
        return F_out
    
class DFEUBlock(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,  
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 drop=0.):
        super().__init__()
        self.dim = dim
        self.n_div = n_div  
        
        self.conv3x3_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        

        self.dfeu = DFEU(dim, n_div=n_div, drop=drop)
        

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = nn.Conv2d(inc, dim, kernel_size=1)  # Simplified Conv without BN/Act
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward_default

    def forward_default(self, x):

        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        
        x = self.conv3x3_1(x)
        x = self.dfeu(x)
        x = self.conv3x3_2(x)
        
        x = shortcut + self.drop_path(x)
        return x

    def forward_layer_scale(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        
        x = self.conv3x3_1(x)
        x = self.dfeu(x)
        x = self.conv3x3_2(x)
        
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

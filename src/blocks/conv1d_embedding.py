import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
import torch.nn.functional as F
from utilities import get_act

class Conv1d4EBMs(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=128, embed_dims=256, act_type='spike'):
        super().__init__()
        kernel_size = 3
        padding = 1
        groups = 1
        self.proj_conv = nn.Conv1d(in_channels, embed_dims, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dims)
        self.proj_lif = get_act(act_type, tau=2.0, detach_reset=True)
        # self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv1d(embed_dims, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.proj_bn1 = nn.BatchNorm1d(embed_dims)
        self.proj_lif1 = get_act(act_type, tau=2.0, detach_reset=True)

        self.proj_conv2 = nn.Conv1d(embed_dims, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.proj_bn2 = nn.BatchNorm1d(embed_dims)
        self.proj_lif2 = get_act(act_type, tau=2.0, detach_reset=True)

        self.proj_conv3 = nn.Conv1d(embed_dims, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.proj_bn3 = nn.BatchNorm1d(embed_dims)
        self.proj_lif3 = get_act(act_type, tau=2.0, detach_reset=True)


        self.rpe_conv = nn.Conv1d(embed_dims, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.act_loss = 0.0
    
    def cal_act_loss(self, x):
        return torch.sum(torch.abs(x))
        
    def forward(self, x):
        self.act_loss = 0.0
        x = x.squeeze(1)
        x = x.permute(0,2,1).contiguous() # B, N, C -> B, C, N
        x = self.proj_conv(x)
        # x = self.maxpool(x)
        
        x = self.proj_bn(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.proj_lif(x).permute(2,1,0).contiguous() # N, C, B -> B, C, N
        x = self.proj_conv1(x)
        
        x = self.proj_bn1(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.proj_lif1(x).permute(2,1,0).contiguous() # N, C, B -> B, C, N
        x = self.proj_conv2(x)

        x = self.proj_bn2(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.proj_lif2(x).permute(2,1,0).contiguous() # N, C, B -> B, C, N
        x = self.proj_conv3(x)

        x_rpe = x.clone()
        
        x_rpe = self.proj_bn3(x_rpe).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x_rpe = self.proj_lif3(x_rpe).permute(2,1,0).contiguous() # N, C, B -> B, C, N
        x_rpe = self.rpe_conv(x_rpe) # B, C, N 
        x = x + x_rpe

        return x # B, C, N


class Conv1d4EB(Conv1d4EBMs):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=128, embed_dims=256, act_type='spike'):
        super().__init__()

        self.rpe_bn = nn.BatchNorm1d(embed_dims)
        self.rpe_lif = get_act(act_type, tau=2.0, detach_reset=True)
        
    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0,2,1).contiguous() # B, N, C -> B, C, N

        x = self.proj_conv(x)
        x = self.proj_bn(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.proj_lif(x).permute(2,1,0).contiguous() # N, B, C -> B, C, N

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.proj_lif1(x).permute(2,1,0).contiguous() # N, C, B -> B, C, N

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.proj_lif2(x).permute(2,1,0).contiguous() # N, C, B -> B, C, N

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.proj_lif3(x).permute(2,1,0).contiguous() # N, C, B -> B, C, N

        x_rpe = x.clone()
        
        x_rpe = self.rpe_conv(x_rpe) # B, C, N 
        x_rpe = self.rpe_bn(x_rpe).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x_rpe = self.rpe_lif(x_rpe).permute(2,1,0).contiguous() # N, C, B -> B, C, N
        
        x = x + x_rpe

        return x # B, C, N
    

class Tokenizer(nn.Module):
    def __init__(self,
                 img_size_h=128,
                 img_size_w=128,
                 patch_size=64,
                 n_conv_layers=4,
                 in_channels=128,
                 embed_dims=256,
                 **kwargs):
        super(Tokenizer, self).__init__()
        in_planes=embed_dims
        n_filter_list = [in_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [embed_dims]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=3,
                          stride=1,
                          padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) if i == -1 else nn.Identity(),
                nn.BatchNorm1d(n_filter_list[i + 1]),
            )
                for i in range(n_conv_layers)
            ])
        
        self.rpe_conv = nn.Conv1d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)

        # self.apply(self.init_weight) # using default initialization is better

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0,2,1).contiguous() # B, N, C -> B, C, N
        x = self.conv_layers(x)
        x_rpe = x.clone()
        x_rpe = self.rpe_conv(x_rpe)
        x = x + x_rpe
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)

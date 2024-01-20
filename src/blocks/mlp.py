import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import get_act

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_type='spike'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = get_act(act_type if act_type == 'spike' else 'gelu', tau=2.0, detach_reset=True)
        self.fc1_dp = nn.Dropout(drop) if drop > 0. else nn.Identity()

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = get_act(act_type, tau=2.0, detach_reset=True)
        self.fc2_dp = nn.Dropout(drop) if drop > 0. else nn.Identity()
 
        self.c_hidden = hidden_features
        self.c_output = out_features
        self.act_loss = torch.tensor(0.0)
    def forward(self, x):
        B,C,N = x.shape
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).permute(2,1,0).contiguous()
        x = self.fc1_lif(x).permute(2,1,0).contiguous()
        x = self.fc1_dp(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).permute(2,1,0).contiguous()
        x = self.fc2_lif(x).permute(2,1,0).contiguous()
        x = self.fc2_dp(x)
        return x


class ConvFFNMs(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_type='spike'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_bn = nn.BatchNorm1d(in_features)
        self.fc1_lif = get_act(act_type if act_type == 'spike' else 'gelu', tau=2.0, detach_reset=True)
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(hidden_features)
        self.fc2_lif = get_act(act_type, tau=2.0, detach_reset=True)
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.act_loss = 0.0

    def cal_act_loss(self, x):
        return torch.sum(torch.abs(x))
    
    def forward(self, x):
        B,C,N = x.shape
        x = self.fc1_bn(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.fc1_lif(x).permute(2,1,0).contiguous()
        x = self.fc1_conv(x)
        
        x = x = self.fc2_bn(x).permute(2,1,0).contiguous()
        x = self.fc2_lif(x).permute(2,1,0).contiguous()
        x = self.fc2_conv(x)
        
        return x

class LinearFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_type='spike'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_linear  = nn.Linear(in_features, hidden_features)
        self.fc1_ln = nn.LayerNorm(hidden_features)
        self.fc1_lif = get_act(act_type if act_type == 'spike' else 'gelu', tau=2.0, detach_reset=True)

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_ln = nn.LayerNorm(out_features)
        self.fc2_lif = get_act(act_type, tau=2.0, detach_reset=True)
 
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B,C,N = x.shape
        # 
        x = x.permute(0,2,1) # B, N, C
        x = x.reshape(B*N, C)
        x = self.fc1_linear(x)
        x = self.fc1_ln(x)
        x = self.fc1_lif(x)

        x = self.fc2_linear(x)
        x = self.fc2_ln(x)
        x = self.fc2_lif(x)
        x = x.reshape(B, N, self.c_output)
        x = x.permute(0,2,1) # B, C, N
        return x
    
class LinearFFNMs(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_type='spike'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1_ln = nn.LayerNorm(in_features)
        self.fc1_lif = get_act(act_type if act_type == 'spike' else 'gelu', tau=2.0, detach_reset=True)
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        
        self.fc2_ln = nn.LayerNorm(hidden_features)
        self.fc2_lif = get_act(act_type, tau=2.0, detach_reset=True)
        self.fc2_linear = nn.Linear(hidden_features, out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features
    
    def forward(self, x):
        B, C, N = x.shape
        # x = x.reshape(B*N, C)
        x = x.permute(0,2,1) # B, N, C

        x = self.fc1_ln(x)
        x = self.fc1_lif(x)
        x = self.fc1_linear(x)
        
        x = self.fc2_ln(x)
        x = self.fc2_lif(x)
        x = self.fc2_linear(x)

        # x = x.reshape(B, self.c_output, N)
        x = x.permute(0,2,1) # B, C, N
        return x

import numpy as np

import torch
from torch import nn
from torch import fft
from torch.nn import init
from torch.nn import functional as F

from scipy.signal import cont2discrete
from utilities import get_act


def leCunUniform(tensor):
    """ 
        LeCun Uniform Initializer
        References: 
        [1] https://keras.rstudio.com/reference/initializer_lecun_uniform.html
        [2] Source code of _calculate_correct_fan can be found in https://pytorch.org/docs/stable/_modules/torch/nn/init.html
        [3] Yann A LeCun, Léon Bottou, Genevieve B Orr, and Klaus-Robert Müller. Efficient backprop. In Neural networks: Tricks of the trade, pages 9–48. Springer, 2012
    """
    fan_in = init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3. / fan_in)
    init.uniform_(tensor, -limit, limit) # fills the tensor with values sampled from U(-limit, limit)


class LMUFFTCell(nn.Module):

    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):

        super(LMUFFTCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.seq_len = seq_len
        self.theta = theta

        self.W_u = nn.Linear(in_features = input_size, out_features = 1)
        self.f_u = nn.ReLU()
        self.W_h = nn.Linear(in_features = memory_size + input_size, out_features = hidden_size)
        self.f_h = nn.ReLU()

        A, B = self.stateSpaceMatrices()
        self.register_buffer("A", A) # [memory_size, memory_size]
        self.register_buffer("B", B) # [memory_size, 1]

        H, fft_H = self.impulse()
        self.register_buffer("H", H) # [memory_size, seq_len]
        self.register_buffer("fft_H", fft_H) # [memory_size, seq_len + 1]

    def stateSpaceMatrices(self):
        """ Returns the discretized state space matrices A and B """

        Q = np.arange(self.memory_size, dtype = np.float64).reshape(-1, 1)
        R = (2*Q + 1) / self.theta
        i, j = np.meshgrid(Q, Q, indexing = "ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = np.ones((1, self.memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system = (A, B, C, D), 
            dt = 1.0, 
            method = "zoh"
        )

        # To torch.tensor
        A = torch.from_numpy(A).float() # [memory_size, memory_size]
        B = torch.from_numpy(B).float() # [memory_size, 1]
        
        return A, B


    def impulse(self):
        """ Returns the matrices H and the 1D Fourier transform of H (Equations 23, 26 of the paper) """

        H = []
        A_i = torch.eye(self.memory_size).to(self.A.device) 
        for t in range(self.seq_len):
            H.append(A_i @ self.B)
            A_i = self.A @ A_i

        H = torch.cat(H, dim = -1) # [memory_size, seq_len]
        fft_H = fft.rfft(H, n = 2*self.seq_len, dim = -1) # [memory_size, seq_len + 1]

        return H, fft_H


    def forward(self, x):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, input_size]
        """
        batch_size, seq_len, input_size = x.shape

        # Equation 18 of the paper
        u = self.f_u(self.W_u(x)) # [batch_size, seq_len, 1]

        # Equation 26 of the paper
        fft_input = u.permute(0, 2, 1) # [batch_size, 1, seq_len]
        fft_u = fft.rfft(fft_input, n = 2*seq_len, dim = -1) # [batch_size, seq_len, seq_len+1]

        # Element-wise multiplication (uses broadcasting)
        # [batch_size, 1, seq_len+1] * [1, memory_size, seq_len+1]
        temp = fft_u * self.fft_H.unsqueeze(0) # [batch_size, memory_size, seq_len+1]

        m = fft.irfft(temp, n = 2*seq_len, dim = -1) # [batch_size, memory_size, seq_len+1]
        m = m[:, :, :seq_len] # [batch_size, memory_size, seq_len]
        m = m.permute(0, 2, 1) # [batch_size, seq_len, memory_size]

        # Equation 20 of the paper (W_m@m + W_x@x  W@[m;x])
        input_h = torch.cat((m, x), dim = -1) # [batch_size, seq_len, memory_size + input_size]
        h = self.f_h(self.W_h(input_h)) # [batch_size, seq_len, hidden_size]

        h_n = h[:, -1, :] # [batch_size*T, hidden_size]

        return h, h_n
    
    def forward_recurrent(self, x, m_last):
        u = self.f_u(self.W_u(x)) # [batch_size, seq_len, 1]
        # A: torch.Size([512, 512]), m_last: torch.Size([256, 512]), B: torch.Size([512, 1]), u: torch.Size([256, 1])
        m = m_last @ self.A.T + u @ self.B.T  # [batch_size, memory_size]
        input_h = torch.cat((m, x), dim = -1) # [batch_size, seq_len, memory_size + input_size]
        h = self.f_h(self.W_h(input_h)) # [batch_size, seq_len, hidden_size]

        return h, m

class SpikingLMUFFTCell(LMUFFTCell):

    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):

        super(SpikingLMUFFTCell, self).__init__(input_size, hidden_size, memory_size, seq_len, theta)

        if_bn = True
        self.bn_u = nn.BatchNorm1d(1) if if_bn else nn.Identity()
        self.f_u = get_act('spike', tau=2.0, detach_reset=True)
        self.bn_m = nn.BatchNorm1d(memory_size) if if_bn else nn.Identity()
        self.f_m = get_act('spike', tau=2.0, detach_reset=True)
        self.bn_h = nn.BatchNorm1d(hidden_size) if if_bn else nn.Identity()
        self.f_h = get_act('spike', tau=2.0, detach_reset=True)
        self.act_loss = 0.0
    def cal_act_loss(self, x):
        return torch.sum(torch.abs(x))
    def forward(self, x):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, input_size]
        ensure the input and output h are all the spikes
        """
        batch_size, seq_len, input_size = x.shape # B, N, C
        # self.H, self.fft_H = self.impulse()
        # Equation 18 of the paper
        u_spike = self.f_u(self.bn_u(self.W_u(x).transpose(-1,-2)).permute(2,0,1).contiguous()) # [B,N,C]->[B,C,N]->[N,B,C]
        u = u_spike.permute(1,0,2).contiguous() # [N,B,C]->[B,N,C] [batch_size, seq_len, 1]

        # Equation 26 of the paper
        fft_input = u.permute(0, 2, 1) # [batch_size, 1, seq_len]
        # print(fft_input.shape, seq_len)
        fft_u = fft.rfft(fft_input, n = 2*seq_len, dim = -1) # [batch_size, seq_len, seq_len+1]

        # Element-wise multiplication (uses broadcasting)
        # [batch_size, 1, seq_len+1] * [1, memory_size, seq_len+1]
        temp = fft_u * self.fft_H.unsqueeze(0) # [batch_size, memory_size, seq_len+1]

        m = fft.irfft(temp, n = 2*seq_len, dim = -1) # [batch_size, memory_size, seq_len+1]
        m = m[:, :, :seq_len] # [batch_size, memory_size, seq_len]
        m = self.f_m(self.bn_m(m).permute(2,1,0).contiguous()).permute(2,1,0).contiguous()
        m = m.permute(0, 2, 1) # [batch_size, seq_len, memory_size]

        # Equation 20 of the paper (W_m@m + W_x@x  W@[m;x])
        input_h = torch.cat((m, x), dim = -1) # [batch_size, seq_len, memory_size + input_size] # [4*100, 784, 469]

        h = self.f_h(self.bn_h(self.W_h(input_h).transpose(-1,-2)).permute(2,0,1).contiguous()) # [B,N,C]->[B,C,N]->[N,B,C]
        h = h.permute(1,0,2).contiguous() # [N,B,C]->[B,N,C]

        h_n = h[:, -1, :] # [batch_size, hidden_size]

        h_n = h_n.unsqueeze(-1)

        return h, h_n
    def forward_recurrent(self, x, m_last):
        u_spike = self.f_u(self.bn_u(self.W_u(x).transpose(-1,-2)).permute(2,0,1).contiguous()) # [B,N,C]->[B,C,N]->[N,B,C]
        u = u_spike.permute(1,0,2).contiguous() # [N,B,C]->[B,N,C] [batch_size, seq_len, 1]

        m = m_last @ self.A.T + u @ self.B.T  # [batch_size, memory_size]
        # print('x: {}'.format(x.shape), 'm: {}'.format(m.shape))
        input_h = torch.cat((m, x), dim = -1) # [batch_size, seq_len, memory_size + input_size]
        h = self.f_h(self.bn_h(self.W_h(input_h).transpose(-1,-2)).permute(2,0,1).contiguous()) 
        h = h.permute(1,0,2).contiguous() # [batch_size, seq_len, hidden_size]

        return h, m
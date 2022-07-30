import math

import torch.nn as nn
from torch.nn import init

import config
import torch


class epdemicSEIIeR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.p=args.p
        self.time_step=args.time_step


    def forward(self, betaI,betaIe, prev_state):

        outs=[]
        S_index = config.COMPARTMENT.get('S')
        E_index = config.COMPARTMENT.get('E')
        I_index = config.COMPARTMENT.get('I')
        Ie_index = config.COMPARTMENT.get('Ie')
        R_index = config.COMPARTMENT.get('R')
        SNindex = config.COMPARTMENT.get('S_N')
        for step in range(self.time_step):
            beta_i=betaI[:,step,:].reshape(-1,1)
            beta_ie=betaIe[:,step,:].reshape(-1,1)
            out = (beta_i * prev_state[:, step, SNindex].reshape(-1,1) * prev_state[:, step, I_index].reshape(-1,1)+ \
                  beta_ie * prev_state[:, step, SNindex].reshape(-1,1) * prev_state[:, step, Ie_index].reshape(-1,1))*self.p
            outs.append(out)



        prediction=torch.stack(outs,dim=1)
        prediction=torch.squeeze(prediction,dim=-1)
        # print('prediction shape : '+str(prediction.shape))
        return prediction

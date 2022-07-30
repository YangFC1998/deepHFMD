import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init as init
import config

class Embedding(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.feature_size = len(config.INTERACTIVE_LIST)
        self.time_step=args.time_step
        self.hidden_size = args.embedding_hidden_size
        self.dropout_prob=args.embedding_dropout_prob
        self.pass_linear=args.pass_linear
        self.build()
    def build(self):
        self.nnLayer=nn.Sequential(
            nn.Linear(self.time_step * self.feature_size, self.time_step * self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.time_step * self.hidden_size, self.time_step * self.feature_size)
        )
        self.weight=nn.Parameter(torch.Tensor(self.time_step,self.feature_size))
        self.bias=nn.Parameter(torch.Tensor(self.feature_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self,x):

        #remove index
        sample_index=[config.INPUT_LIST.index(j) for j in config.INTERACTIVE_LIST]
        x=x[:,:,sample_index]
        # print(x.shape)
        if self.pass_linear:
            out=self.nnLayer(x.reshape(-1,self.time_step*self.feature_size))
            out=out.reshape(-1,self.time_step,self.feature_size)
        else:
            out=x.reshape(-1,self.time_step,self.feature_size)
        # print('data for weight: {}'.format(out.shape))
        outs=[]
        #compute elementwise prodcondense
        for index in range(out.shape[0]):
            feature_map=out[index,:,:]
            outs.append(torch.sum(torch.mul(feature_map,self.weight),dim=-2)+self.bias)
        emb=torch.stack(outs,0)
        # print('data of weight_map: {}'.format(outs[0].shape))
        # print('result{}'.format(emb.shape))
        return emb




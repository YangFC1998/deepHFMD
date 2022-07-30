import torch
import torch.nn as nn
from torch.nn import functional as F
import config


class FM(nn.Module):
    def __init__(self,args,fea_num):
        super().__init__()
        self.latent_dim=args.FM_latent_size
        self.fea_num=fea_num
        # 定义三个矩阵， 一个是全局偏置，一个是一阶权重矩阵， 一个是二阶交叉矩阵，注意这里的参数由于是可学习参数，需要用nn.Parameter进行定义
        self.w0 = nn.Parameter(torch.zeros([1, ]))
        self.w1 = nn.Parameter(torch.rand([self.fea_num, 1]))
        self.w2 = nn.Parameter(torch.rand([self.fea_num, self.latent_dim]))
    def forward(self, inputs):
        # print('FM input_shape:' + str(inputs.shape))
        # 一阶交叉
        first_order = self.w0 + torch.mm(inputs, self.w1)  # (samples_num, 1)
        # 二阶交叉  这个用FM的最终化简公式
        second_order = 1 / 2 * torch.sum(
            torch.pow(torch.mm(inputs, self.w2), 2) - torch.mm(torch.pow(inputs, 2), torch.pow(self.w2, 2)),
            dim=1,
            keepdim=True
        )  # (samples_num, 1)
        # print('FM output.shape: ' + str((first_order + second_order).shape))
        fm=torch.tanh(first_order+second_order)
        return fm



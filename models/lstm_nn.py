import torch
import torch.nn as nn
from torch.nn import functional as F

import config


class LSTM_NN(nn.Module):
    def __init__(self,args,hidden_size,nnLayer,dropout_prob):
        super().__init__()
        self.input_size=len(config.DNN_LIST)
        self.hidden_size=hidden_size
        self.batch_size=args.batch_size
        self.time_step=args.time_step
        self.nnLayer=nnLayer
        self.dropout_prob=dropout_prob
        self.device=args.device
        self.add_fm=args.add_FM
        self.fm_size=args.FM_size if self.add_fm else 0
        self.build()
    def forward_init(self,batch_size):
        device = torch.device(self.device)
        self.hidden_state=torch.randn(1, batch_size, self.hidden_size).to(device)
        self.cell_state=torch.randn(1,batch_size,self.hidden_size).to(device)



    def build(self):
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size+self.fm_size, self.nnLayer[0]),
            nn.PReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.nnLayer[0], self.nnLayer[1]),
            nn.Softplus(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.nnLayer[1], 1)
        )
    def forward(self,x,fm):
        self.forward_init(x.shape[0])
        #对输入采样
        sample_index = [config.INPUT_LIST.index(j) for j in config.DNN_LIST]
        x = x[:, :, sample_index]
        rnn_out, (h_s, c_s) = self.lstm(x, (self.hidden_state, self.cell_state))
        outs=[]
        for step in range(rnn_out.shape[1]):
            if self.add_fm:
                # print(rnn_out[:,step,:].shape)
                # print(fm.shape)
                feature_map=torch.cat([rnn_out[:,step,:],fm],dim=-1)
            else:
                feature_map=rnn_out[:,step,:]
            outs.append(torch.abs(self.linear(feature_map)))
        out=torch.stack(outs,dim=1)
        # print('rnn output shape: '+str(out.shape))
        return out

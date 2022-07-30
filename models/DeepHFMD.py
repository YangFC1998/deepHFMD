import torch
import torch.nn as nn
from torch.nn import functional as F

import config
from models.lstm_nn import LSTM_NN
from models.epdemic_model import epdemicSEIIeR
from models.Embedding import Embedding
from models.RegionEmbedding import RegionEmbedding

from models.FM import FM



class DeepHFMD(nn.Module):
    def __init__(self,args,hidden_size,nnLayer,dropout_prob):
        super().__init__()
        self.args=args
        self.deep_moduleI=LSTM_NN(args, hidden_size, nnLayer, dropout_prob)
        self.deep_moduleIe=LSTM_NN(args, hidden_size, nnLayer, dropout_prob)
        self.k=args.k
        self.embedding=Embedding(args)
        self.region_embedding=RegionEmbedding(args)
        self.factor_machine_caseXnoaa=FM(args,len(config.INTERACTIVE_LIST))
        self.factor_machine_region=FM(args,len(config.COUNTY_LIST))
        self.epdemic_module=epdemicSEIIeR(args)
    def forward(self,X,prev_state,region):

        emb_region=self.region_embedding(region)
        fm_region=self.factor_machine_region(emb_region)

        emb = self.embedding(X)
        fm_caseXnoaa = self.factor_machine_caseXnoaa(emb)


        fm=torch.cat((fm_caseXnoaa,fm_region),dim=-1)
        # fm=fm_caseXnoaa
        # print('fm.shape: '+str(fm.shape))
        betaI=self.deep_moduleI(X,fm)
        betaIe = self.deep_moduleIe(X, fm)
        # betaIe=betaI*self.k
        out=self.epdemic_module(betaI,betaIe,prev_state)
        return out,betaI,betaIe
    def get_interactive(self):
        v=self.factor_machine_caseXnoaa.w2.data.cpu()
        interactive_map = torch.mm(v, v.T)
        return interactive_map
    def get_region_interactive(self):
        v = self.factor_machine_region.w2.data.cpu()
        interactive_map = torch.mm(v, v.T)
        return interactive_map

def build(args):
    model=DeepHFMD(args,hidden_size=128,nnLayer=[128,64],dropout_prob=0.3)
    return model
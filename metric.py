from math import sqrt

import torch
import torch.optim.lr_scheduler as lr_scheduler

import config
import torch.nn as nn

import util


def get_scheduler(args,optimizer):
    if args.scheduler=='multistep':
        return lr_scheduler.MultiStepLR(optimizer,config.MULTI_STEP,gamma=0.5)
    if args.scheduler=='exponent':
        return lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
    if args.scheduler=='cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=1e-4)
    if args.scheduler=='reduceonloss':
        return lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.35,patience=5)
    if args.scheduler=='cosine_warmup':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20,T_mult=1)
def get_curve_fit_curve_regression(curve1,curve2):
    curve1=util.normalize(curve1)
    curve2=util.normalize(curve2)
    top=pow((curve1-curve2),2).sum()
    bottom=pow(curve1,2).sum()
    return abs(1-sqrt(top/bottom))
def get_curve_fit_curve_substract(curve1,curve2):
    curve1 = util.normalize(curve1)
    curve2 = util.normalize(curve2)
    return abs(curve1-curve2).sum()




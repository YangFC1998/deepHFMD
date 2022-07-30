import argparse
import os
from datetime import datetime
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader
import config
import metric
from datatools import hfmd_datatools
from datatools import noaa_datatools
from dataloader import build_dataset
from models.DeepHFMD import build
from dataloader.pre_process import preprocess
from visual import model_visual as model_visual
from metric import get_scheduler


def train(args,model,optimizer,criterion,scheduler,dataloader,epoch):
    device=torch.device(args.device)
    model.train()

    running_loss = 0
    predictions = []
    labels = []
    for index, (data, compartment,label,region) in enumerate(dataloader):
        if torch.cuda.is_available():
            data = data.to(device)
            compartment=compartment.to(device)
            label = label.to(device)
            region=region.to(device)

        prediction,betaI,betaIe = model(data,compartment,region)
        predictions.extend(prediction.data.cpu().reshape(-1))
        labels.extend(label.data.cpu().reshape(-1))
        loss = criterion(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        if args.sample:
            return
    if args.scheduler == 'reduceonloss':
        scheduler.step(running_loss / len(dataloader))
    else:
        scheduler.step()
    mse = mean_squared_error(predictions, labels)
    mae = mean_absolute_error(predictions, labels)
    print("EPOCH {} LOSS {} MSE {} MAE {}".format(epoch, running_loss, mse, mae))
    return running_loss/len(dataloader)


def validate(args,dataloader,model,criterion):
    device = torch.device(args.device)
    model.eval()
    # 不记录模型梯度信息
    with torch.no_grad():
        running_loss = 0
        for index, (data, compartment, label, region) in enumerate(dataloader):
            if torch.cuda.is_available():
                data = data.to(device)
                compartment = compartment.to(device)
                label = label.to(device)
                region = region.to(device)

            prediction, betaI, betaIe = model(data, compartment, region)
            loss = criterion(prediction, label)
            running_loss+=loss.item()
    return running_loss/len(dataloader)


def test(args,dataloader,time_stamp):
    model = torch.load('savemodels/'+time_stamp+'/deepHFMD')
    model.eval()
    device=torch.device(args.device)
    with torch.no_grad():
        predictions=[]
        labels=[]
        betaIs=[]
        betaIes=[]
        for index, (data, compartment, label,region) in enumerate(dataloader):
            if torch.cuda.is_available():
                data = data.to(device)
                compartment = compartment.to(device)
                label = label.to(device)
                region=region.to(device)

            prediction,betaI,betaIe = model(data, compartment,region)

            prediction=prediction.data.cpu().numpy()/args.scale
            betaI=betaI.data.cpu().numpy()
            betaIe=betaIe.data.cpu().numpy()
            label=label.data.cpu().numpy()/args.scale

            predictions.append(prediction[0,-1])
            betaIs.append(betaI[0,-1])
            betaIes.append(betaIe[0,-1])
            labels.append(label[0,-1])

        mse = mean_squared_error(predictions, labels)
        mae = mean_absolute_error(predictions, labels)

    print('mse {} mae{}'.format(mse, mae))
    model_visual.draw_prediction(labels,predictions,args,time_stamp)
    model_visual.draw_two_beta(betaIs,betaIes,args,time_stamp)
def visual(time_stamp):
    pass
    model = torch.load('savemodels/'+time_stamp+'/deepHFMD')
    interactive_heatmap=model.get_interactive()
    model_visual.draw_heatmap_noaa(interactive_heatmap,config.INTERACTIVE_LIST,time_stamp)
    region_interactive_heatmap = model.get_region_interactive()
    model_visual.draw_heatmap_region(region_interactive_heatmap, [i+370200 for i in config.COUNTY_LIST],time_stamp)

def save_args(args,time_stamp):
    argsDict = args.__dict__
    with open('savemodels/'+time_stamp+'/'+'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
def metrics(args,model,optimizer,criterion,scheduler,train_dataloader,val_dataloader,time_stamp):
    train_loss = []
    val_loss = []
    for step in range(args.epochs):
        train_loss.append(train(args,model,optimizer,criterion,scheduler,train_dataloader,step))
        val_loss.append(validate(args,val_dataloader,model,criterion))

    os.makedirs('savemodels/' + time_stamp + '/')
    plt.figure()
    plt.plot(train_loss, color='#e3b4b8', label='train_loss')
    plt.plot(val_loss, color='#c8adc4', label='val_loss')

    plt.legend()
    plt.savefig('savemodels/' + time_stamp + '/loss.png')
    plt.show()
    torch.save(model, 'savemodels/' + time_stamp + '/deepHFMD')

def main(args):
    time_stamp=datetime.now().strftime('%m-%d-%H-%M-%S')
    print(time_stamp)
    print(args)
    device=torch.device(args.device)
    hfmd_datatools.generate_data(args)
    noaa_datatools.generate_data(args)
    if args.preprocess==True:
        preprocess(args)
    train_dataset=build_dataset(args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    config.YEAR_LIST=[config.TEST_YEAR]
    test_dataset=build_dataset(args)
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=1,shuffle=False,drop_last=False)

    config.YEAR_LIST=[config.EVAL_YEAR]
    val_dataset = build_dataset(args)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)


    model=build(args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = get_scheduler(args, optimizer)
    metrics(args, model, optimizer, criterion, scheduler, train_dataloader,val_dataloader,time_stamp)
    if args.sample:
        return
    test(args,test_dataloader,time_stamp)
    test(args,test_dataloader,"04-17-22-03-45")
    if args.visual:
        visual(time_stamp)
        # visual("04-17-22-03-45")
    # save_args(args,time_stamp)




if __name__=='__main__':
    parser = argparse.ArgumentParser('DeepHFMD training and evaluation script', parents=[config.get_args_parser()])
    args=parser.parse_args()
    main(args)
import calendar

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import config
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import datetime
def out_date(year,day):
    fir_day = datetime.datetime(year,1,1)
    zone = datetime.timedelta(days=day-1)
    return datetime.datetime.strftime(fir_day + zone, "%m-%d")
def draw_heatmap_noaa(heat_map,labels,time_stamp):
    for i in range(heat_map.shape[0]):
        heat_map[i][i] = 0
    labels=['就诊人数','累积治愈','平均气温','平均湿度','风速','降水量','近n日降水情况']
    fig,ax=plt.subplots(figsize=(6, 6))
    sns.heatmap(heat_map,cbar=True,annot=True, vmax=heat_map.max(), square=True, cmap="YlGnBu")
    ax.set_yticklabels(labels, fontsize=8, rotation=360, horizontalalignment='right')
    ax.set_xticklabels(labels, fontsize=8, rotation=60,horizontalalignment='right')
    plt.savefig('savemodels/'+time_stamp+'/heatmap_noaa.png')
    plt.show()
def draw_heatmap_region(heat_map,labels,time_stamp):
    for i in range(heat_map.shape[0]):
        heat_map[i][i] = 0
    labels=['市南区',"市北区","黄岛市","崂山区","李沧区","城阳区","即墨区","胶州市","平度市","莱西市"]
    fig, axis=plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(heat_map, cbar=True, annot=True, vmax=heat_map.max(), square=True, cmap="YlGnBu", ax=axis[0])
    axis[0].set_yticklabels(labels, fontsize=8, rotation=360, horizontalalignment='right')
    axis[0].set_xticklabels(labels, fontsize=8, rotation=60, horizontalalignment='right')
    region=plt.imread('Region.jpg')
    axis[1].imshow(region)
    plt.savefig('savemodels/' + time_stamp + '/heatmap_region.png')
    plt.show()
def draw_two_beta(betaI,betaIe,args,time_stamp):
    time_step = args.time_step
    incubation = args.incubation_days
    year = config.EVAL_YEAR
    year_length = 366 if calendar.isleap(year) else 365
    length = year_length - time_step - incubation - 1
    date = []
    for i in range(length):
        date.append(out_date(year, i+time_step+incubation))
    date = date[0:length:30]


    fig,ax=plt.subplots(1,2,figsize=(12,6))
    ax[0].plot(betaI, color='#61649f', label='infection rate')
    ax[0].set_xticks(np.arange(0,length,30))  # 设置刻度
    ax[0].set_xticklabels(date, rotation=60, fontsize='small')  #
    ax[0].legend()

    plt.setp(ax[0].get_xticklabels(), rotation=60, horizontalalignment='right')
    ax[1].plot(betaIe, color='#a7a8bd', label='latent infection rate')
    ax[1].set_xticks(np.arange(0, length, 30))  # 设置刻度
    ax[1].set_xticklabels(date, rotation=60, fontsize='small')  #
    ax[1].legend()
    plt.savefig('savemodels/' + time_stamp + '/beta.png')
    plt.show()
def draw_prediction(labels,predictions,args,time_stamp):
    time_step=args.time_step
    incubation=args.incubation_days
    year=config.EVAL_YEAR
    year_length=366 if calendar.isleap(year) else 365
    length=year_length-time_step-incubation-1
    date=[]
    for i in range(length):
        date.append(out_date(year, i+time_step+incubation))
    date=date[0:length:30]

    plt.figure()
    plt.plot(labels, color='#8ECFC9', label='ground_truth')
    plt.plot(predictions, color='#FA7F6F', label='prediction')
    plt.xticks(np.arange(0,length,30),date,rotation=85,fontsize=8)
    plt.legend()
    plt.savefig('savemodels/'+time_stamp+'/prediction.png')
    plt.show()



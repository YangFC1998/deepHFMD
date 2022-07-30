import datetime
import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def visual_param(frames,param,unit,description,output_path):
    if os.path.exists(output_path)==False:
        os.makedirs(output_path)
    for i in range(len(frames)):
        frame=frames[i]
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.set_title(description)
        axis.set_xlabel(unit[0])
        axis.set_ylabel(unit[1])
        axis.plot([i for i in range(frame.shape[0])], frame[param], color='#9AC9DB', label=param)
        plt.savefig(output_path+description+str(2010+i)+'.tif')
        # plt.show()




def main():
    pass
    frames=[]
    for i in range(12):
        frames.append(pd.read_csv('../dataset/noaa/'+str(2010+i)+'.csv'))
    visual_param(frames,'TEMP',['time','F°'],'青岛市日平均气温变化','../image/noaa/param in 12 year/TEMP/')
    visual_param(frames,'DEWP',['time','F°'],'青岛市日平均露点变化','../image/noaa/param in 12 year/DEWP/')
    visual_param(frames,'HUMI',['time','%'],'青岛市日相对湿度变化','../image/noaa/param in 12 year/DEWP/')
    visual_param(frames,'PRCP',['time','inch'],'青岛市日平均降水量变化','../image/noaa/param in 12 year/PRCP/')
    visual_param(frames,'MAX',['time','F°'],'青岛市日最高气温变化','../image/noaa/param in 12 year/MAX/')
    visual_param(frames,'MIN',['time','F°'],'青岛市日最低气温变化','../image/noaa/param in 12 year/MIN/')

if __name__=='__main__':
    main()
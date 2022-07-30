import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler


def normalize(data):
    data = np.array(data)
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
def standardscale(data):
    data=np.array(data)
    scaler=StandardScaler()
    return scaler.fit_transform(data.reshape(-1,1)).reshape(-1)

def robustscale(data):
    data=np.array(data)
    scaler=RobustScaler()
    return scaler.fit_transform(data.reshape(-1,1)).reshape(-1)



def smooth_savgol_filter(data,window_length,polyorder):
    return abs(savgol_filter(x=data, window_length=window_length, polyorder=polyorder))
def smooth_interp_spline(data):
    x = np.arange(len(data))
    x_smooth = np.linspace(x.min(), x.max(), len(x))
    y_smooth = make_interp_spline(x, data)(x_smooth)
    return y_smooth
def smooth_moving_average(data,window):
    data=pd.DataFrame(data)
    return data.rolling(window=window, min_periods=1).mean()
def smooth_weighted_moving_average(data,window):
    data=pd.DataFrame(data)
    return data.rolling(window=window, min_periods=1, win_type="cosine").mean()
def smooth_e_weighted_moving_average(data,alpha):
    data=pd.DataFrame(data)
    return data.ewm(alpha=alpha, min_periods=1).mean()

def visual_smooth_effect(data,method,**kwargs):
    smooth=data
    if method==smooth_savgol_filter.__name__:
        smooth=smooth_savgol_filter(data,window_length=kwargs.get('window_length'),polyorder=kwargs.get('polyorder'))
    if method==smooth_moving_average.__name__:
        smooth=smooth_moving_average(data,window=kwargs.get('window'))

    if method==smooth_weighted_moving_average.__name__:
        smooth=smooth_moving_average(data,window=kwargs.get('window'))

    if method==smooth_e_weighted_moving_average.__name__:
        smooth=smooth_e_weighted_moving_average(data,alpha=kwargs.get('alpha'))
    plt.figure()
    plt.plot(data,color='#8ECFC9',label='raw')
    plt.plot(smooth,color='#FA7F6F',label=method)
    plt.show()

def test():
    data=pd.read_excel('dataset/dailyIncrease/'+str(2016)+'.xlsx',sheet_name='diagnosis')['total']
    visual_smooth_effect(data,smooth_moving_average.__name__,window=15)
    visual_smooth_effect(data,smooth_e_weighted_moving_average.__name__,alpha=0.1)
    visual_smooth_effect(data,smooth_savgol_filter.__name__,window_length=15,polyorder=1)


def main():
    pass
    test()

if __name__=='__main__':
    main()







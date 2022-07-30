import pandas as pd
import datetime
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units
from matplotlib import pyplot as plt
import os
#验证数据
def validation_data(frame):
    #step 1 数据长度校验
    print('year: '+frame['DATE'][0].split('-')[0])
    print('data length: '+str(frame.shape[0]))

    length=frame.shape[0]
    err_number=0
    #step2
    print('TEMP :'+str(frame[frame['TEMP']==9999.9].shape[0]))
    print('DEWP :'+str(frame[frame['DEWP']==9999.9].shape[0]))
    print('SLP :'+str(frame[frame['SLP']==9999.9].shape[0]))
    print('STP :'+str(frame[frame['STP']==9999.9].shape[0]))
    print('VISIB :'+str(frame[frame['VISIB']==999.9].shape[0]))
    print('WDSP :'+str(frame[frame['VISIB']==999.9].shape[0]))
    print('MXSPD :'+str(frame[frame['MXSPD']==999.9].shape[0]))
    print('GUST :'+str(frame[frame['GUST']==999.9].shape[0]))
    print('MAX :'+str(frame[frame['MAX']==9999.9].shape[0]))
    print('MIN :'+str(frame[frame['MIN']==9999.9].shape[0]))
    print('PRCP :'+str(frame[frame['PRCP']==99.99].shape[0]))
    print('SNDP :'+str(frame[frame['SNDP']==999.9].shape[0]))

def tofahrenheits(data):
    #Pint 支持这些类型的单位以及它们之间的转换。默认定义文件包括华氏度、摄氏度、开尔文和朗肯，缩写为 degF、degC、degK 和 degR。
    return units.Quantity(data, "degF")
def dewpoint2relative_humidity(temprature_list,dewpoint_list):
    relative_humidity_list=[]
    for i in range(len(temprature_list)):
        temprature=tofahrenheits(temprature_list[i])
        dewpoint=tofahrenheits(dewpoint_list[i])
        #Uses temperature and dewpoint to calculate relative humidity as the ratio of vapor pressure to saturation vapor pressures.
        relative_humidity=relative_humidity_from_dewpoint(temprature,dewpoint)
        relative_humidity_list.append(float(relative_humidity))
    return relative_humidity_list


def generate_model_data(input_path,output_path,args):
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    for i in range(12):
        frame = pd.read_csv(input_path + str(2010 + i) + '.csv',converters={'FRSHTT':str})
        for index, row in frame.iterrows():
            if row['PRCP'] == 99.99:
                frame.loc[index, 'PRCP'] = 0
        # validation_data(frame)
        date=frame['DATE']
        temprature=frame['TEMP']
        dewpoint=frame['DEWP']
        relative_humidity=dewpoint2relative_humidity(temprature,dewpoint)
        wdsp=frame['WDSP']
        max_temprature=frame['MAX']
        min_tempratre=frame['MIN']
        prcp=frame['PRCP']
        frshtt=frame['FRSHTT']
        is_rain=[]
        avg_prcp=[]


        interval=args.noaa_interval
        front = int((interval - 1) / 2)
        rear = int((interval + 1) / 2)

        for j in range(len(date)):
            # 向前搜索三天，向后搜索三天
            rain_days = 0
            all_prcp=0
            pointers = [item for item in range(j - front, j + rear)]
            for index in pointers:
                if index < 0 or index >= len(date):
                    rain_days = rain_days + 0
                    all_prcp=all_prcp+0
                else:
                    rain_days = rain_days + int(frshtt[index].zfill(6)[1])
                    all_prcp=all_prcp+prcp[index]
            is_rain.append(rain_days / interval)
            avg_prcp.append(all_prcp/interval)

        output_frame=pd.DataFrame(zip(date,temprature,dewpoint,relative_humidity,wdsp,max_temprature,min_tempratre,avg_prcp,is_rain),columns=['DATE','TEMP','DEWP','HUMI','WDSP','MAX','MIN','PRCP','RAIN'])
        output_frame.to_csv(output_path+ str(2010 + i) + '.csv',index=False)

def generate_data(args):
    input_path='data/noaa-weatherdata/'
    output_path='dataset/noaa/'

    if args.regenerate_dataset == True:
        generate_model_data(input_path,output_path,args)










def main():
    pass
    for i in range(12):
        frame=pd.read_csv('../data/noaa-weatherdata/'+str(2010+i)+'.csv')
        validation_data(frame)
    generate_model_data('../data/noaa-weatherdata/', '../dataset/noaa/')

if __name__=='__main__':
    main()

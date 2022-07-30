from operator import itemgetter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import util
import config


def normalize(data,go):
    if go:
        return util.normalize(data)
        # return util.standardscale(data)
        # return util.robustscale(data)
    else:
        return data

def get_data_by_year(year, args):
    case_dir, noaa_dir = args.case_path + '/' + args.compartment_method + '/', args.noaa_path
    frame_case = pd.read_excel(case_dir + str(year) + args.compartment_method + '.xlsx', sheet_name='accrued')
    frame_region=pd.read_excel(case_dir+str(year)+args.compartment_method+'.xlsx',sheet_name='diagnosis')
    frame_noaa = pd.read_csv(noaa_dir + str(year) + '.csv')

    region=frame_region[['region_370202','region_370203',
                         'region_370211','region_370212','region_370213','region_370214','region_370215',
                         'region_370281','region_370283','region_370285']].values



    morbidity = frame_case['morbidity'].values
    diagnosis = frame_case['diagnosis'].values

    accrued_s = frame_case['S'].values
    accrued_e = frame_case['E'].values
    accrued_i = frame_case['I'].values
    accrued_ie = frame_case['Ie'].values
    accrued_r = frame_case['R'].values
    accrued_s_n = frame_case['S_N'].values
    accrued_n = frame_case['N'].values
    label = frame_case['label'].values
    length = len(morbidity)
    index = [i for i in range(length)]

    temprature = frame_noaa['TEMP'].values
    dewpoint = frame_noaa['DEWP'].values
    humidity = frame_noaa['HUMI'].values
    wdsp = frame_noaa['WDSP'].values
    max_temprature = frame_noaa['MAX'].values
    min_temprature = frame_noaa['MIN'].values
    prcp=frame_noaa['PRCP'].values
    is_rain = frame_noaa['RAIN'].values

    return {'index': index, 'morbidity': morbidity, 'diagnosis': diagnosis,
            'temprature': temprature, 'dewpoint': dewpoint, 'humidity': humidity, 'wdsp': wdsp,
            'max_temprature': max_temprature, 'min_temprature': min_temprature,'prcp':prcp, 'is_rain': is_rain}, \
           {'label': label, 'S': accrued_s, 'E': accrued_e, 'I': accrued_i, 'Ie': accrued_ie, 'R': accrued_r,
            'S_N': accrued_s_n, 'N': accrued_n},region

def make_data(args):
    input_size = len(config.INPUT_LIST)
    year_list = config.YEAR_LIST
    input_list = config.INPUT_LIST
    compartment_list = config.COMPARTMENT.keys()
    region_count=len(config.COUNTY)
    time_step = args.time_step
    alpha = args.incubation_days
    model_input = np.zeros((1, time_step, input_size))
    model_compartment = np.zeros((1, time_step, len(config.COMPARTMENT)))
    model_label = np.zeros((1, time_step))
    model_region=np.zeros((1,time_step,region_count))
    for year in year_list:
        year_dict, compartment_dict,region_dict = get_data_by_year(year, args)

        morbidity = np.array(year_dict.get('morbidity'), dtype=float)
        diagnosis = np.array(year_dict.get('diagnosis'), dtype=float)
        label = np.array(compartment_dict.get('label'), dtype=float)

        index = np.array(year_dict.get('index'), dtype=float)
        temprature = np.array(year_dict.get('temprature'), dtype=float)
        dewpoint = np.array(year_dict.get('dewpoint'), dtype=float)
        humidity = np.array(year_dict.get('humidity'), dtype=float)
        wdsp = np.array(year_dict.get('wdsp'), dtype=float)
        max_temprature = np.array(year_dict.get('max_temprature'), dtype=float)
        min_temprature = np.array(year_dict.get('min_temprature'), dtype=float)
        prcp=np.array(year_dict.get('prcp'),dtype=float)
        is_rain = np.array(year_dict.get('is_rain'), dtype=float)

        accrued_s = np.array(compartment_dict.get('S'), dtype=float).reshape(1, -1)
        accrued_e = np.array(compartment_dict.get('E'), dtype=float).reshape(1, -1)
        accrued_i = np.array(compartment_dict.get('I'), dtype=float).reshape(1, -1)
        accrued_ie = np.array(compartment_dict.get('Ie'), dtype=float).reshape(1, -1)
        accrued_r = np.array(compartment_dict.get('R'), dtype=float).reshape(1, -1)
        accrued_s_n = np.array(compartment_dict.get('S_N'), dtype=float).reshape(1, -1)
        accrued_n = np.array(compartment_dict.get('N'), dtype=float).reshape(1, -1)

        region=np.array(region_dict,dtype=float)
        #[time_len,region_count,]
        # print(region.shape)
        # print(model_region.shape)



        # 标准化
        norl_dict=config.NORMAL_DICT
        index = normalize(index,norl_dict.get('index')).reshape(1, -1)
        morbidity = normalize(morbidity,norl_dict.get('morbidity')).reshape(1, -1)
        diagnosis = normalize(diagnosis,norl_dict.get('diagnosis')).reshape(1, -1)

        temprature = normalize(temprature,norl_dict.get('temprature')).reshape(1, -1)
        dewpoint = normalize(dewpoint,norl_dict.get('dewpoint')).reshape(1, -1)
        humidity = normalize(humidity,norl_dict.get('humidity')).reshape(1, -1)
        wdsp=normalize(wdsp,norl_dict.get('wdsp')).reshape(1,-1)
        max_temprature = normalize(max_temprature,norl_dict.get('max_temprature')).reshape(1, -1)
        min_temprature = normalize(min_temprature,norl_dict.get('min_temprature')).reshape(1, -1)
        prcp=normalize(prcp,norl_dict.get('prcp')).reshape(1,-1)
        is_rain = normalize(is_rain,norl_dict.get('is_rain')).reshape(1, -1)
        data_dict = {'index': index,
                     'morbidity': morbidity, 'diagnosis': diagnosis,
                     'temprature': temprature, 'dewpoint': dewpoint, 'humidity': humidity,'wdsp':wdsp,
                     'max_temprature': max_temprature,
                     'min_temprature': min_temprature,
                     'prcp':prcp,
                     'is_rain': is_rain,
                     'accrued_n':accrued_n,
                     }

        compartment_dict.clear()
        compartment_dict = {'S': accrued_s, 'E': accrued_e, 'I': accrued_i, 'Ie': accrued_ie, 'R': accrued_r,
                            'S_N': accrued_s_n}
        # type time
        data = np.concatenate((itemgetter(*input_list)(data_dict)), axis=0)
        # time type
        data = np.swapaxes(data, 0, 1)

        # type time
        compartment = np.concatenate((itemgetter(*compartment_list)(compartment_dict)), axis=0)
        # time shape
        compartment = np.swapaxes(compartment, 0, 1)



        compartment = compartment * args.scale
        label = label * args.scale

        # 采样
        localize(data, compartment)
        for day in range(data.shape[0] - time_step - alpha - 1):
            model_input = np.concatenate((model_input, data[day:day + time_step, :].reshape(1, time_step, -1)),
                                         axis=0)
            model_compartment = np.concatenate(
                (model_compartment, compartment[day + 1:day + 1 + time_step, :].reshape(1, time_step, - 1)), axis=0)
            model_label = np.concatenate(
                (model_label, label[day + 1 + alpha:day + time_step + alpha + 1].reshape(1, time_step)), axis=0)
            model_region=np.concatenate(
                (model_region,region[day:day+time_step,:].reshape(1,time_step,-1)),axis=0)
        # print('标签维度'+str(model_label.shape))
        # print('输入维度'+str(model_input.shape))
        # print('地理维度'+str(model_region.shape))
    return model_input, model_compartment, model_label,model_region

# 取样,取样数据集
def localize(data_dict, compartment_dict):
    frame_data_dict = pd.DataFrame(data_dict)
    frame_compartment = pd.DataFrame(compartment_dict)
    writer = pd.ExcelWriter('test' + '.xlsx')
    frame_data_dict.to_excel(writer, sheet_name='data', index=False, )
    frame_compartment.to_excel(writer, sheet_name='compartment', index=False, )
    writer.save()
    pass

class HFMD_DataSet(Dataset):
    def __init__(self, args):
        super().__init__()

        model_input, model_compartment, model_label,model_region = make_data(args)

        self.data = torch.tensor(model_input, dtype=torch.float)
        self.compartment = torch.tensor(model_compartment, dtype=torch.float)
        self.label = torch.tensor(model_label, dtype=torch.float)
        self.region=torch.tensor(model_region,dtype=torch.float)

    def __getitem__(self, index):
        return self.data[index + 1], self.compartment[index + 1], self.label[index + 1],self.region[index+1]

    def __len__(self):
        return self.data.shape[0] - 1

def build(args):
    dataset = HFMD_DataSet(args)
    return dataset

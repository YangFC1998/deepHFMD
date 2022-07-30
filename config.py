import argparse
import util
from dataloader import pre_process

INPUT_LIST=['index','morbidity','diagnosis','temprature','dewpoint','humidity','wdsp','max_temprature','min_temprature','prcp','is_rain','accrued_n']
NORMAL_DICT = {'index': 0, 'morbidity': 0, 'diagnosis': 0,
               'temprature': 0, 'dewpoint': 0, 'humidity': 0, 'wdsp': 0,
               'max_temprature': 0, 'min_temprature': 0, 'prcp': 0, 'is_rain': 0,
               'accrued_n':1}

DNN_LIST=['index','morbidity','diagnosis','temprature','dewpoint','humidity','wdsp','prcp','is_rain']
INTERACTIVE_LIST=['diagnosis','accrued_n','temprature','humidity','wdsp','prcp','is_rain']

COUNTY=[2,3,11,12,13,14,15,81,83,85]
COUNTY_LIST=[2,3,11,12,13,14,15,81,83,85]


YEAR=[2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
YEAR_LIST=[2012,2013,2016,2017,2019]
EVAL_YEAR=2019
TEST_YEAR=2018


COMPARTMENT={'S':0,'E':1,'I':2,'Ie':3,'R':4,'S_N':5}
DECAY_WINDOW=[0.1,0.2,0.5]
BORN_POPULATION=[73146,66137,75342,70464,62274,72201,63153,81119,70599,95815,62902,118418,115683,90099,88848,66503]

#scheduler
MULTI_STEP=[40,80]
#FM
FM_SIZE=None
def get_args_parser():
    parser = argparse.ArgumentParser('DeepHFMD', add_help=False)
    parser.add_argument('--sample',default=False,type=lambda x:x.lower()=='true')
    # 数据集分类
    parser.add_argument('--dataset_type', default='simple_single_dimension', type=str)
    # model train
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--time_step', default=21, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    # 原始数据
    parser.add_argument('--case_path',default='dataset/',type=str)
    parser.add_argument('--noaa_path',default='dataset/noaa/',type=str)
    parser.add_argument('--utildata_path',default='dataset/util.xlsx',type=str)
    parser.add_argument('--regenerate_dataset', default=False , type=lambda x: x.lower() == 'true', help='重新生成数据集')
    parser.add_argument('--max_age', default=5, type=int, help='数据集年龄限制')
    # 数据集处理noaa
    parser.add_argument('--noaa_interval',default=7,type=int,help='近n日天气情况')

    #case
    parser.add_argument('--recover_days',default=7,type=int)
    parser.add_argument('--incubation_days',default=4,type=int)
    parser.add_argument('--p',default=0.09186,type=float)
    parser.add_argument('--k',default=0.5,type=float)

    parser.add_argument('--scale',default=1,type=float)

    parser.add_argument('--preprocess',default=True,type=lambda x:x.lower=='true')
    parser.add_argument('--compartment_method',default=pre_process.SEIIeR_timeDalay_Born_Decay.__name__,type=str)

    # 加权移动平均平滑处理的参数
    parser.add_argument('--smooth_case', default=True, type=lambda x: x.lower() == 'true', help='平滑每日新增病例')
    parser.add_argument('--smooth_method', default=util.smooth_moving_average.__name__, type=str, help='平滑方法')
    parser.add_argument('--moving_average_window_size', default=15, type=int)
    parser.add_argument('--ewma_alpha', default=0.1, type=float, help='指数加权移动平均')
    parser.add_argument('--savgol_filter_window_length', default=15, type=int)
    parser.add_argument('--savgol_filter_polyorder', default=3, type=int)

    #embedding
    parser.add_argument('--pass_linear',default=True,type=lambda x:x.lower()=='true')
    parser.add_argument('--embedding_hidden_size',default=64,type=int)
    parser.add_argument('--embedding_dropout_prob',default=0.3,type=float)

    parser.add_argument('--FM_size',default=2,type=int)
    parser.add_argument('--add_FM',default=True,type=lambda x:x.lower()=='true')
    parser.add_argument('--FM_latent_size',default=4,type=int)
    parser.add_argument('--FM_activation',default='tanh',type=str)




    parser.add_argument('--label_name', default='diagnosis', type=str)
    #multistep exponent cosine reduceonloss cosine_warmup
    parser.add_argument('--scheduler',default='reduceonloss',type=str)
    #visual
    parser.add_argument('--visual',default=True,type=lambda x:x.lower()=='true')
    return parser

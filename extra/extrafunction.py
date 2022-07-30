import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def how_many_people_infected_everyyear(case_dir):
    year=[]
    for i in range(10):
        frame=pd.read_excel(case_dir+str(2010+i)+'.xlsx',sheet_name='diagnosis')['total'].values
        year.append(sum(frame))
    print(year)
    print('avrage infections:'+str(sum(year)/10))
def can_it_work(input_path):
    for i in range(10):
        frame_noborn=pd.read_excel(input_path+'SEIIeR_timeDelay_noBorn/'+str(2010+i)+'SEIIeR_timeDelay_noBorn.xlsx',sheet_name='accrued')['S_N'].values
        frame_born=pd.read_excel(input_path+'SEIIeR_timeDelay_Born/'+str(2010+i)+'SEIIeR_timeDelay_Born.xlsx',sheet_name='accrued')['S_N'].values
        plt.plot(frame_noborn,color='r',label='不使用人口增加')
        plt.plot(frame_born,color='g',label='使用自然出生人口')
        plt.legend()
        plt.show()
def Ie_Count(input_path):
    for i in range(10):
        frame_noborn=pd.read_excel(input_path+'SEIIeR_timeDelay_noBorn/'+str(2010+i)+'SEIIeR_timeDelay_noBorn.xlsx',sheet_name='accrued')
        frame_born=pd.read_excel(input_path+'SEIIeR_timeDelay_Born/'+str(2010+i)+'SEIIeR_timeDelay_Born.xlsx',sheet_name='accrued')
        plt.plot(frame_noborn['Ie']/frame_noborn['N'],color='r',label='不使用人口增加')
        plt.plot(frame_born['Ie']/frame_born['N'],color='g',label='使用自然出生人口')
        plt.legend()
        plt.show()
def get_yearly_case(input_path):
    for i in range(11):
        frame=pd.read_excel(input_path+str(2010+i)+'.xlsx')
        y0=len(frame[(frame['age'])==0])
        y1=len(frame[(frame['age'])==1])
        y2=len(frame[(frame['age'])==2])
        y3=len(frame[(frame['age'])==3])
        y4=len(frame[(frame['age'])==4])
        y5=len(frame[(frame['age'])==5])
        print(str(2010+i))
        print(y0)
        print(y1)
        print(y2)
        print(y3)
        print(y4)
        print(y5)
        print(frame['age'].size)
def get_init():
    p=0.09
    born_population = [73146, 66137, 75342, 70464, 62274, 72201, 63153, 81119, 70599, 95815, 62902, 118418, 115683,90099, 88848, 66503]
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    frame_age=pd.read_excel('C:\\Users\\29561\Desktop\\util.xlsx',sheet_name='age')
    for year in years:
        index=year-2010
        increase_population = born_population[index + 5]
        base_population = sum(born_population[index:index + 5])
        y_0=sum(frame_age.iloc[index,[2,3,4,5,6]])
        y_1=sum(frame_age.iloc[index+1,[2,3,4,5]])
        y_2=sum(frame_age.iloc[index+2,[2,3,4]])
        y_3=sum(frame_age.iloc[index+3,[2,3]])
        y_4=sum(frame_age.iloc[index+4,[2]])
        y_0=y_0+y_0/p*(1-p)
        y_1=y_1+y_1/p*(1-p)
        y_2=y_2+y_2/p*(1-p)
        y_3=y_3+y_3/p*(1-p)
        y_4=y_4+y_4/p*(1-p)
        R=int(y_0+y_1+y_2+y_3+y_4)
        print(str(R)+'基本人口'+str(base_population)+'占比'+str(R/base_population*100)+'%')


def count_p():
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    base=[]
    p=[]
    for year in years:
        born_population = [73146, 66137, 75342, 70464, 62274, 72201, 63153, 81119, 70599, 95815, 62902, 118418, 115683,
                           90099, 88848, 66503]
        index = year - 2010
        base_population = sum(born_population[index:index + 6])
        base.append(base_population)

        frame=pd.read_excel('../dataset/raw/dailyIncrease/'+str(year)+'.xlsx',sheet_name='diagnosis')['total']
        p.append(sum(frame) / (sum(frame) + base_population * .2))
        # print('总患病群体'+str(sum(frame)))
        # print('总患病人群'+str(base_population))
        # print(sum(frame)/(sum(frame)+base_population*.2))

    print(sum(p)/len(p))





def main():
    pass
    # how_many_people_infected_everyyear('../dataset/raw/dailyIncrease/')
    # can_it_work('../dataset/')
    # Ie_Count('../dataset/')
    # get_yearly_case('../dataset/raw/split_case/')
    # get_init()
    count_p()
if __name__=='__main__':
    main()

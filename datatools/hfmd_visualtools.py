import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 统计数据的特征,适用于数据清洗过后的hfmd.xlsx
def analysis(input_path, output_dir):
    frame = pd.read_excel(input_path)
    # frame=frame.loc[:10,:]

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    # 年龄比例
    age_dict = {}
    # 男性数量
    male = [0, 0]
    # 人群占比
    type = [0, 0, 0]
    # 严重程度占比
    severe = [0, 0]
    # 统计地区分布
    region_dic = {'370202': 0, '370203': 0,
                  '370211': 0, '370212': 0, '370213': 0, '370214': 0, '370215': 0,
                  '370281': 0, '370283': 0, '370285': 0, 'other': 0}
    regions = ['370202', '370203',
               '370211', '370212', '370213', '370214', '370215',
               '370281', '370283', '370285']
    # 统计发病到诊断的时间
    incubation_dic = {}

    for index, row in frame.iterrows():
        # 分析年龄分布
        if row['age'] not in age_dict.keys():
            age_dict[row['age']] = 1
        else:
            age_dict[row['age']] = age_dict[row['age']] + 1
        # 统计男女比例
        male[row['gender']] = male[row['gender']] + 1
        # 统计人群比例
        if row['type'] != -1:
            type[row['type']] = type[row['type']] + 1
        else:
            type[2] = type[2] + 1
        # 统计危重患者占比
        severe[row['severe']] = severe[row['severe']] + 1
        # 统计地区占比
        if str(row['region']) == '370202':
            region_dic['370202'] = region_dic['370202'] + 1
        if str(row['region']) == '370203':
            region_dic['370203'] = region_dic['370203'] + 1

        if str(row['region']) == '370211':
            region_dic['370211'] = region_dic['370211'] + 1
        if str(row['region']) == '370212':
            region_dic['370212'] = region_dic['370212'] + 1
        if str(row['region']) == '370213':
            region_dic['370213'] = region_dic['370213'] + 1
        if str(row['region']) == '370214':
            region_dic['370214'] = region_dic['370214'] + 1
        if str(row['region']) == '370215':
            region_dic['370215'] = region_dic['370215'] + 1

        if str(row['region']) == '370281':
            region_dic['370281'] = region_dic['370281'] + 1
        if str(row['region']) == '370283':
            region_dic['370283'] = region_dic['370283'] + 1
        if str(row['region']) == '370285':
            region_dic['370285'] = region_dic['370285'] + 1

        if str(row['region']) not in regions:
            region_dic['other'] = region_dic['other'] + 1

        # 分析发病到确诊时间占比
        if row['incubation'] not in incubation_dic.keys():
            incubation_dic[row['incubation']] = 1
        else:
            incubation_dic[row['incubation']] = incubation_dic[row['incubation']] + 1

    # 将统计的年龄分布信息变为有序的
    sorted_data = sorted(age_dict.items(), key=lambda d: d[0])
    age_dict.clear()
    for item in sorted_data:
        age_dict[item[0]] = item[1]

    # 将统计的年龄分布信息变为有序的
    sorted_data.clear()
    sorted_data = sorted(incubation_dic.items(), key=lambda d: d[0])
    incubation_dic.clear()
    for item in sorted_data:
        incubation_dic[item[0]] = item[1]

    with open(output_dir + 'report.txt', 'a+') as writer:
        writer.write('年龄比例: ' + str(age_dict) + '最大值: ' + str(max(age_dict.items(), key=lambda d: d[1])))
        writer.write('男女比例: ' + str(male))
        writer.write('人群比例: ' + str(type))
        writer.write('重症患者: ' + str(severe))
        writer.write('地区比例: ' + str(region_dic))
        writer.write('发病时间: ' + str(incubation_dic))
    writer.close()

    # 年龄比例
    labels = [str(age) for age in age_dict.keys()]  # 直方图x轴
    height = age_dict.values()  # 直方图数值，可以有多个
    index = np.arange(len(labels))  # 横轴
    width = 0.5  # 直方图宽度
    fig, ax = plt.subplots()
    rects = ax.bar(x=index, height=height, color='#87CEFA', width=width, label='群体数量')
    # 假如需要控制位置x=index - width / 2,index + width / 2
    # 直方图上方打标
    for rect in rects:
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom', fontsize=8)
    ax.set_title('青岛市手足口病患病年龄分布')
    ax.set_xticks(ticks=index)
    ax.set_xticklabels(labels)
    ax.set_xlabel('年龄')
    ax.set_ylabel('群体数量')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(output_dir + '青岛市手足口病患病年龄分布.tif', dpi=300)
    plt.show()

    # 画出青岛男女分布
    fig, ax = plt.subplots()
    rect = ax.pie(x=male, labels=['女性', '男性'], autopct='%1.1f%%')  # 绘制饼图
    ax.set_title('青岛市手足口病患者男女比例')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(output_dir + '青岛市手足口病患者男女比例.tif', dpi=300)
    plt.show()

    # 画出青岛内外分布
    fig, ax = plt.subplots()
    rect = ax.pie(x=list(region_dic.values()), labels=region_dic.keys(), autopct='%1.1f%%')  # 绘制饼图
    ax.set_title('青岛市手足口病患病群体地理位置比例')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(output_dir + '青岛市手足口病患病患者地理位置分布.tif', dpi=300)
    plt.show()

    # 画出患病人群分布
    fig, ax = plt.subplots()
    rect = ax.pie(x=type, labels=['散居儿童', '幼托儿童', '其他'], autopct='%1.1f%%')  # 绘制饼图
    ax.set_title('青岛市手足口病患病人群分布')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(output_dir + '青岛市手足口病患病人群分布.tif', dpi=300)
    plt.show()

    # 画出危重病人群分布
    fig, ax = plt.subplots()
    rect = ax.pie(x=severe, labels=['非危重病人', '危重病人'], autopct='%1.1f%%')  # 绘制饼图
    ax.set_title('青岛市手足口病危重病人分布')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(output_dir + '青岛市手足口病危重病人分布.tif', dpi=300)
    plt.show()

    # 发病-就诊时间分布
    width = 0.5  # 直方图宽度
    fig, ax = plt.subplots()
    rects = ax.bar(x=np.arange(len(incubation_dic.keys())), height=incubation_dic.values(), color='#87CEFA',
                   width=width, label='群体数量')
    for rect in rects:
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom', fontsize=8)
    ax.set_title('青岛市手足口病发病-就诊时间分布')
    ax.set_xticks(ticks=np.arange(len(incubation_dic.keys())))
    ax.set_xticklabels(incubation_dic.keys())
    ax.set_xlabel('时间')
    ax.set_ylabel('群体数量')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig(output_dir + '青岛市手足口病发病-就诊时间分布.tif', dpi=300)
    plt.show()


# 可视化，查看任意日期的手足口病发病情况,折线图形式,dailyincrease.xlsx
def visual_dailyincrease(input_path, output_path, starttime=None, endtime=None):
    frame_morbidity = pd.read_excel(input_path, sheet_name='morbidity')
    frame_diagnosis = pd.read_excel(input_path, sheet_name='diagnosis')
    origintime = datetime.datetime(2010, 1, 1)

    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    if starttime == None and endtime == None:
        starttime = frame_morbidity['time'][0]
        endtime = frame_morbidity['time'][frame_morbidity.shape[0]-1]
        timeduration = frame_morbidity.shape[0]
        case_morbifity = frame_morbidity['total']
        case_diagnosis = frame_diagnosis['total']
    else:
        starttime = datetime.datetime.strptime(starttime, '%Y-%m-%d')
        endtime = datetime.datetime.strptime(endtime, '%Y-%m-%d')
        # 找到起始游标
        startanchor = (starttime - origintime).days
        endanchor = (endtime - origintime).days

        timeduration = len(frame_morbidity.loc[startanchor:endanchor, 'time'])
        case_morbifity = frame_morbidity.loc[startanchor:endanchor, 'total']
        case_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'total']

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.set_title('青岛市手足口病日发病人数')
    axis.set_xlabel('日期')
    axis.set_ylabel('人数')
    axis.plot(np.arange(0, timeduration), case_morbifity, color='#9AC9DB', label='手足口病患病人数-发病')
    axis.plot(np.arange(0, timeduration), case_diagnosis, color='#F8AC8C', label='手足口病患病人数-诊断')

    axis.legend()
    plt.savefig(output_path + str(starttime.date()) + '-' + str(endtime.date()) + '日发病统计-发病诊断.tif')  # 保存图片
    # plt.show()
def visual_dailyincrease_countywise(input_path, output_path, starttime=None, endtime=None):
    frame_morbidity = pd.read_excel(input_path, sheet_name='morbidity')
    frame_diagnosis = pd.read_excel(input_path, sheet_name='diagnosis')
    origintime = datetime.datetime(2010, 1, 1)
    path=os.path.dirname(output_path)
    if os.path.exists(path) == False:
        os.makedirs(path)

    if starttime == None and endtime == None:
        starttime = frame_morbidity['time'][0]
        endtime = frame_morbidity['time'][frame_morbidity.shape[0] - 1]
        timeduration = frame_morbidity.shape[0]
        case_morbifity = frame_morbidity['total']
        case_diagnosis = frame_diagnosis['total']
        region_370202_morbidity = frame_morbidity['region_370202']
        region_370203_morbidity = frame_morbidity['region_370203']
        region_370211_morbidity = frame_morbidity['region_370211']
        region_370212_morbidity = frame_morbidity['region_370212']
        region_370213_morbidity = frame_morbidity['region_370213']
        region_370214_morbidity = frame_morbidity['region_370214']
        region_370215_morbidity = frame_morbidity['region_370215']
        region_370281_morbidity = frame_morbidity['region_370281']
        region_370283_morbidity = frame_morbidity['region_370283']
        region_370285_morbidity = frame_morbidity['region_370283']

        region_370202_diagnosis = frame_diagnosis['region_370202']
        region_370203_diagnosis = frame_diagnosis['region_370203']
        region_370211_diagnosis = frame_diagnosis['region_370211']
        region_370212_diagnosis = frame_diagnosis['region_370212']
        region_370213_diagnosis = frame_diagnosis['region_370213']
        region_370214_diagnosis = frame_diagnosis['region_370214']
        region_370215_diagnosis = frame_diagnosis['region_370215']
        region_370281_diagnosis = frame_diagnosis['region_370281']
        region_370283_diagnosis = frame_diagnosis['region_370283']
        region_370285_diagnosis = frame_diagnosis['region_370285']
    else:
        starttime = datetime.datetime.strptime(starttime, '%Y-%m-%d')
        endtime = datetime.datetime.strptime(endtime, '%Y-%m-%d')
        # 找到起始游标
        startanchor = (starttime - origintime).days
        endanchor = (endtime - origintime).days

        timeduration = len(frame_morbidity.loc[startanchor:endanchor, 'time'])
        case_morbifity = frame_morbidity.loc[startanchor:endanchor, 'total']
        case_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'total']

        region_370202_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370202']
        region_370203_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370203']
        region_370211_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370211']
        region_370212_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370212']
        region_370213_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370213']
        region_370214_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370214']
        region_370215_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370215']
        region_370281_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370281']
        region_370283_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370283']
        region_370285_morbidity = frame_morbidity.loc[startanchor:endanchor, 'region_370285']

        region_370202_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370202']
        region_370203_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370203']
        region_370211_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370211']
        region_370212_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370212']
        region_370213_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370213']
        region_370214_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370214']
        region_370215_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370215']
        region_370281_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370281']
        region_370283_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370283']
        region_370285_diagnosis = frame_diagnosis.loc[startanchor:endanchor, 'region_370285']

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.set_title('青岛市手足口病日发病人数')
    axis.set_xlabel('日期')
    axis.set_ylabel('人数')
    # axis.plot(np.arange(0, timeduration), case_morbifity, color='#F8AC8C', label='手足口病患病人数-发病')
    axis.plot(np.arange(0, timeduration), region_370202_morbidity, label='370202')
    axis.plot(np.arange(0, timeduration), region_370203_morbidity, label='370203')
    axis.plot(np.arange(0, timeduration), region_370211_morbidity, label='370211')
    axis.plot(np.arange(0, timeduration), region_370212_morbidity, label='370212')
    axis.plot(np.arange(0, timeduration), region_370213_morbidity, label='370213')
    axis.plot(np.arange(0, timeduration), region_370214_morbidity, label='370214')
    axis.plot(np.arange(0, timeduration), region_370215_morbidity, label='370215')
    axis.plot(np.arange(0, timeduration), region_370281_morbidity, label='370281')
    axis.plot(np.arange(0, timeduration), region_370283_morbidity, label='370283')
    axis.plot(np.arange(0, timeduration), region_370285_morbidity, label='370285')
    axis.legend()
    plt.savefig(output_path + str(starttime.date()) + '-' + str(endtime.date()) + '日发病统计-县区-发病.tif')  # 保存图片
    # plt.show()

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.set_title('青岛市手足口病日发病人数')
    axis.set_xlabel('日期')
    axis.set_ylabel('人数')
    # axis.plot(np.arange(0, timeduration), case_diagnosis, color='#F8AC8C', label='手足口病患病人数-诊断')
    axis.plot(np.arange(0, timeduration), region_370202_diagnosis, label='370202')
    axis.plot(np.arange(0, timeduration), region_370203_diagnosis, label='370203')
    axis.plot(np.arange(0, timeduration), region_370211_diagnosis, label='370211')
    axis.plot(np.arange(0, timeduration), region_370212_diagnosis, label='370212')
    axis.plot(np.arange(0, timeduration), region_370213_diagnosis, label='370213')
    axis.plot(np.arange(0, timeduration), region_370214_diagnosis, label='370214')
    axis.plot(np.arange(0, timeduration), region_370215_diagnosis, label='370215')
    axis.plot(np.arange(0, timeduration), region_370281_diagnosis, label='370281')
    axis.plot(np.arange(0, timeduration), region_370283_diagnosis, label='370283')
    axis.plot(np.arange(0, timeduration), region_370285_diagnosis, label='370285')
    axis.legend()
    plt.savefig(output_path + str(starttime.date()) + '-' + str(endtime.date()) + '日发病统计-县区-诊断.tif')  # 保存图片
    # plt.show()
def visual_accrued(input_path, output_path, starttime=None, endtime=None):
    frame_accrued = pd.read_excel(input_path, sheet_name='accrued')
    origintime = datetime.datetime(2010, 1, 1)

    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    if starttime == None and endtime == None:
        starttime = frame_accrued['date'][0]
        endtime = frame_accrued['date'][frame_accrued.shape[0] - 1]
        timeduration = frame_accrued.shape[0]
        accrued_i = frame_accrued['accrued_i']
        accrued_r = frame_accrued['accrued_r']
    else:
        starttime = datetime.datetime.strptime(starttime, '%Y-%m-%d')
        endtime = datetime.datetime.strptime(endtime, '%Y-%m-%d')
        # 找到起始游标
        startanchor = (starttime - origintime).days
        endanchor = (endtime - origintime).days

        timeduration = len(frame_accrued.loc[startanchor:endanchor, 'time'])
        accrued_i = frame_accrued.loc[startanchor:endanchor, 'accrued_i']
        accrued_r = frame_accrued.loc[startanchor:endanchor, 'accrued_r']

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.set_title('青岛市手足口病累积发病人数')
    axis.set_xlabel('日期')
    axis.set_ylabel('人数')
    axis.plot(np.arange(0, timeduration), accrued_i, color='#9AC9DB', label='手足口病患病人数-累积治愈')
    # axis.plot(np.arange(0, timeduration), accrued_r, color='#F8AC8C', label='手足口病患病人数-累积康复')

    axis.legend()
    plt.savefig(output_path + str(starttime.date()) + '-' + str(endtime.date()) + '累积发病统计-发病治愈.tif')  # 保存图片
    # plt.show()

# 任意一年的所有变量的变化
class HFMD:
    def __init__(self):
        self.year=0
        self.male=0#性别
        self.severe=0#是否为重症患者
        self.age={}#年龄
        self.type = [0, 0, 0]
        self.count=0#总数
    def process(self,frame):

        for index,row in frame.iterrows():
            # 首先确定年份
            if row['morbidity'].year==self.year:
                # 记录总数
                self.count=self.count+1
                # 记录性别
                if row['gender']==1:
                    self.male=self.male+1
                #记录重症患者
                if row['severe']==1:
                    self.severe=self.severe+1
                # 记录年龄
                if str(row['age']) in self.age:
                    self.age[str(row['age'])]=self.age[str(row['age'])]+1
                else:
                    self.age[str(row['age'])] = 1

                # 统计人群比例
                if row['type'] != -1:
                    self.type[row['type']] = self.type[row['type']] + 1
                else:
                    self.type[2] = self.type[2] + 1

    def draw(self,outputpath):


        fig,ax=plt.subplots(2,2,figsize=(12,12))
        ax[0][0].set_title(str(self.year) + '年青岛市手足口病危重患者占比')  # 绘制标题
        ax[0][0].pie([self.severe,self.count-self.severe], labels=['重症患者', '非重症患者'], autopct='%1.1f%%')  # 绘制饼图


        ax[0][1].pie([self.male, self.count - self.male], labels=['男性', '女性'], autopct='%1.1f%%')  # 绘制饼图
        ax[0][1].set_title(str(self.year) + '年青岛市手足口病男性患者占比')  # 绘制标题


        ax[1][0].pie(self.type, labels=['散居儿童', '幼托儿童','其他'], autopct='%1.1f%%')  # 绘制饼图
        ax[1][0].set_title(str(self.year) + '年青岛市手足口病患者群体占比')  # 绘制标题

        print(self.age)
        ax[1][1].pie(self.age.values(), labels=self.age.keys(), autopct='%1.1f%%')  # 绘制饼图
        ax[1][1].set_title(str(self.year) + '年青岛市手足口病患者年龄占比')  # 绘制标题

        plt.legend()
        plt.savefig(outputpath+str(self.year) + '年青岛市手足口病患者分析.tif')  # 保存图片

#固定量在12年内的变化
def visual_severe(HFMDS,output_path):
    plt.figure(figsize=(12, 12))
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        ax.set_title(str(2010 + i) + '年')
        ax.pie([HFMDS[i].severe, HFMDS[i].count - HFMDS[i].severe], labels=['重症患者', '非重症患者'], autopct='%1.1f%%')  # 绘制饼图
    plt.suptitle('手足口病患病重症比例')
    plt.savefig(output_path+'手足口病患病重症比例.tif')  # 保存图片

def visual_gender(HFMDS, output_path):
    plt.figure(figsize=(12, 12))
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        ax.set_title(str(2010 + i) + '年')
        ax.pie([HFMDS[i].male, HFMDS[i].count - HFMDS[i].male], labels=['男性', '女性'], autopct='%1.1f%%')  # 绘制饼图
    plt.suptitle('手足口病患病性别分布')
    plt.savefig(output_path+'手足口病患病性别分布.tif')  # 保存图片

def visual_group(HFMDS, output_path):
    plt.figure(figsize=(12, 12))
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        ax.set_title(str(2010 + i) + '年')
        ax.pie(HFMDS[i].type, labels=['散居儿童', '幼托儿童', '其他'], autopct='%1.1f%%')  # 绘制饼图
    plt.suptitle('手足口病患病群体分布')
    plt.savefig(output_path+'手足口病患病群体分布.tif')  # 保存图片

def visual_age(HFMDS, output_path):
    plt.figure(figsize=(12, 12))
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        ax.set_title(str(2010 + i) + '年')
        ax.pie(HFMDS[i].age.values(), labels=HFMDS[i].age.keys(), autopct='%1.1f%%')  # 绘制饼图
        with open(output_path + 'age.txt', 'a+') as writer:
            writer.write('年份: ' + (str(2010 + i) + '年\n'))
            for j in range(6):
               writer.write("年龄"+str(j)+" 群体数量"+str(HFMDS[i].age.get(str(j)))+"\n")
        writer.close()
    plt.suptitle('手足口病患病年龄分布')
    plt.savefig(output_path+'手足口病患病年龄分布.tif')  # 保存图片

def analysis_yearwise(input_dir,output_dir):
    if os.path.exists(output_dir+'/splitYear/') == False:
        os.mkdir(output_dir+'/splitYear/')
    HFMDS=[]
    for i in range(12):
        HFMDS.append(HFMD())
        HFMDS[i].year= 2010+i
        frame=pd.read_excel(input_dir+str(2010+i)+'.xlsx')
        HFMDS[i].process(frame)
        HFMDS[i].draw(output_dir+'/splitYear/')
    if os.path.exists(output_dir+'/splitYear/') == False:
        os.mkdir(output_dir+'/splitYear/')
    visual_severe(HFMDS, output_dir+'/splitYear/')
    visual_group(HFMDS, output_dir+'/splitYear/')
    visual_gender(HFMDS, output_dir+'/splitYear/')
    visual_age(HFMDS, output_dir+'/splitYear/')


def visual_Sim(input_path,output_path):
    if os.path.exists(output_path)==False:
        os.mkdir(output_path)
    for i in range(12):
        frame=pd.read_excel(input_path+str(2010+i)+'.xlsx',sheet_name='accrued')
        S=frame['S']
        E=frame['E']
        I=frame['I']
        Ie=frame['Ie']
        R=frame['R']
        plt.figure()
        plt.grid()
        plt.title(str(2010+i)+'累积人群')
        plt.plot(S, color='b', label='Susceptible')
        plt.plot(E, color='#58508d', label='Exposed')
        plt.plot(I, color='r', label='Infected')
        plt.plot(Ie, color='#ffa600', label='Latent Infected')
        plt.plot(R, color='#488f31', label='Recovered with immunity')
        plt.xlabel('Time t, [days]')
        plt.ylabel('Numbers of individuals')
        plt.legend()
        plt.savefig(output_path+str(2010+i)+' 累积人群.png')
        plt.show()
def visual_SimEIIe(input_path,output_path):
    if os.path.exists(output_path)==False:
        os.mkdir(output_path)
    for i in range(12):
        frame=pd.read_excel(input_path+str(2010+i)+'.xlsx',sheet_name='accrued')
        S=frame['S']
        E=frame['E']
        I=frame['I']
        Ie=frame['Ie']
        R=frame['R']
        plt.figure()
        plt.grid()
        plt.title(str(2010+i)+'累积人群')
        # plt.plot(S, color='b', label='Susceptible')
        # plt.plot(E, color='#58508d', label='Exposed')
        plt.plot(I, color='r', label='Infected')
        # plt.plot(Ie, color='#ffa600', label='Latent Infected')
        # plt.plot(R, color='#488f31', label='Recovered with immunity')
        plt.xlabel('Time t, [days]')
        plt.ylabel('Numbers of individuals')
        plt.legend()
        plt.savefig(output_path+str(2010+i)+' 累积人群.png')
        plt.show()
def visual_totalYears(input_path):
    frame=pd.read_excel(input_path)
    case_count=frame['total'][:365*10]
    plt.figure()
    plt.grid()
    plt.title('2010-2019年青岛市手足口病日发病情况')
    plt.plot(case_count, color='#FA7F6F', label='Infected')
    plt.xlabel('Time (days)')
    plt.ylabel('Numbers of individuals')
    plt.legend()
    plt.xticks([])
    plt.show()

def main():
    pass
    # 数据集总体特征
    analysis('../dataset/raw/hfmd.xlsx', '../image/')
    #按照年份分析单日增长情况
    for i in range(12):
        visual_dailyincrease('../dataset/raw/dailyIncrease/'+str(2010+i)+'.xlsx','../image/daily/')
        visual_dailyincrease_countywise('../dataset/raw/dailyIncrease/'+str(2010+i)+'.xlsx', '../image/daily/coutywise/')
    analysis_yearwise('../dataset/raw/split_case/','../image')
    visual_totalYears('../dataset/raw/dailyIncrease.xlsx')

    # for i in range(12):
    #     visual_accrued('../dataset/dailyIncrease/'+str(2010+i)+'.xlsx','../image/accrued/')
    # visual_Sim('../dataset/tau5 gamma8 p0.091/','../image/Sim/')
    # visual_SimEIIe('../dataset/tau5 gamma8 p0.091/','../image/SimEIIe/')

if __name__ == '__main__':
    main()

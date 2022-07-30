import datetime
import os
import re

import numpy as np
import pandas as pd


# 将两个原始文件合并为一个，但是合并之后需要手动去除三个错误病例
def merge_excel(input_path1, input_path2, output_path):
    frame1 = pd.read_excel(input_path1)
    frame2 = pd.read_excel(input_path2)

    frame = pd.concat([frame1, frame2])
    file_path = os.path.dirname(output_path)
    if os.path.exists(file_path) == False:
        os.mkdir(file_path)
    frame.to_excel(output_path, index=False)

# 由于青岛市的区域划分经历过多次变化,因此需要将往年的国标转换为当今最新的国标编号
def generate_regionmarker(regionmarker):
    if regionmarker == '370282山东省青岛市即墨市':
        return '370215'
    if regionmarker == '370284山东省青岛市胶南市':
        return '370211'
    if regionmarker == '370203山东省青岛市四方区':
        return '370203'
    if regionmarker == '370211山东省青岛市经济技':
        return '370211'
    if regionmarker == '370205山东省青岛市四方区':
        return '370203'
    if regionmarker == '370211山东省青岛市胶南市':
        return '370211'
    if regionmarker == '370211山东省青岛市青岛市':
        return '370211'
    else:
        return regionmarker[:6]

# 格式化case.xlsx,不对数据做任何的删除
def format_file(input_path, output_path):
    frame = pd.read_excel(input_path)
    # frame=frame.loc[:3,:]
    # region 最新的国标 incubation 新增数据,从发病到诊断使用的时间
    region = []
    incubation = []

    for index, row in frame.iterrows():
        # 性别 男1 女0
        frame.loc[index, '性别'] = 1 if row['性别'] == '男' else 0

        # 年龄小于1岁的替换为0岁并把岁去除
        if row['年龄'][-1] == '月' or row['年龄'][-1] == '天':
            frame.loc[index, '年龄'] = '0岁'
        frame.loc[index, '年龄'] = int(re.sub('岁', '', frame.loc[index, '年龄']))

        # 重症患者0 非重症患者1
        frame.loc[index, '重症患者'] = 1 if row['重症患者'] == '是' else 0
        # 散居儿童0 幼托儿童1 其他-1
        if row['人群分类'] == "散居儿童" or row['人群分类'] == "幼托儿童":
            if row['人群分类'] == "散居儿童":
                frame.loc[index, '人群分类'] = 0  # 散居儿童
            if row['人群分类'] == "幼托儿童":
                frame.loc[index, '人群分类'] = 1  # 幼托儿童
        else:
            frame.loc[index, '人群分类'] = -1  # 其他

        # 居住地址国标
        region_code = str(row['现住地址国标'])[:6]
        region_name = row['现住详细地址'][:9]
        region.append(generate_regionmarker(region_code + region_name))

        # 发病至诊断的间隙
        incubation.append((row['诊断时间'] - row['发病日期']).days)

    # 去除无用列
    frame.drop('出生日期', axis=1, inplace=True)
    frame.drop('现住地址国标', axis=1, inplace=True)
    frame.drop('现住详细地址', axis=1, inplace=True)

    frame['incubation'] = incubation
    frame['region'] = region

    # print(frame.columns)
    ## Index(['性别', '年龄', '人群分类', '发病日期', '诊断时间', '重症患者', 'incubation', 'region'], dtype='object')
    frame.columns = ['gender', 'age', 'type', 'morbidity', 'diagnosis', 'severe', 'incubation', 'region']

    # 转存
    frame = pd.DataFrame(frame,
                         columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])

    file_path = os.path.dirname(output_path)
    if os.path.exists(file_path) == False:
        os.mkdir(file_path)
    frame.to_excel(output_path, index=False)

def clean(input_path, output_path, age_limit):
    # 找出不符合限制的病例
    del_list = []

    # 青岛省内国标
    regions = ['370202', '370203',
               '370211', '370212', '370213', '370214', '370215',
               '370281', '370283', '370285']

    frame = pd.read_excel(input_path)
    print('total rows are ' + str(frame.shape[0]))
    for index, row in frame.iterrows():
        if int(row['age']) > age_limit:
            del_list.append(index)
        if str(row['region']) not in regions:
            del_list.append(index)

    del_list = (list(sorted(set(del_list))))
    frame = frame.drop(labels=del_list, axis=0)
    frame = frame.reset_index(drop=True)

    file_path = os.path.dirname(output_path)
    if os.path.exists(file_path) == False:
        os.mkdir(file_path)
    frame.to_excel(output_path, index=False)

# 生成每日新增病例数据
def generate_dailyincrease(frame, time_type):
    starttime = datetime.datetime(2010, 1, 1, 00, 00, 00)
    endTime = datetime.datetime(2021, 10, 31, 23, 59, 59)
    duration = (endTime - starttime).days + 1

    total = np.zeros(duration)
    time = []

    region_370202 = np.zeros(duration)
    region_370203 = np.zeros(duration)
    region_370211 = np.zeros(duration)
    region_370212 = np.zeros(duration)
    region_370213 = np.zeros(duration)
    region_370214 = np.zeros(duration)
    region_370215 = np.zeros(duration)
    region_370281 = np.zeros(duration)
    region_370283 = np.zeros(duration)
    region_370285 = np.zeros(duration)

    for i in range(duration):
        time.append(starttime + datetime.timedelta(days=i))

    for index, row in frame.iterrows():
        region_marker = str(row['region'])
        row_duration = (row[time_type] - starttime).days
        if row_duration < 0 or row_duration >= duration:
            print('illegal duration')
        else:
            # 为了以后好看懂，写的蠢一点吧
            if region_marker == '370202':
                region_370202[row_duration] = region_370202[row_duration] + 1

            if region_marker == '370203':
                region_370203[row_duration] = region_370203[row_duration] + 1

            if region_marker == '370211':
                region_370211[row_duration] = region_370211[row_duration] + 1

            if region_marker == '370212':
                region_370212[row_duration] = region_370212[row_duration] + 1

            if region_marker == '370213':
                region_370213[row_duration] = region_370213[row_duration] + 1

            if region_marker == '370214':
                region_370214[row_duration] = region_370214[row_duration] + 1

            if region_marker == '370215':
                region_370215[row_duration] = region_370215[row_duration] + 1

            if region_marker == '370281':
                region_370281[row_duration] = region_370281[row_duration] + 1

            if region_marker == '370283':
                region_370283[row_duration] = region_370283[row_duration] + 1

            if region_marker == '370285':
                region_370285[row_duration] = region_370285[row_duration] + 1

    # 将所有的加在一起
    for i in range(len(total)):
        total[i] = region_370202[i] + region_370203[i] + region_370211[i] + region_370212[i] + region_370213[i] + \
                   region_370214[i] + region_370215[i] + region_370281[i] + region_370283[i] + region_370285[i]

    # save
    frame = pd.DataFrame(zip(time, total,
                             region_370202, region_370203, region_370211, region_370212, region_370213,
                             region_370214, region_370215, region_370281, region_370283, region_370285),
                         columns=['time', 'total', "region_370202", "region_370203", "region_370211",
                                  "region_370212",
                                  "region_370213",
                                  "region_370214", "region_370215", "region_370281", "region_370283",
                                  "region_370285"])
    return frame

def generate_dailyincreasedata(input_path, output_path):
    frame = pd.read_excel(input_path)
    frame_morbidity = generate_dailyincrease(frame, 'morbidity')
    frame_diagnosis = generate_dailyincrease(frame, 'diagnosis')

    file_path = os.path.dirname(output_path)
    if os.path.exists(file_path) == False:
        os.mkdir(file_path)
    writer = pd.ExcelWriter(output_path)
    frame_morbidity.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()


# 分离病例数据
def split_case(input_path, output_dir):
    frame = pd.read_excel(input_path)

    frame_2010 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2011 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2012 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2013 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2014 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2015 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2016 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2017 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2018 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2019 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2020 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])
    frame_2021 = pd.DataFrame(
        columns=['gender', 'age', 'type', 'severe', 'region', 'morbidity', 'diagnosis', 'incubation'])

    for index, row in frame.iterrows():
        if str(row['diagnosis'].year) == '2010':
            frame_2010.loc[frame_2010.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2011':
            frame_2011.loc[frame_2011.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2012':
            frame_2012.loc[frame_2012.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2013':
            frame_2013.loc[frame_2013.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2014':
            frame_2014.loc[frame_2014.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2015':
            frame_2015.loc[frame_2015.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2016':
            frame_2016.loc[frame_2016.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2017':
            frame_2017.loc[frame_2017.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2018':
            frame_2018.loc[frame_2018.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2019':
            frame_2019.loc[frame_2019.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2020':
            frame_2020.loc[frame_2020.shape[0]] = frame.loc[index]

        if str(row['diagnosis'].year) == '2021':
            frame_2021.loc[frame_2021.shape[0]] = frame.loc[index]
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    frame_2010.to_excel(output_dir + '2010.xlsx', index=False)
    frame_2011.to_excel(output_dir + '2011.xlsx', index=False)
    frame_2012.to_excel(output_dir + '2012.xlsx', index=False)
    frame_2013.to_excel(output_dir + '2013.xlsx', index=False)
    frame_2014.to_excel(output_dir + '2014.xlsx', index=False)
    frame_2015.to_excel(output_dir + '2015.xlsx', index=False)
    frame_2016.to_excel(output_dir + '2016.xlsx', index=False)
    frame_2017.to_excel(output_dir + '2017.xlsx', index=False)
    frame_2018.to_excel(output_dir + '2018.xlsx', index=False)
    frame_2019.to_excel(output_dir + '2019.xlsx', index=False)
    frame_2020.to_excel(output_dir + '2020.xlsx', index=False)
    frame_2021.to_excel(output_dir + '2021.xlsx', index=False)

# 分离daily每一年的frame分为新增患病与确诊
def split_dailysheet(input_path, sheet):
    frame = pd.read_excel(input_path, sheet_name=sheet)

    frame_2010 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211', 'region_370212', 'region_370213',
                 'region_370214', 'region_370215', 'region_370281', 'region_370283', 'region_370285'])
    frame_2011 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2012 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2013 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2014 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2015 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2016 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2017 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2018 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2019 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2020 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])
    frame_2021 = pd.DataFrame(
        columns=['time', 'total', 'region_370202', 'region_370203', 'region_370211',
                 'region_370212', 'region_370213', 'region_370214', 'region_370215',
                 'region_370281', 'region_370283', 'region_370285'])

    starttime = datetime.datetime(2010, 1, 1, 00, 00, 00)
    frame_2010 = frame.iloc[(datetime.datetime(2010, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2010, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2011 = frame.iloc[(datetime.datetime(2011, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2011, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2012 = frame.iloc[(datetime.datetime(2012, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2012, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2013 = frame.iloc[(datetime.datetime(2013, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2013, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2014 = frame.iloc[(datetime.datetime(2014, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2014, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2015 = frame.iloc[(datetime.datetime(2015, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2015, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2016 = frame.iloc[(datetime.datetime(2016, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2016, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2017 = frame.iloc[(datetime.datetime(2017, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2017, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2018 = frame.iloc[(datetime.datetime(2018, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2018, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2019 = frame.iloc[(datetime.datetime(2019, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2019, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2020 = frame.iloc[(datetime.datetime(2020, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2020, 12,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]
    frame_2021 = frame.iloc[(datetime.datetime(2021, 1, 1, 00, 00, 00) - starttime).days:(datetime.datetime(2021, 10,
                                                                                                            31, 23, 59,
                                                                                                            59) - starttime).days + 1]

    return frame_2010, frame_2011, frame_2012, frame_2013, frame_2014, frame_2015, frame_2016, frame_2017, frame_2018, frame_2019, frame_2020, frame_2021

# 分离daily每一年的frame分为新增患病与确诊,保存在文件中
def split_daily(input_path, output_dir):
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    frame_morbidity = split_dailysheet(input_path, 'morbidity')
    frame_morbidity_2010 = frame_morbidity[0]
    frame_morbidity_2011 = frame_morbidity[1]
    frame_morbidity_2012 = frame_morbidity[2]
    frame_morbidity_2013 = frame_morbidity[3]
    frame_morbidity_2014 = frame_morbidity[4]
    frame_morbidity_2015 = frame_morbidity[5]
    frame_morbidity_2016 = frame_morbidity[6]
    frame_morbidity_2017 = frame_morbidity[7]
    frame_morbidity_2018 = frame_morbidity[8]
    frame_morbidity_2019 = frame_morbidity[9]
    frame_morbidity_2020 = frame_morbidity[10]
    frame_morbidity_2021 = frame_morbidity[11]

    frame_diagnosis = split_dailysheet(input_path, 'diagnosis')
    frame_diagnosis_2010 = frame_diagnosis[0]
    frame_diagnosis_2011 = frame_diagnosis[1]
    frame_diagnosis_2012 = frame_diagnosis[2]
    frame_diagnosis_2013 = frame_diagnosis[3]
    frame_diagnosis_2014 = frame_diagnosis[4]
    frame_diagnosis_2015 = frame_diagnosis[5]
    frame_diagnosis_2016 = frame_diagnosis[6]
    frame_diagnosis_2017 = frame_diagnosis[7]
    frame_diagnosis_2018 = frame_diagnosis[8]
    frame_diagnosis_2019 = frame_diagnosis[9]
    frame_diagnosis_2020 = frame_diagnosis[10]
    frame_diagnosis_2021 = frame_diagnosis[11]

    writer = pd.ExcelWriter(output_dir + '2010.xlsx')
    frame_morbidity_2010.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2010.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2011.xlsx')
    frame_morbidity_2011.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2011.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2012.xlsx')
    frame_morbidity_2012.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2012.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2013.xlsx')
    frame_morbidity_2013.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2013.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2014.xlsx')
    frame_morbidity_2014.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2014.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2015.xlsx')
    frame_morbidity_2015.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2015.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2016.xlsx')
    frame_morbidity_2016.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2016.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2017.xlsx')
    frame_morbidity_2017.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2017.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2018.xlsx')
    frame_morbidity_2018.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2018.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2019.xlsx')
    frame_morbidity_2019.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2019.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2020.xlsx')
    frame_morbidity_2020.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2020.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()

    writer = pd.ExcelWriter(output_dir + '2021.xlsx')
    frame_morbidity_2021.to_excel(writer, sheet_name='morbidity', index=False)
    frame_diagnosis_2021.to_excel(writer, sheet_name='diagnosis', index=False)
    writer.save()


# 将每年的单日新增放入一个表格中,但是每一年占一列
def year_wise_collection(output_dir):
    frame_morbidity = pd.DataFrame()
    for i in range(12):
        temp = pd.read_excel(output_dir + str(2010 + i) + '.xlsx', sheet_name='morbidity')
        frame_morbidity[str(2010 + i)] = temp['total']

    frame_diagnosis = pd.DataFrame()
    for i in range(12):
        temp = pd.read_excel(output_dir + str(2010 + i) + '.xlsx', sheet_name='diagnosis')
        frame_diagnosis[str(2010 + i)] = temp['total']

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    writer = pd.ExcelWriter(output_dir + 'year_wise_collection.xlsx')
    frame_diagnosis.to_excel(excel_writer=writer, sheet_name='morbidity', index=False)
    frame_morbidity.to_excel(excel_writer=writer, sheet_name='diagnosis', index=False)
    writer.close()

def generate_data(args):
    rawdata_path1 = 'data/hand-foot-mouth-desease/2010-2015.xlsx'
    rawdata_path2 = 'data/hand-foot-mouth-desease/2016-2021.10.30.xlsx'
    casedata_path = 'data/hand-foot-mouth-desease/case.xlsx'

    dataset_path = 'dataset/formatdata.xlsx'
    if os.path.exists(casedata_path) == False:
        merge_excel(rawdata_path1, rawdata_path2, casedata_path)
    if os.path.exists(dataset_path) == False:
        format_file(casedata_path, dataset_path)

    if args.regenerate_dataset == True:
        clean(dataset_path, os.path.dirname(dataset_path) + '/raw/hfmd.xlsx', args.max_age)
        generate_dailyincreasedata(os.path.dirname(dataset_path) + '/raw/hfmd.xlsx',
                                   os.path.dirname(dataset_path) + '/raw/dailyIncrease.xlsx')
        split_case(os.path.dirname(dataset_path) + '/raw/hfmd.xlsx', os.path.dirname(dataset_path) + '/raw/split_case/')
        split_daily(os.path.dirname(dataset_path) + '/raw/dailyIncrease.xlsx',
                    os.path.dirname(dataset_path) + '/raw/dailyIncrease/')


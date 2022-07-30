import calendar
import os
import pandas as pd
import config
import util
import numpy as np


def smooth(morbidty, diagnosis, args):
    if args.smooth_case == False:
        return morbidty, diagnosis
    method = args.smooth_method
    if method == util.smooth_savgol_filter.__name__:
        morbidty = util.smooth_savgol_filter(morbidty, window_length=args.savgol_filter_window_length,
                                             polyorder=args.savgol_filter_polyorder)
        diagnosis = util.smooth_savgol_filter(diagnosis, window_length=args.savgol_filter_window_length,
                                              polyorder=args.savgol_filter_polyorder)

    if method == util.smooth_moving_average.__name__:
        morbidty = util.smooth_moving_average(morbidty, window=args.moving_average_window_size)
        diagnosis = util.smooth_moving_average(diagnosis, window=args.moving_average_window_size)

    if method == util.smooth_weighted_moving_average.__name__:
        morbidty = util.smooth_weighted_moving_average(morbidty, window=args.moving_average_window_size)
        diagnosis = util.smooth_weighted_moving_average(diagnosis, window=args.moving_average_window_size)

    if method == util.smooth_e_weighted_moving_average.__name__:
        morbidty = util.smooth_e_weighted_moving_average(morbidty, alpha=args.ewma_alpha)
        diagnosis = util.smooth_e_weighted_moving_average(diagnosis, alpha=args.ewma_alpha)

    return np.array(morbidty).reshape(-1), np.array(diagnosis).reshape(-1)


def SEIIeR_timeDelay_noBorn(args):
    input_path=args.case_path+'raw/dailyIncrease/'
    output_path=args.case_path+'SEIIeR_timeDelay_noBorn/'
    if os.path.exists(output_path)==False:
        os.makedirs(output_path)

    decay_window=config.DECAY_WINDOW
    decay_window_size=len(decay_window)

    alpha=args.incubation_days
    gamma=args.recover_days

    p=args.p
    years=config.YEAR
    born_population=config.BORN_POPULATION

    for year in years:
        index=year-2010
        population=sum(born_population[index:index+6])
        frame_morbidity=pd.read_excel(input_path+str(year)+'.xlsx',sheet_name='morbidity')
        frame_diagnosis=pd.read_excel(input_path+str(year)+'.xlsx',sheet_name='diagnosis')


        morbidity=frame_morbidity['total'].values
        diagnosis=frame_diagnosis['total'].values

        morbidity,diagnosis=smooth(morbidity,diagnosis,args)
        year_dict = {'morbidity': morbidity, 'diagnosis': diagnosis}

        if year!=2010:
            pre_morbidity=pd.read_excel(input_path+str(year-1)+'.xlsx',sheet_name='morbidity')['total'].values
            pre_diagnosis=pd.read_excel(input_path+str(year-1)+'.xlsx',sheet_name='diagnosis')['total'].values

            pre_morbidity,pre_diagnosis=smooth(pre_morbidity,pre_diagnosis,args)

        else:
            pre_morbidity,pre_diagnosis=np.zeros(gamma+decay_window_size),np.zeros(gamma+decay_window_size)
        pre_morbidity=pre_morbidity[-(gamma+decay_window_size):]
        pre_diagnosis=pre_diagnosis[-(gamma+decay_window_size):]
        prev_dict = {'morbidity': pre_morbidity, 'diagnosis': pre_diagnosis}

        pre_infected=prev_dict.get(args.label_name)
        infected = year_dict.get(args.label_name)

        length = len(infected)  # 当前年份的时间长度

        # 合并窗口
        infected = np.concatenate((pre_infected, infected), axis=0)
        # 每日新增的显性感染者和隐性感染者具有一定的比率|利用广播
        infected_ie = infected / p * (1 - p)

        accrued_s = np.zeros(length)
        accrued_e = np.zeros(length)
        accrued_i = np.zeros(length)
        accrued_ie = np.zeros(length)
        accrued_r = np.zeros(length)
        accrued_sn = np.zeros(length)
        accrued_n = np.zeros(length)

        for j in range(length):
            # I
            temp = infected[j:j + decay_window_size] * decay_window
            accrued_i[j] = sum(infected[j + decay_window_size:j + gamma + decay_window_size + 1]) + sum(temp)
            # Ie
            temp = infected_ie[j:j + decay_window_size] * decay_window
            accrued_ie[j] = sum(infected_ie[j + decay_window_size:j + gamma + decay_window_size + 1]) + sum(temp)
            # E
            accrued_e[j] = infected[j + decay_window_size + gamma - alpha] \
                           + infected_ie[j + decay_window_size + gamma - alpha]

            # R
            if j - gamma > 0:
                accrued_r[j] = sum(infected[decay_window_size + gamma:j + decay_window_size]) + \
                               sum(infected_ie[decay_window_size + gamma:j + decay_window_size])
            else:
                accrued_r[j] = 0
            # S
            accrued_s[j] = population - accrued_e[j] - accrued_i[j] - accrued_ie[j] - accrued_r[j]
            # SN
            accrued_sn[j] = accrued_s[j] / population
            accrued_n[j] = population

        date = frame_morbidity['time']
        frame_accrued = pd.DataFrame(
            zip(date, accrued_s, accrued_e, accrued_i, accrued_ie, accrued_r, accrued_sn, accrued_n,infected[decay_window_size+gamma:],morbidity,diagnosis),
            columns=['date', 'S', 'E', 'I', 'Ie', 'R', 'S_N', 'N','label','morbidity','diagnosis'])
        writer = pd.ExcelWriter(output_path + str(year) + 'SEIIeR_timeDelay_noBorn.xlsx')
        frame_morbidity.to_excel(writer, sheet_name='morbidity', index=False)
        frame_diagnosis.to_excel(writer, sheet_name='diagnosis', index=False)
        frame_accrued.to_excel(writer, sheet_name='accrued', index=False)
        writer.save()

def SEIIeR_timeDelay_Born(args):
    input_path=args.case_path+'raw/dailyIncrease/'
    output_path=args.case_path+'SEIIeR_timeDelay_Born/'
    if os.path.exists(output_path)==False:
        os.makedirs(output_path)

    decay_window=config.DECAY_WINDOW
    decay_window_size=len(decay_window)

    alpha=args.incubation_days
    gamma=args.recover_days

    p=args.p
    years=config.YEAR
    born_population=config.BORN_POPULATION

    for year in years:
        index=year-2010
        days=366 if calendar.isleap(year) else 365
        increase_population = born_population[index + 5] / days
        base_population = sum(born_population[index:index + 5])

        frame_morbidity=pd.read_excel(input_path+str(year)+'.xlsx',sheet_name='morbidity')
        frame_diagnosis=pd.read_excel(input_path+str(year)+'.xlsx',sheet_name='diagnosis')


        morbidity=frame_morbidity['total'].values
        diagnosis=frame_diagnosis['total'].values

        morbidity,diagnosis=smooth(morbidity,diagnosis,args)
        year_dict = {'morbidity': morbidity, 'diagnosis': diagnosis}

        if year!=2010:
            pre_morbidity=pd.read_excel(input_path+str(year-1)+'.xlsx',sheet_name='morbidity')['total'].values
            pre_diagnosis=pd.read_excel(input_path+str(year-1)+'.xlsx',sheet_name='diagnosis')['total'].values

            pre_morbidity,pre_diagnosis=smooth(pre_morbidity,pre_diagnosis,args)

        else:
            pre_morbidity=np.zeros(gamma+decay_window_size)
            pre_diagnosis=np.zeros(gamma+decay_window_size)

        pre_morbidity=pre_morbidity[-(gamma+decay_window_size):]
        pre_diagnosis=pre_diagnosis[-(gamma+decay_window_size):]
        prev_dict = {'morbidity': pre_morbidity, 'diagnosis': pre_diagnosis}

        pre_infected=prev_dict.get(args.label_name)
        infected = year_dict.get(args.label_name)

        length = len(infected)  # 当前年份的时间长度

        # 合并窗口
        infected = np.concatenate((pre_infected, infected), axis=0)
        # 每日新增的显性感染者和隐性感染者具有一定的比率|利用广播
        infected_ie = infected / p * (1 - p)

        accrued_s = np.zeros(length)
        accrued_e = np.zeros(length)
        accrued_i = np.zeros(length)
        accrued_ie = np.zeros(length)
        accrued_r = np.zeros(length)
        accrued_sn = np.zeros(length)
        accrued_n = np.zeros(length)

        for j in range(length):
            # I
            temp = infected[j:j + decay_window_size] * decay_window
            accrued_i[j] = sum(infected[j + decay_window_size:j + gamma + decay_window_size + 1]) + sum(temp)
            # Ie
            temp = infected_ie[j:j + decay_window_size] * decay_window
            accrued_ie[j] = sum(infected_ie[j + decay_window_size:j + gamma + decay_window_size + 1]) + sum(temp)
            # E
            accrued_e[j] = infected[j + decay_window_size + gamma - alpha] \
                           + infected_ie[j + decay_window_size + gamma - alpha]

            # R
            if j - gamma > 0:
                accrued_r[j] = sum(infected[decay_window_size + gamma:j + decay_window_size]) + \
                               sum(infected_ie[decay_window_size + gamma:j + decay_window_size])
            else:
                accrued_r[j] = 0
            # S
            N = base_population + (j + 1) * increase_population
            accrued_s[j] = N - accrued_e[j] - accrued_i[j] - accrued_ie[j] - accrued_r[j]
            # SN
            accrued_sn[j] = accrued_s[j] / N
            accrued_n[j] = N

        date = frame_morbidity['time']
        frame_accrued = pd.DataFrame(
            zip(date, accrued_s, accrued_e, accrued_i, accrued_ie, accrued_r, accrued_sn, accrued_n,infected[decay_window_size+gamma:],morbidity,diagnosis),
            columns=['date', 'S', 'E', 'I', 'Ie', 'R', 'S_N', 'N','label','morbidity','diagnosis'])
        writer = pd.ExcelWriter(output_path + str(year) + 'SEIIeR_timeDelay_Born.xlsx')
        frame_morbidity.to_excel(writer, sheet_name='morbidity', index=False)
        frame_diagnosis.to_excel(writer, sheet_name='diagnosis', index=False)
        frame_accrued.to_excel(writer, sheet_name='accrued', index=False)
        writer.save()

def SEIIeR_timeDalay_Born_Decay(args):
    input_path = args.case_path + 'raw/dailyIncrease/'
    output_path = args.case_path + 'SEIIeR_timeDalay_Born_Decay/'
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    decay_window = config.DECAY_WINDOW
    decay_window_size = len(decay_window)

    alpha = args.incubation_days
    gamma = args.recover_days

    p = args.p
    years = config.YEAR
    born_population = config.BORN_POPULATION

    for year in years:
        if year==2014:
            kkk=3
        index = year - 2010
        days = 366 if calendar.isleap(year) else 365
        increase_population = born_population[index + 5] / days
        decay_population=born_population[index]/days

        base_population = sum(born_population[index:index + 5])

        frame_morbidity = pd.read_excel(input_path + str(year) + '.xlsx', sheet_name='morbidity')
        frame_diagnosis = pd.read_excel(input_path + str(year) + '.xlsx', sheet_name='diagnosis')

        morbidity = frame_morbidity['total'].values
        diagnosis = frame_diagnosis['total'].values

        morbidity, diagnosis = smooth(morbidity, diagnosis, args)
        year_dict = {'morbidity': morbidity, 'diagnosis': diagnosis}

        if year != 2010:
            pre_morbidity = pd.read_excel(input_path + str(year - 1) + '.xlsx', sheet_name='morbidity')['total'].values
            pre_diagnosis = pd.read_excel(input_path + str(year - 1) + '.xlsx', sheet_name='diagnosis')['total'].values

            pre_morbidity, pre_diagnosis = smooth(pre_morbidity, pre_diagnosis, args)

        else:
            pre_morbidity = np.zeros(gamma + decay_window_size)
            pre_diagnosis = np.zeros(gamma + decay_window_size)

        pre_morbidity = pre_morbidity[-(gamma + decay_window_size):]
        pre_diagnosis = pre_diagnosis[-(gamma + decay_window_size):]
        prev_dict = {'morbidity': pre_morbidity, 'diagnosis': pre_diagnosis}

        pre_infected = prev_dict.get(args.label_name)
        infected = year_dict.get(args.label_name)

        length = len(infected)  # 当前年份的时间长度

        # 合并窗口
        infected = np.concatenate((pre_infected, infected), axis=0)
        # 每日新增的显性感染者和隐性感染者具有一定的比率|利用广播
        infected_ie = infected / p * (1 - p)

        accrued_s = np.zeros(length)
        accrued_e = np.zeros(length)
        accrued_i = np.zeros(length)
        accrued_ie = np.zeros(length)
        accrued_r = np.zeros(length)
        accrued_sn = np.zeros(length)
        accrued_n = np.zeros(length)

        #计算初始累积治愈者
        frame_age = pd.read_excel(args.utildata_path, sheet_name='age')
        y_0 = sum(frame_age.iloc[index, [2, 3, 4, 5, 6]])
        y_1 = sum(frame_age.iloc[index + 1, [2, 3, 4, 5]])
        y_2 = sum(frame_age.iloc[index + 2, [2, 3, 4]])
        y_3 = sum(frame_age.iloc[index + 3, [2, 3]])
        y_4 = sum(frame_age.iloc[index + 4, [2]])
        y_0 = y_0 + y_0 / p * (1 - p)
        y_1 = y_1 + y_1 / p * (1 - p)
        y_2 = y_2 + y_2 / p * (1 - p)
        y_3 = y_3 + y_3 / p * (1 - p)
        y_4 = y_4 + y_4 / p * (1 - p)
        R = int(y_0 + y_1 + y_2 + y_3 + y_4)
        for j in range(length):
            # I
            temp = infected[j:j + decay_window_size] * decay_window
            accrued_i[j] = sum(infected[j + decay_window_size:j + gamma + decay_window_size + 1]) + sum(temp)
            # Ie
            temp = infected_ie[j:j + decay_window_size] * decay_window
            accrued_ie[j] = sum(infected_ie[j + decay_window_size:j + gamma + decay_window_size + 1]) + sum(temp)
            # E
            accrued_e[j] = infected[j + decay_window_size + gamma - alpha] \
                           + infected_ie[j + decay_window_size + gamma - alpha]

            # R
            if j - gamma > 0:
                accrued_r[j] = R-decay_population*(j+1)+sum(infected[decay_window_size + gamma:j + decay_window_size]) + \
                               sum(infected_ie[decay_window_size + gamma:j + decay_window_size])
            else:
                accrued_r[j] = R-decay_population*(j+1)
            # S
            N = base_population + (j + 1) * increase_population
            accrued_s[j] = N - accrued_e[j] - accrued_i[j] - accrued_ie[j] - accrued_r[j]
            # SN
            accrued_sn[j] = accrued_s[j] / N
            accrued_n[j] = N

        date = frame_morbidity['time']
        frame_accrued = pd.DataFrame(
            zip(date, accrued_s, accrued_e, accrued_i, accrued_ie, accrued_r, accrued_sn, accrued_n,
                infected[decay_window_size + gamma:], morbidity, diagnosis),
            columns=['date', 'S', 'E', 'I', 'Ie', 'R', 'S_N', 'N', 'label', 'morbidity', 'diagnosis'])
        writer = pd.ExcelWriter(output_path + str(year) + 'SEIIeR_timeDalay_Born_Decay.xlsx')
        frame_morbidity.to_excel(writer, sheet_name='morbidity', index=False)
        frame_diagnosis.to_excel(writer, sheet_name='diagnosis', index=False)
        frame_accrued.to_excel(writer, sheet_name='accrued', index=False)
        writer.save()


def preprocess(args):
    SEIIeR_timeDelay_Born(args)
    SEIIeR_timeDelay_noBorn(args)
    SEIIeR_timeDalay_Born_Decay(args)




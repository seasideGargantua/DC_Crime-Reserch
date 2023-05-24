# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 14:24
# @Author  : seasideGargantua
# @Site    : https://github.com/seasideGargantua
# @File    : DataLoader.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import os


class DCCrimeDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.data_paths = os.listdir(data_root)
        self.dataset = self.load_raw_data(self.data_paths)
        self.crime_weight = {
            'CHI_UK': {
                'ARSON': 0.6, 'ASSAULT W/DANGEROUS WEAPON': 0.4, 'BURGLARY': 0.4,
                'HOMICIDE': 100., 'ROBBERY': 6.7, 'MOTOR VEHICLE THEFT': 0.4,
                'SEX ABUSE': 6.7, 'THEFT F/AUTO': 0., 'THEFT/OTHER': 0.
            },
            'CHI_US': {
                'ARSON': 22.3, 'ASSAULT W/DANGEROUS WEAPON': 12.4, 'BURGLARY': 14.9,
                'HOMICIDE': 52.1, 'ROBBERY': 27.3, 'MOTOR VEHICLE THEFT': 22.3,
                'SEX ABUSE': 100., 'THEFT F/AUTO': 0., 'THEFT/OTHER': 0.
            },
            'NSCS': {
                'ARSON': 56.4, 'ASSAULT W/DANGEROUS WEAPON': 41., 'BURGLARY': 15.4,
                'HOMICIDE': 100., 'ROBBERY': 23.1, 'MOTOR VEHICLE THEFT': 20.5,
                'SEX ABUSE': 51.3, 'THEFT F/AUTO': 17.9, 'THEFT/OTHER': 7.7
            },
            'NORMAL': {
                'ARSON': 100., 'ASSAULT W/DANGEROUS WEAPON': 100., 'BURGLARY': 100.,
                'HOMICIDE': 100., 'ROBBERY': 100., 'MOTOR VEHICLE THEFT': 100.,
                'SEX ABUSE': 100., 'THEFT F/AUTO': 100., 'THEFT/OTHER': 100.
            }
        }

    def load_raw_data(self, data_paths):
        '''
        读取初始数据，以列表形式存储，每个元素代表一年的数据
        :param data_paths:[list]数据读取的路径
        :return: dataset[dict]
        '''
        dataset = {}
        for idx,data_path in enumerate(data_paths):
            tmp = self.data_paths[idx].split('_')[-1]
            year = tmp.split('.')[0]
            dataset[year] = pd.read_csv(self.data_root+data_path)
        return dataset

    def create_time_series(self, res, about_year=None):
        '''
        将初始数据中每年中不同粒度的犯罪量统计出来（以犯罪开始时间为准）,
        并且是根据犯罪类型来进行统计
        :param res: [str]year,month,day之一，用来选择分辨率
        :param about_year: [str]需要使用特定年份数据时传入该年份
        :return: time_series_data[DataFrame]
        '''
        crime_type_list = []
        for year, data in self.dataset.items():
            tmp_type_list = list(set(data['OFFENSE'].tolist()))
            crime_type_list += tmp_type_list
        crime_type_list = list(set(crime_type_list))
        tmp_data = {}
        if about_year:
            # 将数据存在字典中计数每天的犯罪率
            if res == 'day':
                for idx,start_time in enumerate(self.dataset[about_year]['REPORT_DAT']):
                    new_start_time = str(start_time).split()[0]
                    crime_type = str(self.dataset[about_year]['OFFENSE'][idx])
                    crime_idx = crime_type_list.index(crime_type)
                    if new_start_time == 'nan':
                        continue
                    if new_start_time not in tmp_data:
                        tmp_data[new_start_time] = np.zeros(len(crime_type_list),dtype=int)
                        tmp_data[new_start_time][crime_idx] += 1
                    else:
                        tmp_data[new_start_time][crime_idx] += 1
            # 将数据存在字典中计数每月的犯罪率
            elif res == 'month':
                for idx,start_time in enumerate(self.dataset[about_year]['REPORT_DAT']):
                    new_start_time = str(start_time).split()[0]
                    crime_type = str(self.dataset[about_year]['OFFENSE'][idx])
                    crime_idx = crime_type_list.index(crime_type)
                    if new_start_time == 'nan':
                        continue
                    tmp_month_data = new_start_time.split('/')
                    month_data = tmp_month_data[0] + '/' + tmp_month_data[1]
                    if month_data not in tmp_data:
                        tmp_data[month_data] = np.zeros(len(crime_type_list),dtype=int)
                        tmp_data[month_data][crime_idx] += 1
                    else:
                        tmp_data[month_data][crime_idx] += 1
            # 将数据存在字典中计数每年的犯罪率
            else:
                for idx,start_time in enumerate(self.dataset[about_year]['REPORT_DAT']):
                    new_start_time = str(start_time).split()[0]
                    crime_type = str(self.dataset[about_year]['OFFENSE'][idx])
                    crime_idx = crime_type_list.index(crime_type)
                    if new_start_time == 'nan':
                        continue
                    year_data = new_start_time.split('/')[0]
                    if year_data not in tmp_data:
                        tmp_data[year_data] = np.zeros(len(crime_type_list),dtype=int)
                        tmp_data[year_data][crime_idx] += 1
                    else:
                        tmp_data[year_data][crime_idx] += 1
        else:
            # 将数据存在字典中计数每天的犯罪率
            if res == 'day':
                for year, data in self.dataset.items():
                    for idx,start_time in enumerate(data['REPORT_DAT']):
                        new_start_time = str(start_time).split()[0]
                        crime_type = str(data['OFFENSE'][idx])
                        crime_idx = crime_type_list.index(crime_type)
                        if new_start_time == 'nan':
                            continue
                        if new_start_time not in tmp_data:
                            tmp_data[new_start_time] = np.zeros(len(crime_type_list),dtype=int)
                            tmp_data[new_start_time][crime_idx] += 1
                        else:
                            tmp_data[new_start_time][crime_idx] += 1
            # 将数据存在字典中计数每月的犯罪率
            elif res == 'month':
                for year, data in self.dataset.items():
                    for idx,start_time in enumerate(data['REPORT_DAT']):
                        new_start_time = str(start_time).split()[0]
                        crime_type = str(data['OFFENSE'][idx])
                        crime_idx = crime_type_list.index(crime_type)
                        if new_start_time == 'nan':
                            continue
                        tmp_month_data = new_start_time.split('/')
                        month_data = tmp_month_data[0]+'/'+tmp_month_data[1]
                        if month_data not in tmp_data:
                            tmp_data[month_data] = np.zeros(len(crime_type_list),dtype=int)
                            tmp_data[month_data][crime_idx] += 1
                        else:
                            tmp_data[month_data][crime_idx] += 1
            # 将数据存在字典中计数每年的犯罪率
            else:
                for year, data in self.dataset.items():
                    for idx,start_time in enumerate(data['REPORT_DAT']):
                        new_start_time = str(start_time).split()[0]
                        crime_type = str(data['OFFENSE'][idx])
                        crime_idx = crime_type_list.index(crime_type)
                        if new_start_time == 'nan':
                            continue
                        year_data = new_start_time.split('/')[0]
                        if year_data not in tmp_data:
                            tmp_data[year_data] = np.zeros(len(crime_type_list),dtype=int)
                            tmp_data[year_data][crime_idx] += 1
                        else:
                            tmp_data[year_data][crime_idx] += 1
        # 将字典中的数据组织成DataFrame
        num_list = []
        date_list = []
        crime_type_list.append('total')
        for time, crime_num in tmp_data.items():
            total_num = np.sum(crime_num)
            date_list.append(time)
            tmp_list = list(crime_num)
            tmp_list.append(total_num)
            num_list.append(tmp_list)
        time_series_data = pd.DataFrame(data=num_list, columns=crime_type_list, index=date_list)
        return time_series_data

    def add_season_label(self, res, about_year=None):
        '''
        为数据添加季节标签
        :param res: [str]year,month,day之一，用来选择分辨率
        :param about_year: [str]需要使用特定年份数据时传入该年份
        :return:time_series_data[DataFrame]
        '''
        # 输入分辨率为年时，无法添加季节标签，异常处理
        if res == 'year':
            raise Exception('Year resolution don\'t support add season label!')
        # 读取时序数据
        time_series_data = self.create_time_series(res, about_year=about_year)
        # 添加一列季节标签
        time_series_data.insert(time_series_data.shape[1], 'season', 'spring')
        # 定义季节包含的月份
        spring = ['01', '02', '03']
        summer = ['04', '05', '06']
        autumn = ['07', '08', '09']
        winter = ['10', '11', '12']
        for idx, time in enumerate(time_series_data.index):
            month = time.split('/')[1]
            if month in spring:
                continue
            elif month in summer:
                time_series_data['season'][idx] = 'summer'
            elif month in autumn:
                time_series_data['season'][idx] = 'autumn'
            else:
                time_series_data['season'][idx] = 'winter'
        return time_series_data

    def weight_for_crime(self, weight_type='CHI_US', about_year=None):
        '''
        根据不同的犯罪赋权方法生成加权数据
        :param about_year: [str]使用所有数据或针对某一年
        :param weight_type: [str]CHI_UK,CHI_US,NSCS,NORMAL之一，加权的方法
        :return: weigh_data[DataFrame]存储经纬度坐标和权重的数据
        '''
        weight = []
        tmp_data = {}
        if about_year:
            lat = self.dataset[about_year]['Y'].tolist()
            lng = self.dataset[about_year]['X'].tolist()
            for idx in range(len(self.dataset[about_year])):
               weight.append(self.crime_weight[weight_type][self.dataset[about_year]['OFFENSE'][idx]]/100)
            tmp_data['lat'] = lat
            tmp_data['lng'] = lng
            tmp_data['weight'] = weight
        else:
            for year, data in self.dataset.items():
                tmp_data[year]['lat'] = []
                tmp_data[year]['lng'] = []
                tmp_data[year]['weight'] = []
                lat = self.dataset[year]['Y'].tolist()
                lng = self.dataset[year]['X'].tolist()
                for idx in range(len(self.dataset[year])):
                    weight.append(self.crime_weight[weight_type][self.dataset[year]['OFFENSE'][idx]]/100)
                tmp_data[year]['lat'] += lat
                tmp_data[year]['lng'] += lng
                tmp_data[year]['weight'] += weight
        weight_data = pd.DataFrame(tmp_data)
        return weight_data


class DC311ServicesDataset:
    def __init__(self, data_root):
        self.data_root = data_root
        self.data_paths = os.listdir(data_root)
        self.dataset = self.load_raw_data(self.data_paths)
        self.service_type_list = []
        for year, data in self.dataset.items():
            tmp_type_list = list(set(data['SERVICECODEDESCRIPTION'].tolist()))
            self.service_type_list += tmp_type_list
        self.service_type_list = list(set(self.service_type_list))
        self.service_type_list.append('total')

    def load_raw_data(self, data_paths):
        '''
        读取初始数据，以列表形式存储，每个元素代表一年的数据
        :param data_paths:[list]数据读取的路径
        :return: dataset[dict]
        '''
        dataset = {}
        for idx,data_path in enumerate(data_paths):
            tmp = self.data_paths[idx].split('_')[-1]
            year = tmp.split('.')[0]
            dataset[year] = pd.read_csv(self.data_root+data_path)
        return dataset

    def create_time_series(self, res, about_year=None):
        '''
        将初始数据中每年中不同粒度的报修量统计出来
        :param res: [str]year,month,day之一，用来选择分辨率
        :param about_year: [str]需要使用特定年份数据时传入该年份
        :return: time_series_data[DataFrame]
        '''
        tmp_data = {}
        if about_year:
            # 将数据存在字典中计数每天的服务量
            if res == 'day':
                for idx, start_time in enumerate(self.dataset[about_year]['ADDDATE']):
                    new_start_time = str(start_time).split()[0]
                    service_type = str(self.dataset[about_year]['SERVICECODEDESCRIPTION'][idx])
                    service_idx = self.service_type_list.index(service_type)
                    if new_start_time == 'nan':
                        continue
                    if new_start_time not in tmp_data:
                        tmp_data[new_start_time] = np.zeros(len(self.service_type_list), dtype=int)
                        tmp_data[new_start_time][service_idx] += 1
                    else:
                        tmp_data[new_start_time][service_idx] += 1
            # 将数据存在字典中计数每月的服务量
            elif res == 'month':
                for idx, start_time in enumerate(self.dataset[about_year]['ADDDATE']):
                    new_start_time = str(start_time).split()[0]
                    service_type = str(self.dataset[about_year]['SERVICECODEDESCRIPTION'][idx])
                    service_idx = self.service_type_list.index(service_type)
                    if new_start_time == 'nan':
                        continue
                    tmp_month_data = new_start_time.split('/')
                    month_data = tmp_month_data[0] + '/' + tmp_month_data[1]
                    if month_data not in tmp_data:
                        tmp_data[month_data] = np.zeros(len(self.service_type_list), dtype=int)
                        tmp_data[month_data][service_idx] += 1
                    else:
                        tmp_data[month_data][service_idx] += 1
            # 将数据存在字典中计数每年的服务量
            else:
                for idx, start_time in enumerate(self.dataset[about_year]['ADDDATE']):
                    new_start_time = str(start_time).split()[0]
                    service_type = str(self.dataset[about_year]['SERVICECODEDESCRIPTION'][idx])
                    service_idx = self.service_type_list.index(service_type)
                    if new_start_time == 'nan':
                        continue
                    year_data = new_start_time.split('/')[0]
                    if year_data not in tmp_data:
                        tmp_data[year_data] = np.zeros(len(self.service_type_list), dtype=int)
                        tmp_data[year_data][service_idx] += 1
                    else:
                        tmp_data[year_data][service_idx] += 1
        else:
            # 将数据存在字典中计数每天的服务量
            if res == 'day':
                for year, data in self.dataset.items():
                    for idx, start_time in enumerate(data['ADDDATE']):
                        new_start_time = str(start_time).split()[0]
                        service_type = str(data['SERVICECODEDESCRIPTION'][idx])
                        service_idx = self.service_type_list.index(service_type)
                        if new_start_time == 'nan':
                            continue
                        if new_start_time not in tmp_data:
                            tmp_data[new_start_time] = np.zeros(len(self.service_type_list), dtype=int)
                            tmp_data[new_start_time][service_idx] += 1
                        else:
                            tmp_data[new_start_time][service_idx] += 1
            # 将数据存在字典中计数每月的犯服务量
            elif res == 'month':
                for year, data in self.dataset.items():
                    for idx, start_time in enumerate(data['ADDDATE']):
                        new_start_time = str(start_time).split()[0]
                        service_type = str(data['SERVICECODEDESCRIPTION'][idx])
                        service_idx = self.service_type_list.index(service_type)
                        if new_start_time == 'nan':
                            continue
                        tmp_month_data = new_start_time.split('/')
                        month_data = tmp_month_data[0]+'/'+tmp_month_data[1]
                        if month_data not in tmp_data:
                            tmp_data[month_data] = np.zeros(len(self.service_type_list), dtype=int)
                            tmp_data[month_data][service_idx] += 1
                        else:
                            tmp_data[month_data][service_idx] += 1
            # 将数据存在字典中计数每年的服务量
            else:
                for year, data in self.dataset.items():
                    for idx, start_time in enumerate(data['ADDDATE']):
                        new_start_time = str(start_time).split()[0]
                        service_type = str(data['SERVICECODEDESCRIPTION'][idx])
                        service_idx = self.service_type_list.index(service_type)
                        if new_start_time == 'nan':
                            continue
                        year_data = new_start_time.split('/')[0]
                        if year_data not in tmp_data:
                            tmp_data[year_data] = np.zeros(len(self.service_type_list), dtype=int)
                            tmp_data[year_data][service_idx] += 1
                        else:
                            tmp_data[year_data][service_idx] += 1
        # 将字典中的数据组织成DataFrame
        num_list = []
        date_list = []
        for time, service_num in tmp_data.items():
            total_num = np.sum(service_num)
            date_list.append(time)
            service_num[-1] = total_num
            tmp_list = list(service_num)
            num_list.append(tmp_list)
        time_series_data = pd.DataFrame(data=num_list, columns=self.service_type_list, index=date_list)
        return time_series_data

    def sparse_data_process(self, time_series_data, threshold=0.05):
        '''
        由于该数据中许多服务是平时较少提出的，这些数据可能是影响不大，
        但也可能是少数起重要作用的服务，因此将其分离出来
        :param time_series_data:[DataFrame]需要进行处理的时序数据
        :param threshold:[float]设定的阈值，当服务占比小于该值时将其认定为低频服务
        :return: LF_data[DataFrame],HF_data[DataFrame]
        '''
        time_series_data.loc['Col_sum'] = time_series_data.apply(lambda x: x.sum())  # 各列求和，添加新的行
        # 初始化低频服务和高频服务
        LF_data = {}
        HF_data = {}
        for service in self.service_type_list:
            if service == 'total':
                continue
            frequency = time_series_data[service]['Col_sum']/time_series_data['total']['Col_sum']
            if frequency < threshold:
                LF_data[service] = time_series_data[service].tolist()
            else:
                HF_data[service] = time_series_data[service].tolist()
        return pd.DataFrame(LF_data,index=time_series_data.index), pd.DataFrame(HF_data,index=time_series_data.index)

    def weight_for_response(self, about_year=None):
        '''
        根据社区响应服务的速度生成加权数据
        :param about_year: [str]使用所有数据或针对某一年
        :return: weigh_data[DataFrame]存储经纬度坐标和权重的数据
        '''
        weight = []
        tmp_data = {}
        if about_year:
            lat = self.dataset[about_year]['Y'].tolist()
            lng = self.dataset[about_year]['X'].tolist()
            for idx in range(len(self.dataset[about_year])):
                start_time = self.dataset[about_year]['ADDDATE'][idx]
                solve_time = self.dataset[about_year]['RESOLUTIONDATE'][idx]
                # 当问题未被解决则将权重赋值为1
                if str(solve_time) == 'nan':
                    weight.append(1)
                else:
                    start_year, start_month, start_day = start_time.split()[0].split('/')
                    solve_year, solve_month, solve_day = solve_time.split()[0].split('/')
                    if int(solve_year) - int(start_year) > 0:
                        weight.append(1)
                    elif int(solve_month) - int(start_month) > 6:
                        weight.append(0.75)
                    elif int(solve_month) - int(start_month) > 3:
                        weight.append(0.5)
                    elif int(solve_month) - int(start_month) > 0:
                        weight.append(0.25)
                    elif int(solve_day) - int(start_day) > 7:
                        weight.append(0.1)
                    else:
                        weight.append(0)
            tmp_data['lat'] = lat
            tmp_data['lng'] = lng
            tmp_data['weight'] = weight
        else:
            for year, data in self.dataset.items():
                tmp_data['lat'] = []
                tmp_data['lng'] = []
                tmp_data['weight'] = []
                lat = self.dataset[about_year]['Y'].tolist()
                lng = self.dataset[about_year]['X'].tolist()
                for idx in range(len(self.dataset[year])):
                    start_time = self.dataset[year]['ADDDATE'][idx]
                    solve_time = self.dataset[year]['RESOLUTIONDATE'][idx]
                    # 当问题未被解决则将权重赋值为1
                    if str(solve_time) == 'nan':
                        weight.append(1)
                    else:
                        start_year, start_month, start_day = start_time.split()[0].split('/')
                        solve_year, solve_month, solve_day = solve_time.split()[0].split('/')
                        if int(solve_year) - int(start_year) > 0:
                            weight.append(1)
                        elif int(solve_month) - int(start_month) > 6:
                            weight.append(0.75)
                        elif int(solve_month) - int(start_month) > 3:
                            weight.append(0.5)
                        elif int(solve_month) - int(start_month) > 0:
                            weight.append(0.25)
                        elif int(solve_day) - int(start_day) > 7:
                            weight.append(0.1)
                        else:
                            weight.append(0)
                tmp_data[year]['lat'] += lat
                tmp_data[year]['lng'] += lng
                tmp_data[year]['weight'] += weight
        weight_data = pd.DataFrame(tmp_data)
        return weight_data


class DCCOVID19Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = pd.read_csv(self.data_path)

    def create_time_series(self, res):
        '''
        将初始数据中每年中不同粒度的犯罪量统计出来（以犯罪开始时间为准）,
        并且是根据犯罪类型来进行统计
        :param res: [str]year,month,day之一，用来选择分辨率
        :return: time_series_data[DataFrame]
        '''
        ward_list = list(set(self.dataset['WARD'].tolist()))
        tmp_data = {}
        # 将数据存在字典中计数每天的感染人数
        if res == 'day':
            for idx, start_time in enumerate(self.dataset['REPORT_DATE']):
                new_start_time = str(start_time).split()[0]
                ward = str(self.dataset['WARD'][idx])
                ward_idx = ward_list.index(ward)
                positive_num = self.dataset['POSITIVE_CASES'][idx]
                if str(positive_num) == 'nan':
                    continue
                if new_start_time not in tmp_data:
                    tmp_data[new_start_time] = np.zeros(len(ward_list), dtype=int)
                    tmp_data[new_start_time][ward_idx] += positive_num
                else:
                    tmp_data[new_start_time][ward_idx] += positive_num
        # 将数据存在字典中计数每月的感染人数
        elif res == 'month':
            for idx, start_time in enumerate(self.dataset['REPORT_DATE']):
                new_start_time = str(start_time).split()[0]
                ward = str(self.dataset['WARD'][idx])
                ward_idx = ward_list.index(ward)
                positive_num = self.dataset['POSITIVE_CASES'][idx]
                if str(positive_num) == 'nan':
                    continue
                tmp_month_data = new_start_time.split('/')
                month_data = tmp_month_data[0] + '/' + tmp_month_data[1]
                if month_data not in tmp_data:
                    tmp_data[month_data] = np.zeros(len(ward_list), dtype=int)
                    tmp_data[month_data][ward_idx] += positive_num
                else:
                    tmp_data[month_data][ward_idx] += positive_num
        # 将数据存在字典中计数每年的感染人数
        else:
            for idx, start_time in enumerate(self.dataset['REPORT_DATE']):
                new_start_time = str(start_time).split()[0]
                ward = str(self.dataset['WARD'][idx])
                ward_idx = ward_list.index(ward)
                positive_num = self.dataset['POSITIVE_CASES'][idx]
                if str(positive_num) == 'nan':
                    continue
                year_data = new_start_time.split('/')[0]
                if year_data not in tmp_data:
                    tmp_data[year_data] = np.zeros(len(ward_list), dtype=int)
                    tmp_data[year_data][ward_idx] += positive_num
                else:
                    tmp_data[year_data][ward_idx] += positive_num
        # 将字典中的数据组织成DataFrame
        num_list = []
        date_list = []
        ward_list.append('total')
        for time, positive_num in tmp_data.items():
            total_num = np.sum(positive_num)
            date_list.append(time)
            tmp_list = list(positive_num)
            tmp_list.append(total_num)
            num_list.append(tmp_list)
        time_series_data = pd.DataFrame(data=num_list, columns=ward_list, index=date_list)
        return time_series_data

    def data_construct_by_ward(self):
        '''
        将原始数据组织为更方便使用的形式
        :return:new_data[dict]
        '''
        new_data = {}
        for idx in range(len(self.dataset)):
            ward = self.dataset['WARD'][idx]
            date = self.dataset['REPORT_DATE'][idx]
            positive = self.dataset['POSITIVE_CASES'][idx]
            if str(positive) == 'nan':
                continue
            if ward not in new_data:
                new_data[ward] = []
            new_data[ward].append([date, positive])
        return new_data

    def data_construct_by_date(self):
        '''
        将原始数据组织为更方便使用的形式
        :return:new_data[dict]
        '''
        new_data = {}
        for idx in range(len(self.dataset)):
            ward = self.dataset['WARD'][idx]
            date = self.dataset['REPORT_DATE'][idx]
            positive = self.dataset['POSITIVE_CASES'][idx]
            if str(positive) == 'nan':
                continue
            if date not in new_data:
                new_data[date] = []
            new_data[date].append([ward, positive])
        return new_data


if __name__ == '__main__':
    DC_311_dataset = DC311ServicesDataset('./dataset/CSVs/311_City_Services/')
    raw_data = DC_311_dataset.dataset
    time_series_data = DC_311_dataset.create_time_series('month', about_year='2020')
    LF_data,HF_data = DC_311_dataset.sparse_data_process(time_series_data)
    data_with_weight = DC_311_dataset.weight_for_response(about_year='2020')
    # DC_dataset = DCCrimeDataset('./dataset/CSVs/Crime/')
    # time_series_data_with_season = DC_dataset.add_season_label('month', about_year='2020')
    # crime_with_weight = DC_dataset.weight_for_crime()
    # data_with_weight = DC_dataset.weight_for_crime(about_year='2020')
    # DC_COVID_dataset = DCCOVID19Dataset('./dataset/CSVs/COVID/DC_COVID-19_Cases_by_Ward.csv')
    # COVID_dataset = DC_COVID_dataset.create_time_series('month')
    # data_by_ward = DC_COVID_dataset.data_construct_by_ward()
    # data_by_date = DC_COVID_dataset.data_construct_by_date()

    print('here')

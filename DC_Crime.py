# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 15:23
# @Author  : seasideGargantua
# @Site    : https://github.com/seasideGargantua
# @File    : DC_Crime.py
# @Software: PyCharm

import pandas as pd
import os
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap,HeatMapWithTime
import numpy as np
from tqdm import tqdm
from DataLoader import DCCrimeDataset, DC311ServicesDataset, DCCOVID19Dataset
from pandas import DataFrame
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_predict
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose


def MakeHeatMap(crtype, df, year):
    # 获取需要分析的犯罪类型的经纬度数据
    lat = df['LATITUDE']
    lon = df['LONGITUDE']
    crlat = []
    crlon = []
    if crtype == 'all':
        crlat = list(lat)
        crlon = list(lon)
    else :
        for i in range(len(df)):
            if df['OFFENSE'][i] == crtype:
                crlat.append(lat[i])
                crlon.append(lon[i])

    # 定义folium地图
    map = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=12)

    # 设置瓦片图层
    folium.TileLayer('Stamen Toner').add_to(map)

    # 添加热力图
    HeatMap(
        list(zip(crlat, crlon)),
        radius=13,  # 每个点的半径范围
        gradient={0:'green', 0.2: 'lime', 0.4: 'blue',
                  0.6: 'orange', 0.8: 'red'},  # 分段颜色
        min_opacity=0.2,
        max_opacity=0.8
    ).add_to(map)

    # 保存地图，如果有特殊字符需要替换
    map.save('output/' + crtype.replace('/', '_') + '_'+year+'.html')


def MakeHeatMapWithTime(data, lat, lon, title):
    # 定义folium地图
    map = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=12)

    # 设置瓦片图层
    folium.TileLayer('Stamen Toner').add_to(map)

    # 添加热力图
    HeatMapWithTime(
        data,
        radius=8,  # 每个点的半径范围
        gradient={0:'green', 0.2: 'lime', 0.4: 'blue',
                  0.6: 'orange', 0.8: 'red'},  # 分段颜色
        min_opacity=0.2,
        max_opacity=0.8
    ).add_to(map)
    # map.fit_bounds(map.get_bounds())
    # 保存地图，如果有特殊字符需要替换
    map.save('output/' + title +'.html')


def MakeServiceHeatMap(df, year):
    # 获取需要分析的经纬度数据
    lat = df['lat']
    lon = df['lng']
    # 定义folium地图
    map = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=12)

    # 设置瓦片图层
    folium.TileLayer('Stamen Toner').add_to(map)

    # 添加热力图
    HeatMap(
        data=df[['lat', 'lng', 'weight']].values.tolist(),
        radius=13,  # 每个点的半径范围
        gradient={0:'green', 0.2: 'lime', 0.4: 'blue',
                  0.6: 'orange', 0.8: 'red'},  # 分段颜色
        min_opacity=0.2,
        max_opacity=0.8
    ).add_to(map)

    # 保存地图，如果有特殊字符需要替换
    map.save('output/服务加权热力图'+'_'+year+'.html')


def MakeHeatMapWithWeight(df, title):
    # 获取需要分析的经纬度数据
    lat = df['lat']
    lon = df['lng']
    # 定义folium地图
    map = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=12)

    # 设置瓦片图层
    folium.TileLayer('Stamen Toner').add_to(map)

    # 添加热力图
    HeatMap(
        data=df[['lat', 'lng', 'weight']].values.tolist(),
        radius=13,  # 每个点的半径范围
        gradient={0: 'green', 0.2: 'lime', 0.4: 'blue',
                  0.6: 'orange', 0.8: 'red'},  # 分段颜色
        min_opacity=0.2,
        max_opacity=0.8
    ).add_to(map)

    # 保存地图，如果有特殊字符需要替换
    map.save('output/'+title+'.html')


def TimeSeriesAnalyse(dataset, resolution):
    # 获取时间序列数据
    time_series_data = dataset.create_time_series(resolution)

    # fit model
    model = sm.tsa.arima.ARIMA(time_series_data['total'], order=(5, 1, 3))
    model_fit = model.fit()
    print(model_fit.summary())
    # plot residual errors
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    plt.savefig('output/time_series_10years_' + resolution + '_resid1.png')
    plt.show()
    residuals.plot(kind='kde')
    plt.savefig('output/time_series_10years_' + resolution + '_kde1.png')
    plt.show()
    print(residuals.describe())

    plt.figure()
    time_series_data.plot(label='原始数据')
    plot_predict(model_fit, label='预测数据')
    blue_line = mlines.Line2D([], [], linestyle='-', color='blue', markersize=2, label=u'原始数据图')
    red_line = mlines.Line2D([], [], linestyle='--', color='red', markersize=2, label=u'预测数据图')
    plt.legend(handles=[blue_line, red_line], loc='upper left')
    plt.grid(True)
    plt.savefig('output/time_series_10years_' + resolution + '_mf1.png')
    plt.show()
    print('Analyse completed!')


def draw_two_lines(title, x, y1, y2, xlabel, y1label, y2label, y1color, y2color):
    # 初始化画板
    fig, ax1 = plt.subplots(figsize=(12, 7))
    plt.suptitle(title)
    # 降低x轴显示密度，达到美观目的
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2 = ax1.twinx()  # 创建一个共享x轴的第二个y轴
    color = y1color
    # 绘制第一条折线
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # 绘制第二条折线
    color = y2color
    ax2.set_ylabel(y2label, color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # 保存以及输出
    plt.savefig('./output/'+title+'.png')
    plt.show()


def granger_causality(x, y, maxlag, test):
    # 因果分析
    data = np.column_stack((x, y))
    result = grangercausalitytests(data, maxlag, verbose=False)
    p_values = [round(result[i+1][0][test][1], 4) for i in range(maxlag)]
    return p_values


def draw_heatmap(source_label, df, lag):
    # 读取标签
    xlabel = list(source_label)
    ylabel = xlabel
    # 将DataFrame转化为numpy矩阵
    data = np.zeros((len(xlabel),len(ylabel)))
    for idx in range(len(df)):
        if df['Lag'][idx] != lag:
            continue
        x, y = df['Cause'][idx].split('-')
        x_idx = xlabel.index(x)
        y_idx = ylabel.index(y)
        data[x_idx][y_idx] = df['P-Value'][idx]

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(data, cmap='coolwarm')

    # 显示颜色刻度条
    cbar = ax.figure.colorbar(im, ax=ax)

    # 设置横纵坐标轴刻度
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # 显示横纵坐标轴刻度标签
    ax.set_xticklabels(xlabel)
    ax.set_yticklabels(ylabel)

    # 在每个格子内显示对应值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, data[i, j], ha='center', va='center', color='w')

    # 标题
    ax.set_title('因果分析热力图')

    # 旋转x轴刻度标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    # 保存和显示图片
    plt.savefig('./output/'+'因果分析_'+lag+'.png')
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    '''
    第一天：热力图生成以及热点分析
    '''
    # df = pd.read_csv('./dataset/Crime_Incidents_in_2019.csv')
    # date_axis = np.arange(1, 13)  # 生成月份数组
    # num_axis = np.zeros(12, dtype=int)  # 生成犯罪数量计数数组
    #
    # # 计数每个月报道的犯罪数量
    # for i in range(len(df)):
    #     month = int(str(df['REPORT_DAT'][i]).split('/')[1])
    #     num_axis[month-1] += 1
    # plt.plot(date_axis, num_axis, 'b.-', alpha=0.5, linewidth=1)  # 绘制折线图
    # plt.xlabel('month')  # x轴标签
    # plt.ylabel('number')  # y轴标签
    # plt.savefig('output/data_crime_num.png')  # 保存图片
    # plt.show()
    #
    # # 获取所有犯罪种类
    # crtype_list = ['all']
    # for i in tqdm(range(len(df))):
    #     if df['OFFENSE'][i] not in crtype_list:
    #         crtype_list.append(df['OFFENSE'][i])
    # print('犯罪种类获取完毕')
    #
    # # 获取所有犯罪种类的热力分析图
    # for crtype,_ in zip(crtype_list, tqdm(range(len(crtype_list)))):
    #     MakeHeatMap(crtype,df)
    # print('热力图分析完毕')

    # 读取十年华盛顿犯罪数据
    '''
    第二天：时序分析
    '''
    # DC_dataset = DCCrimeDataset('./dataset/CSVs/Crime/')
    # # 定义分析数据的粒度
    # resolution = 'month'
    # # 进行时序分析
    # TimeSeriesAnalyse(DC_dataset, resolution)
    '''
    第三天：犯罪种类之间关系可视化
    '''
    # DC_Crime_dataset = DCCrimeDataset('./dataset/CSVs/Crime/')
    # start = 2013
    # for i in range(11):
    #     cur_year = start+i
    #     Crime_Time_Series_Data = DC_Crime_dataset.create_time_series('month',about_year=str(cur_year))
    #     sns.pairplot(Crime_Time_Series_Data)
    #     # DC_Services_dataset = DC311ServicesDataset('./dataset/CSVs/311_City_Services/')
    #     # Services_Time_Series_Data = DC_Services_dataset.create_time_series('month', about_year='2023')
    #     # sns.pairplot(Services_Time_Series_Data)
    #     plt.savefig('./output/time_series_pairplot_'+str(cur_year)+'.png')

    # Crime_Time_Series_Data = DC_Crime_dataset.add_season_label('month', about_year=None)
    # sns.pairplot(Crime_Time_Series_Data, hue='season', palette="husl")
    # sns.pairplot(Crime_Time_Series_Data, kind="reg", diag_kind="kde")
    # plt.savefig('./output/time_series_pairplot_reg.png')
    # plt.show()
    '''
    第四天：时序分析、因果分析新冠、城市服务、犯罪率之间的关系
    '''
    # # 犯罪数据
    res = 'month'
    DC_Crime_Dataset = DCCrimeDataset('./dataset/CSVs/Crime/')
    Crime_2019 = DC_Crime_Dataset.create_time_series(res, about_year='2019')
    Crime_2020 = DC_Crime_Dataset.create_time_series(res, about_year='2020')
    Crime_2021 = DC_Crime_Dataset.create_time_series(res, about_year='2021')
    Crime_2022 = DC_Crime_Dataset.create_time_series(res, about_year='2022')
    Crime_2023 = DC_Crime_Dataset.create_time_series(res, about_year='2023')
    Crime_com = pd.concat([Crime_2019, Crime_2020, Crime_2021, Crime_2022, Crime_2023]).groupby(level=0).sum().sort_index()
    # 将两个DataFrame合并
    Crime_total = pd.DataFrame(Crime_com.loc[:, 'total']).rename(columns={'total': 'total_crime'})
    # 新冠数据
    DC_COVID_Dataset = DCCOVID19Dataset('./dataset/CSVs/COVID/DC_COVID-19_Cases_by_Ward.csv')
    COVID_time_series = DC_COVID_Dataset.create_time_series('month').sort_index()
    # 城市服务数据
    DC_Service_Dataset = DC311ServicesDataset('./dataset/CSVs/311_City_Services/')
    Service_2019 = DC_Service_Dataset.create_time_series(res, about_year='2019')
    Service_2020 = DC_Service_Dataset.create_time_series(res, about_year='2020')
    Service_2021 = DC_Service_Dataset.create_time_series(res, about_year='2021')
    Service_2022 = DC_Service_Dataset.create_time_series(res, about_year='2022')
    Service_2023 = DC_Service_Dataset.create_time_series(res, about_year='2023')
    Service_com = pd.concat([Service_2019, Service_2020, Service_2021, Service_2022, Service_2023]).groupby(level=0).sum().sort_index()
    Service_total = pd.DataFrame(Service_com.loc[:, 'total']).rename(columns={'total': 'total_service'})
    service_crime_df = pd.merge(Service_total, Crime_total, left_index=True, right_index=True, how='outer')
    LF_data, HF_data = DC_Service_Dataset.sparse_data_process(Service_com)
    HF_data.insert(loc=len(HF_data.columns), column='COVID', value=Service_com['Coronavirus (COVID-19) Tracking'])
    Covid_total = pd.DataFrame(COVID_time_series.loc[:, 'total']).rename(columns={'total': 'total_positive'})
    # 将两个DataFrame合并
    new_df = pd.merge(Service_total, Covid_total, left_index=True, right_index=True, how='outer')
    # 将重合时间段的数据提取出来
    common_df = new_df.loc[new_df['total_positive'].notnull(), :]
    # covid_crime_df = pd.merge(Crime_total, Covid_total, left_index=True, right_index=True, how='outer')
    # draw_two_lines('新冠城市服务分析', new_df.index, new_df['total_service'], new_df['total_positive'],
    #                '时期', '城市服务数量', '新冠感染人数', 'tab:blue', 'tab:red')
    # draw_two_lines('犯罪城市服务分析', service_crime_df.index,
    #                service_crime_df['total_crime'], service_crime_df['total_service'],
    #                '时期', '犯罪数量', '城市服务数量', 'tab:blue', 'tab:red')
    # draw_two_lines('新冠犯罪分析', covid_crime_df.index,
    #                covid_crime_df['total_crime'], covid_crime_df['total_positive'],
    #                '时期', '犯罪数量', '新冠感染人数', 'tab:blue', 'tab:red')
    # 将重合时间段的数据提取出来
    # common_df = new_df.loc[new_df['total_positive'].notnull(), :]
    # 计算相关系数
    '''
    这里直接使用三年的跨度算出的相关性系数太差，我们尝试使用一个滑动窗口来得到不同时间段内的相关性系数，查看哪些时间内最相关
    '''
    window = 6
    r = spearmanr(service_crime_df['total_service'], service_crime_df['total_crime'])
    print("皮尔逊相关系数为：", r)
    correlation = []
    pvalue = []
    stage = []
    time_list = service_crime_df.index
    for i in range(len(service_crime_df)-window):
        cor, pval = spearmanr(service_crime_df['total_service'][i:i+window], service_crime_df['total_crime'][i:i+window])
        correlation.append(cor)
        pvalue.append(pval)
        period = time_list[i]+'-'+time_list[i+window]
        stage.append(period)
        print(period+':'+'correlation=', cor, ', pvalue=', pval)
    draw_two_lines('服务-犯罪相关分析',stage, correlation, pvalue, '时间段', 'correlation', 'pvalue', 'tab:orange', 'tab:green')
    '''
    第五天：因果分析以及服务响应程度与犯罪率之间的关系分析
    '''
    year = '2020'
    result_df = pd.merge(Crime_com, Covid_total, left_index=True, right_index=True, how='outer')
    result_df = pd.merge(result_df, HF_data, left_index=True, right_index=True, how='outer')
    result_df = result_df.loc[result_df['total_positive'].notnull(), :]
    maxlag = 3  # 最大滞后期
    test = 'ssr_chi2test' # 检验方法
    p_values = []
    for i in result_df.columns:
        for j in result_df.columns:
            p_values.append(granger_causality(result_df[i], result_df[j], maxlag, test))
    cols = [f'{i}-{j}' for i in result_df.columns for j in result_df.columns]
    results = pd.DataFrame(np.array(p_values), columns=[f'lag_{i}' for i in range(1, maxlag + 1)])
    results.index = cols
    results.index.name = 'Cause'
    results.columns.name = 'Lag'
    results = results.stack().reset_index()
    results.columns = ['Cause', 'Lag', 'P-Value']
    draw_heatmap(result_df.columns, results, 'lag_1')
    # print(results)
    # weight_df = DC_Service_Dataset.weight_for_response(about_year='2020')
    # MakeServiceHeatMap(weight_df, year)
    # MakeHeatMap('all', DC_Crime_Dataset.dataset['2020'], year)
    # print('热力图分析完毕')

    # pca = PCA(n_components=2)
    # pca.fit(Service_com)
    # result = pca.transform(Service_com)
    # print(result)
    '''
    第六天：后续完善，季节分析、加权犯罪热力图、动态热力图
    '''
    # 分解数据查看季节性   period为周期
    # plt.rcParams['figure.figsize'] = 30, 15
    # ts_decomposition = seasonal_decompose(Crime_total['total_crime'], period=12)
    # ts_decomposition.plot()
    # plt.savefig('./output/season_year.png')
    # plt.show()
    '''
    犯罪热力图绘制
    '''
    # DC_Crime_Dataset = DCCrimeDataset('./dataset/CSVs/Crime/')
    # start = 2016
    # score_type = 'NORMAL'
    # all_data = []
    # data_with_weight = pd.DataFrame()
    # for i in range(8):
    #     cur_year = str(start + i)
    #     data_with_weight = DC_Crime_Dataset.weight_for_crime(score_type, about_year=cur_year)
    #     MakeHeatMapWithWeight(data_with_weight, '犯罪加权热力图'+'_'+score_type+'_'+cur_year)
    #     all_data.append(np.array(data_with_weight).tolist())
    # MakeHeatMapWithTime(all_data, data_with_weight['lat'], data_with_weight['lng'], '犯罪时序变化热力图'+score_type)
    '''
    市政服务质量热力图绘制
    '''
    # DC_Service_Dataset = DC311ServicesDataset('./dataset/CSVs/311_City_Services/')
    # start = 2016
    # all_data = []
    # data_with_weight = pd.DataFrame()
    # for i in range(8):
    #     cur_year = str(start + i)
    #     data_with_weight = DC_Service_Dataset.weight_for_response(about_year=cur_year)
    #     MakeHeatMapWithWeight(data_with_weight, '服务加权热力图'+'_'+cur_year)
    #     all_data.append(np.array(data_with_weight).tolist())
    # MakeHeatMapWithTime(all_data, data_with_weight['lat'], data_with_weight['lng'], '市政服务质量时序变化热力图')

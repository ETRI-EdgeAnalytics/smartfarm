import os
import numpy as np
import pandas as pd
import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--config_path', type=str, default='./configs/preprocessing.json')
conf = parser.parse_args()

def HD_cumulative_time(df=None):
    if df is None:
        raise ValueError(df)
    
    am_df = df[df['시간'] < 120000]
    am_suitable_df = am_df[(am_df['HD'] >= 2.8) & (am_df['HD'] <= 6)]
    am_unsuitable_df = am_df[am_df['HD'] < 1.1]
    
    sun_df = df[(df['시간'] >= 60000) & (df['시간'] <= 190000)]
    sun_suitable_df = sun_df[(sun_df['HD'] >= 2.8) & (sun_df['HD'] <= 6)]
    
    all_day_suitable_df = df[(df['HD'] >= 2.8) & (df['HD'] <= 6)]
    
    pm_df = df[df['시간'] >= 120000]
    pm_unsuitable_df = pm_df[pm_df['HD'] >= 15]
    
    HD_cumulative_list = [len(am_suitable_df)/60, len(sun_suitable_df)/60, len(all_day_suitable_df)/60,
                          len(pm_unsuitable_df)/60, len(am_unsuitable_df)/60]
    
    HD = pd.DataFrame([HD_cumulative_list])
    HD.columns = ['오전적정증산(HD)누적시간', '일출일몰적정증산(HD)누적시간', '하루적정증산(HD)누적시간', '오후심각증산(HD)누적시간', '오전심각증산(HD)누적시간']
    
    return HD

def temperature_cumulative_time(df=None):
    if df is None:
        raise ValueError(df)
    
    low_temp_df = df[df['실내온도'] <= 12]
    high_temp_df = df[df['실내온도'] >= 30]
    
    suitable_df = df[(df['실내온도'] >= 18) & (df['실내온도'] <= 20)]
    
    temp_cumulative_list = [len(low_temp_df)/60, len(high_temp_df)/60, len(suitable_df)/60]
    
    temp = pd.DataFrame([temp_cumulative_list])
    temp.columns = ['12도이하온도누적시간', '30도이상온도누적시간', '적정온도누적시간']
    
    return temp

def temperature_average(df=None):
    if df is None:
        raise ValueError(df)
    
    daytime_df = df[(df['시간'] >= 60000) & (df['시간'] <= 190000)]
    daytime_temp = daytime_df['실내온도'].mean()
    
#     nighttime_df = df[(df['시간'] < 60000) | (df['시간'] > 190000)]
#     nighttime_temp = nighttime_df['실내온도'].mean()
    
    afternoon_df = df[(df['시간'] >= 120000) & (df['시간'] <= 190000)]
    afternoon_temp = afternoon_df['실내온도'].mean()
    
    morning_df = df[(df['시간'] >= 60000) & (df['시간'] <= 120000)]
    morning_temp = morning_df['실내온도'].mean()
    
#     temp_avg_list = [daytime_temp, nighttime_temp, afternoon_temp]
    temp_avg_list = [daytime_temp, afternoon_temp, morning_temp]
    temp = pd.DataFrame([temp_avg_list])
#     temp.columns = ['주간평균온도', '야간평균온도', '오후부터일몰까지평균온도']
    temp.columns = ['주간평균온도', '오후부터일몰까지평균온도', '일출부터12시까지평균온도']
    
    return temp

def suitable_info(df=None):
    if df is None:
        raise ValueError(df)
    
    suitable_humid = df[(df['실내습도'] >= 65) & (df['실내습도'] <= 85)]
    
    day_temp_lowerbound = df['실내온도'].min() - 16
    day_temp_upperbound = df['실내온도'].max() - 26
    
    sunrise = df[(df['시간'] >= 50000) & (df['시간'] <= 70000)]
    sunrise_diff = sunrise['실내온도'].max() - sunrise['실내온도'].min()
    
    sunset = df[(df['시간'] >= 180000) & (df['시간'] <= 200000)]
    sunset_diff = sunset['실내온도'].max() - sunset['실내온도'].min()
    
    suitable_list = [len(suitable_humid)/60, sunrise_diff, sunset_diff, 
                     day_temp_lowerbound, day_temp_upperbound]
    
    suitable = pd.DataFrame([suitable_list])
    suitable.columns = ['적정습도누적시간', '일출전후한시간평균온도', '일몰전후한시간평균온도',
                       '적정온도변화폭(하위)', '적정온도변화폭(상위)']
    return suitable

def nighttime_info(df=None):
    if df is None:
        raise ValueError(df)
    
    date = df['날짜'].unique()
    today = df[df['날짜'] == date[0]]
    tomorrow = df[df['날짜'] == date[1]]
    
    sunset_df = today[today['시간'] > 190000]
    sunrise_df = tomorrow[(tomorrow['시간'] < 60000)]
    nighttime_df = pd.concat([sunset_df, sunrise_df])
    nighttime_temp = nighttime_df['실내온도'].mean()
    nighttime_hd_df= nighttime_df[(nighttime_df['HD'] >= 2.8) & (nighttime_df['HD'] <= 6)]
    nighttime_hd = len(nighttime_hd_df['HD']) / 60
    nighttime_humi = nighttime_df['실내습도'].mean()
    night_list = [nighttime_temp, nighttime_hd, nighttime_humi]
    
    night = pd.DataFrame([night_list])
    night.columns = ["야간평균온도", "일몰일출적합증산(HD)누적시간", "야간평균습도"]
    
    return night
    
def temperature_diff_info(df=None):
    if df is None:
        raise ValueError(df)
        
    date = df['날짜'].unique()
    count = 0
    if len(date) == 2:
        yesterday = df[df['날짜'] == date[0]][-60:]
        today = df[df['날짜'] == date[1]]
        new_today = pd.concat([yesterday, today])

        for i in range(60, len(new_today)):
            partial = new_today[i-60:i+1]
            diff = partial['실내온도'].max() - partial['실내온도'].min()
            if np.absolute(diff) >= 5:
                count += 1
    #         for t in range(1, 61):
    #             diff = new_today.iloc[i]['실내온도'] - new_today.iloc[i-t]['실내온도']
    #             if np.absolute(diff) >= 5:
    #                 count += 1
    else:
        for i in range(60, len(df)):
            partial = df[i-60:i+1]
            diff = partial['실내온도'].max() - partial['실내온도'].min()
            if np.absolute(diff) >= 5:
                count += 1
    
    hourly_temp_diff = pd.DataFrame([count])
    hourly_temp_diff.columns = ["시간당5도이상의변화수"]
    
    return hourly_temp_diff

def humidity_info(df=None): # 주간 평균 습도, 낮 12시부터 일몰까지 평균 습도, 일출부터 낮 12시까지 평균 습도
    if df is None:
        raise ValueError(df)
    daytime_df = df[(df['시간'] >= 60000) & (df['시간'] <= 190000)]
    daytime_humi = daytime_df['실내습도'].mean()
    
    afternoon_df = df[(df['시간'] >= 120000) & (df['시간'] <= 190000)]
    afternoon_humi = afternoon_df['실내습도'].mean()
    
    morning_df = df[(df['시간'] >= 60000) & (df['시간'] <= 120000)]
    morning_humi = morning_df['실내습도'].mean()

    humi_avg_list = [daytime_humi, afternoon_humi, morning_humi]
    humi = pd.DataFrame([humi_avg_list])
    humi.columns = ['주간평균습도', '오후부터일몰까지평균습도', '일출부터12시까지평균습도']
    
    return humi

def env_data_preprocessing(args):
    scr = args.env.src_path
    files = args.env.names
    save_name = os.path.join(args.env.dest_path, args.env.save_name)

    origin_headers = ['날짜', '시간', '실내온도', '실내습도', 'dew point (Td)', 'HD', 'Height above sea level', 'atmospheric pressure',
           'pws', 'W (moisture content)', 'h (enthalpy)', 'v (specific volume)', 'rho (density)', 
           'pw (Partial pressure) water vapor', 'Absolute Humidity (AH)', 'absoluteAH']

    env_df = pd.DataFrame()
    for f in files:
        file_path = os.path.join(scr, f)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')
        sheet_df.columns = origin_headers
        env_df = pd.concat([env_df, sheet_df])
    
    new_env_df = env_df.drop(env_df.index[0])
    new_env_df['날짜'] = new_env_df['날짜'].astype(float)
    new_env_df = new_env_df.drop(columns=['Height above sea level'])
    new_env_df = new_env_df.dropna(axis=0)
    new_env_df['시간'] = new_env_df['시간'].astype(float)

    headers = ['날짜', '시간','실내온도', '실내습도', '이슬점(Td)', '증산(HD)', '대기압',
       'PWS(?)', '수분량', '엔탈피', '비부피',
       '밀도', 'PW(?)수증기', '절대습도', '절대AH(?)']

    std_headers = []
    min_headers = []
    max_headers = []

    for h in headers:
        std_headers += [h + '표준편차']
        min_headers += ['최소' + h]
        max_headers += ['최대' + h]

    avg_env_df = pd.DataFrame()

    date_list = new_env_df['날짜'].unique()

    for i in range(0, len(date_list)-1):
        a = new_env_df[new_env_df['날짜'] == date_list[i]].mean().to_frame().T
        a.columns = headers
        s = new_env_df[new_env_df['날짜'] == date_list[i]].std().to_frame().T
        s.columns = std_headers
        min_ = new_env_df[new_env_df['날짜'] == date_list[i]].min().to_frame().T
        min_.columns = min_headers
        max_ = new_env_df[new_env_df['날짜'] == date_list[i]].max().to_frame().T
        max_.columns = max_headers
        
        hd = HD_cumulative_time(df=new_env_df[new_env_df['날짜'] == date_list[i]])
        tt = temperature_cumulative_time(df=new_env_df[new_env_df['날짜'] == date_list[i]])
        ta = temperature_average(df=new_env_df[new_env_df['날짜'] == date_list[i]])
        su = suitable_info(df=new_env_df[new_env_df['날짜'] == date_list[i]])
        n = nighttime_info(df=new_env_df[(new_env_df['날짜'] == date_list[i]) | (new_env_df['날짜'] == date_list[i+1])])
        if i > 0:
            td = temperature_diff_info(df=new_env_df[(new_env_df['날짜'] == date_list[i-1]) | (new_env_df['날짜'] == date_list[i])])
        else:
            td = temperature_diff_info(df=new_env_df[(new_env_df['날짜'] == date_list[i])])
        hu = humidity_info(df=new_env_df[new_env_df['날짜'] == date_list[i]])
        
        stats = pd.concat([a, s, min_, max_, hd, tt, ta, su, n, td, hu], axis=1)
        avg_env_df = pd.concat([avg_env_df, stats])

        avg_env_df = avg_env_df.reset_index()
        avg_env_df = avg_env_df.drop(columns=['index'])
        avg_env_df = avg_env_df.interpolate()
        avg_env_df = avg_env_df.drop(columns=['날짜표준편차', '최소날짜', '최대날짜', '시간', '시간표준편차', '최소시간', '최대시간'])
        avg_env_df['주야간온도차이'] = avg_env_df['주간평균온도'] - avg_env_df['야간평균온도']
        avg_env_df['주야간습도차이'] = avg_env_df['주간평균습도'] - avg_env_df['야간평균습도']

        avg_env_df['주야간온도차8도이상인날수'] = 0
        x = avg_env_df.iloc[0]['주야간온도차이']
        if np.absolute(x) >= 8:
            avg_env_df.iloc[0]['주야간온도차8도이상인날수'] = 1
                
        for i in range(1, len(avg_env_df)):
            x = avg_env_df.iloc[i]['주야간온도차이']
            if np.absolute(x) >= 8:
                avg_env_df.at[i, '주야간온도차8도이상인날수'] = avg_env_df.iloc[i - 1]['주야간온도차8도이상인날수'] + 1
            else:
                avg_env_df.at[i, '주야간온도차8도이상인날수'] = avg_env_df.iloc[i - 1]['주야간온도차8도이상인날수']

        avg_env_df.to_excel(save_name, na_rep=0, header=True, index=False)

def growth_data_preprocessing(args):
    scr = args.growth.src_path
    files = args.growth.names
    save_name = os.path.join(args.growth.dest_path, args.growth.save_name)

    file_path = os.path.join(scr, files)
    df = pd.ExcelFile(file_path)
    growth_df = pd.read_excel(df, 'Sheet1')
    
    growth_df['엽면적지수'] = growth_df['잎길이(cm)'] * growth_df['잎폭(cm)'] * growth_df['잎수(개)'] * (args.growth.planted_hills / args.growth.farm_area) * 0.6 / 10000

    growth_df['주간생육길이_생육상태'] = 0
    growth_df.loc[growth_df['주간생육길이(cm)'] < 15, '주간생육길이_생육상태'] = -1
    growth_df.loc[growth_df['주간생육길이(cm)'] > 20, '주간생육길이_생육상태'] = 1

    growth_df['줄기굵기_생육상태'] = 0
    growth_df.loc[growth_df['줄기굵기(mm)'] < 9, '줄기굵기_생육상태'] = -1
    growth_df.loc[growth_df['줄기굵기(mm)'] > 11, '줄기굵기_생육상태'] = 1

    growth_df['잎길이_생육상태'] = 0
    growth_df.loc[growth_df['잎길이(cm)'] < 25, '잎길이_생육상태'] = -1
    growth_df.loc[growth_df['잎길이(cm)'] > 35, '잎길이_생육상태'] = 1

    growth_df['입폭_생육상태'] = 0
    growth_df.loc[growth_df['잎폭(cm)'] < 20, '입폭_생육상태'] = -1
    growth_df.loc[growth_df['잎폭(cm)'] > 30, '입폭_생육상태'] = 1

    growth_df['입수_생육상태'] = 0
    growth_df.loc[growth_df['잎수(개)'] < 10, '입수_생육상태'] = -1
    growth_df.loc[growth_df['잎수(개)'] > 15, '입수_생육상태'] = 1

    growth_df['엽면적지수_생육상태'] = 0
    growth_df.loc[growth_df['엽면적지수'] < 3, '엽면적지수_생육상태'] = -1
    growth_df.loc[growth_df['엽면적지수'] > 3.5, '엽면적지수_생육상태'] = 1

    growth_df['개화화방위치_생육상태'] = 0
    growth_df.loc[growth_df['개화화방위치(cm)'] < 10, '개화화방위치_생육상태'] = -1
    growth_df.loc[growth_df['개화화방위치(cm)'] > 15, '개화화방위치_생육상태'] = 1

    growth_df['꽃과줄기거리_생육상태'] = 0
    growth_df.loc[growth_df['꽃과 줄기거리(cm)'] < 3, '꽃과줄기거리_생육상태'] = -1
    growth_df.loc[growth_df['꽃과 줄기거리(cm)'] > 4, '꽃과줄기거리_생육상태'] = 1

    growth_df['생육상태점수'] = growth_df['주간생육길이_생육상태'] + growth_df['주간생육길이_생육상태'] + \
                            growth_df['잎길이_생육상태'] + growth_df['입폭_생육상태'] + \
                            growth_df['입수_생육상태'] + growth_df['엽면적지수_생육상태'] + \
                            growth_df['개화화방위치_생육상태'] + growth_df['꽃과줄기거리_생육상태']
    growth_df['생장구분'] = 0
    growth_df.loc[growth_df['생육상태점수'] < 0, '생장구분'] = -1
    growth_df.loc[growth_df['생육상태점수'] > 0, '생장구분'] = 1

    growth_df.to_excel(save_name, na_rep=0, header=True, index=False)

if __name__ == "__main__":
    args = ConfLoader(conf.config_path).opt

    env_data_preprocessing(args)

    growth_data_preprocessing(args)
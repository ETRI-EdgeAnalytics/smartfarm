
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import openpyxl

__all__ = ['rf_dataloader']

def rf_dataloader(args):
    """ 
        데이터 전처리  
        입력: 생육, 환경
    """ 
    # 환경 정보 읽어오기
    path = args.input.path
    grw = args.input.grw_filename
    env = args.input.env_filename

    dl = args.settings.dataloader

    one_week = 7

    """ 
        생육 데이터 전처리 
    """
    df_grw = pd.read_excel(os.path.join(path, grw),engine='openpyxl')

    # 날짜 정의
    df_grw_date = pd.DataFrame(df_grw[dl.date].unique())

    # 날짜와 샘플에 따라 생육 데이터 정리
    df_grw_sort = df_grw.set_index(dl.sort_index)
    df_grw_sort = df_grw_sort.sort_index()

    # 주간 생육상태 features 추출 
    df_grw_feat = df_grw_sort.reindex(columns = dl.grw_features)

    # 주간 생육상태 features 전처리
    for [samp,date], rows in df_grw_feat.iterrows():
        df_grw_feat.loc[samp,:].replace(to_replace=0.0,
                                            method='ffill',inplace=True)

    # 주간 생육상태 target 지정
    df_grw_target = df_grw_feat \
        .shift(-1 * dl.shift_week) \
        .reindex(columns = dl.grw_features)
    
    assert (len(dl.grw_target) == len(dl.grw_features))

    for x in range(len(dl.grw_target)):
        df_grw_target.rename(columns = 
            {dl.grw_features[x] : dl.grw_target[x]}, inplace=True)  

    # 주간 생육 성장 features 추출
    df_diff_feat = df_grw_sort.reindex(columns = dl.diff_features)

    # 주간 생육 성장 features 전처리
    for [samp, date], rows in df_diff_feat.iterrows():
        df_diff_feat.loc[samp, :].replace(to_replace=0.0, 
                                            method='ffill', inplace=True)

    # 주간 생육상태 target 지정
    df_diff_target = df_grw_sort.shift(-1 * dl.shift_week) \
                                    .reindex(columns = dl.diff_target)

    # 생육 데이터 통합
    df_growth = pd.concat([
        df_diff_feat.reset_index(drop=True),
        df_grw_feat.reset_index(drop=True),
        df_diff_target.reset_index(drop=True),
        df_grw_target.reset_index(drop=True)], axis=1)

    """ 
        환경 데이터 전처리 
    """
    df_env = pd.read_excel(os.path.join(path, env), engine='openpyxl')


    # 주간 생육상태 features 추출 
    df_env = df_env.reindex(columns=dl.env_features)

    # 날짜 정리
    df_env_date   = pd.DataFrame(df_env['날짜'])
    grw_startdate = df_grw_date.iloc[0].values[0]
    grw_enddate   = df_grw_date.iloc[-1].values[0]
    env_startindex = df_env_date.index[df_env_date['날짜']==grw_startdate][0]

    # 생육 기록 날짜로부터 index 찾기 
    grw_index = []
    for idx, row in df_grw_date.iterrows():
        grw_index.append(df_env_date.index[
            df_env_date['날짜'] == row.values[0]][0])

    # 생육 기록 날짜 전후로 환경 날짜 index 찾기
    env_index = []
    for x in grw_index:
        arr = []
        for t in range(0, dl.date_range):
            arr.append(x + t - dl.past_range)
        env_index.append(arr)
    
    # 전후 환경 index 바탕으로 환경 feature 전부 합치기 
    df_env_feat = []
    for x in range(len(env_index)):
        df_env_feat.append(df_env.iloc[env_index[x]].stack().T.values)
    df_env_feat = pd.DataFrame(df_env_feat)
    
    # 편의를 위해 환경 데이터에 이름+숫자으로 표기
    env_cols_names = []
    for y in range(dl.date_range):
        for x in df_env.columns:
            env_cols_names.append(x + "_" + str(y))
    assert(len(env_cols_names) == len(df_env_feat.columns))
    for x in range(len(env_cols_names)):
        df_env_feat.rename(columns = 
                            {df_env_feat.columns[x] : env_cols_names[x]}, 
                            inplace = True)

    """ 
        환경 및 생육 데이터 통합 및 처리 
    """

    # 환경 생육 데이터 통합
    df_env_feat = pd.concat([df_env_feat] * 16, axis = 0)
    dataset = pd.concat([df_env_feat.reset_index(drop = True),
                        df_growth.reset_index(drop = True)], axis = 1)

    # 최초 날짜 및 최종 날짜 삭제
    drop_weeks = []
    start_week = 0
    end_week = dl.total_weeks - 1
    for x in range(int(dataset.shape[0] / dl.total_weeks)):
        drop_weeks.append(start_week + (dl.total_weeks * x))
        drop_weeks.append(end_week + (dl.total_weeks * x))
    dataset.drop(drop_weeks,inplace=True)
   
    # 날짜_N 열 지우기
    date_names = []
    for y in range(dl.date_range):
        date_names.append("날짜" + "_" + str(y))   
    dataset.drop(date_names,axis=1,inplace=True)

    # 전처리 완료
    print("dataloader complete...")
    return dataset
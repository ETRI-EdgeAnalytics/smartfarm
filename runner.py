import os, sys
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

import pandas as pd

import numpy as np
from numpy import asarray, concatenate

import xgboost as xgb
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump, load
from sklearn.metrics import mean_squared_error

import plotly_express as px
import plotly.offline as po
import plotly.graph_objects as go

from utils.radar import ComplexRadar
import json

from models import *
from utils import *
from dataloader import *
from optimizer import *

__all__ = ['Runner']

class Runner():
    def __init__(self, args):
        self.args = args

    def train(self):
        # ihshin
        if self.args.mode == 'multi_encoder_train':
            multi_encoder_train(self.args)
        # jypark
        elif self.args.mode == 'rf_train':
            rf_train(self.args)
        else:
            raise ValueError("Wrong Predict Argument for Runner.train")
    
    def infer(self):
        # ihshin
        if self.args.mode == 'multi_encoder_infer':
            multi_encoder_infer(self.args)
        # jypark
        elif self.args.mode == 'rf_infer':
            rf_infer(self.args)
        else:
            raise ValueError("Wrong Predict Argument for Runner.infer")

    def run(self):
        if 'train' in self.args.mode:
            self.train()
        elif 'infer' in self.args.mode:
            self.infer()
        else:
            raise ValueError("Wrong Predict Argument for Runner.run")

def rf_train(args):
    """ 
    생육 환경 데이터 훈련
    """

    # 파일로부터 설정 불러옴
    print("Loading settings...")
    tr = args.settings.train 
    dl = args.settings.dataloader
    ou = args.output

    os.makedirs(ou.dir, exist_ok=True)
    os.makedirs(ou.model_dir, exist_ok=True)
    os.makedirs(ou.stat_dir, exist_ok=True)
    os.makedirs(ou.pred_dir, exist_ok=True)
    
    

    # 데이터 로드 및 전처리
    print("Setting dataloader...")
    ds = get_dataloader(args)

    # 훈련/검증 데이터 설정
    print("Spliting train/val features/labels...")
    X_train = ds.sample(frac=tr.ratio, random_state=0)
    X_val  = ds.drop(X_train.index)
    y_train = X_train[dl.grw_target + dl.diff_target].copy()
    y_test = X_val[dl.grw_target + dl.diff_target].copy()
    X_train.drop(dl.grw_target,axis=1,inplace=True)
    X_val.drop(dl.grw_target,axis=1,inplace=True)

    # 데이터 설정
    print("Normalizing numeric data...")
    train_stats = X_train.describe()
    train_stats = train_stats.transpose()
    test_stats = X_val.describe()
    test_stats = test_stats.transpose()
    X_train = (X_train - train_stats['mean']) / train_stats['std']
    X_val = (X_val - test_stats['mean']) / test_stats['std']
    config = [len(X_train.keys()),len(y_train.keys())]
    model = get_model(args, config)

    # 모델 훈련
    print("Model Fitting...")
    model.fit(X_train.values, y_train.values)

    # 모델 파일 출력
    print("output model file")
    file_path = os.path.join(ou.model_dir, ou.model_file)
    dump(model, file_path)

    # 모델 검증
    print("Model Testing...")
    y_pred = pd.DataFrame(model.predict(X_val))

    # 검증 결과, 점수, Feature중요 분석표 출력
    print("Printing out prediction results...")
    pred_path = os.path.join(ou.pred_dir , ou.pred_file)
    y_pred.to_csv(pred_path)
    

    s = []
    for x in range(len(y_pred.columns)):
        pred = y_pred.iloc[:,x]
        grou = y_test.iloc[:,x]
        s.append(mean_squared_error(pred,grou,squared=True))
    score_path = os.path.join(ou.stat_dir, ou.score_file)
    pd.DataFrame(s).to_csv(score_path)

    col_sorted_by_importance=model.feature_importances_.argsort()

    feat_imp=pd.DataFrame({
        'cols':X_val.columns[col_sorted_by_importance],
        'imps':model.feature_importances_[col_sorted_by_importance]
    })

    fig = px.bar(feat_imp.sort_values(['imps'], ascending=False)[:15],
        x = 'cols', y = 'imps', labels = {'cols':' ', 'imps':'feature importance'})
    fig.show()
    feat_imp_path = os.path.join(ou.stat_dir, ou.feat_file)
    fig.write_image(feat_imp_path)

def rf_infer(args):
    """ 
    생육 환경 데이터 추론
    """
    def clip(input,ranges):
        # clip 함수
        for col in range(len(input.columns)):
            for row in range(len(input)):
                if input.iloc[row,col] <= ranges[col][0]:
                    input.iloc[row,col] = ranges[col][0]
                elif input.iloc[row,col] >= ranges[col][1]:
                    input.iloc[row,col] = ranges[col][1]
                else:
                    input.iloc[row,col]
        return input

    # 파일로부터 설정 불러옴
    print("Loading settings...")
    tr = args.settings.train 
    dl = args.settings.dataloader
    inn = args.input
    ou = args.output
    pl = args.settings.plot_settings
    os.makedirs(ou.model_dir, exist_ok=True)
    os.makedirs(ou.stat_dir, exist_ok=True)
    os.makedirs(ou.pred_dir, exist_ok=True)
    os.makedirs(pl.plot_dir, exist_ok=True)

    # 데이터 읽어오기
    print("Setting dataloader...")
    ds = get_dataloader(args)


    # 테스트 데이터 설정
    print("Setting test data feature/labels...")
    X_test = ds
    y_test = X_test[dl.grw_target + dl.diff_target].copy()
    X_test.drop(dl.grw_target,axis=1,inplace=True)

    # 테스트 데이터 정제
    print("Normalizing Data...")
    test_stats = X_test.describe()
    test_stats = test_stats.transpose()
    X_test = (X_test - test_stats['mean']) / test_stats['std']
    X_test = X_test.dropna()
    config = [len(X_test.keys()),len(y_test.keys())]

    # 모델 불러오기
    print("Loading model file...")
    model = get_model(args, config)
    model = load(inn.model_dir + inn.model_filename)
    print(model)

    # 데이터 예측
    print("Printing prediction results...")
    y_pred = pd.DataFrame(model.predict(X_test))
    y_pred.to_csv(os.path.join(ou.pred_dir + ou.pred_file))

    # 데이터 점수 평가
    score = ou.stat_dir + ou.score_file
    s = []
    for x in range(len(y_pred.columns)):
        pred = y_pred.iloc[:,x]
        grou = y_test.iloc[:,x]
        s.append(mean_squared_error(pred, grou,squared=True))
    pd.DataFrame(s).to_csv(score)

    # 성장 예측표 그림
    print("Printing Growth Rate")
    variables = pl.axis_values
    ranges = pl.axis_ranges

    for x in range(pl.plot_num):
        y_pred = clip(y_pred, ranges)
        y_test = clip(y_test,ranges)

        fig1 = plt.figure(figsize=(6, 6))
        radar = ComplexRadar(fig1, variables, ranges)
        radar.plot(y_pred.iloc[x],'r')
        radar.plot(y_test.iloc[x],'b')
        radar.plot(y_pred.iloc[x],color='r',marker='o')
        plt.savefig(pl.plot_dir + "pred" + str(x) + ".png")

    print("Print Complete!")
    return 0
        

def multi_encoder_train(args):
    """ 
    multi encoder를 사용해서 Product 훈련 및 검증 
    """

    def train_step(inp, targ, model, optimizer):
        """ 
        훈련 과정 1 step; gradient계산 및 적용
        """
        with tf.GradientTape() as tape:
            out = model(inp, training=True)
            loss = keras.losses.mean_squared_error(targ, out)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, out

    def inference(inp, targ, model):
        """ 
        추론 함수
        """
        test_loss = keras.metrics.Mean(name="test_loss")
        out = model(inp, training=False)
        loss = keras.losses.mean_squared_error(targ, out)
        test_loss(loss)
        template = 'Test Loss: {}'
        print(template.format(test_loss.result()))
        return out

    # 파일로부터 설정 가져오기
    print("Loading json file...")
    ou = args.output
    tr = args.settings.train
    os.makedirs(ou.stat_dir,exist_ok=True)
    os.makedirs(ou.pred_dir,exist_ok=True)
    os.makedirs(ou.heatmap_dir,exist_ok=True)
    os.makedirs(ou.checkpoint_dir,exist_ok=True)
    

    # get data loader - [[data], [label]]
    print("Getting dataloader...")
    ds, input_shapes, output_shape = get_dataloader(args)
    print("Input and output shape: ")
    print(input_shapes, output_shape)
    print("=" * 20)
    train_ds = ds[: -3]
    test_ds = ds[-2 : -1]
    # input parameters for lstm_inc_dec model
    args.settings.dataloader.input_shapes = input_shapes
    args.settings.dataloader.output_shape = output_shape

    """ 
    디바이스 설정 및 훈련 시작
    """
    print("Select device... "+args.settings.device.name)
    with tf.device(args.settings.device.name):
        # 모델 불러오기
        print("Setting up Model and optimizer...")
        model = get_model(args)
        optimizer = get_optimizer(args)
        train_loss = keras.metrics.Mean(name="train_loss")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        # 훈련 시작
        print("Training start...")
        for epoch in range(tr.total_epochs):
            train_loss.reset_states()
            for X_train, y_train in train_ds:
                loss, out = train_step(X_train, y_train, model, optimizer)
                train_loss(loss)
            
            # 10 Epoch마다 loss 표기
            if epoch % 10 == 0:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1, train_loss.result()))

            # 100 Epoch마다 checkpoint 저장 및 테스트 확인
            if epoch % 100 == 0 or epoch == (tr.total_epochs - 1):
                print("Epoch {}: Saving Checkpoint...".format(epoch))
                checkpoint.save(file_prefix= ou.checkpoint_dir)

                # validation 검증
                print("Validating training...")
                X_val , y_val = test_ds[-1]
                y_pred = inference(X_val, y_val, model)
                
                print("Saving tensor to file ...")
                tensor2csv(ou.pred_dir + ou.pred_file , y_pred)
                tensor2csv(ou.pred_dir + ou.ground_file, y_val)
                draw_harvest_per_sample(y_pred, 
                                        y_val, ou.pred_dir + ou.harvest_file)
        
        # Drawing heatmap and bar chart for explanation
        if args.settings.dataloader.explain:
            if args.settings.dataloader.env_only:
                if args.settings.env_heatmap.avail:
                    e = model.explain(test_ds, return_heatmap=True)
                    x_labels = args.settings.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.input.seek_days, 0, -1)]
                    
                    heatmap_path = os.path.join(ou.heatmap_dir, args.settings.env_heatmap.name)
                    draw_heatmap(heatmap=e, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
            else:
                e, g, _, _ = model.explain(test_ds, return_heatmap=True)
                if args.settings.env_heatmap.avail:
                    x_labels = args.settings.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.input.seek_days, 0, -1)]
                    heatmap_path = os.path.join(ou.heatmap_dir, args.settings.env_heatmap.name)
                    draw_heatmap(heatmap=e, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
                    bar_name = "bar_" + args.settings.env_heatmap.name
                    bar_path = os.path.join(ou.heatmap_dir, bar_name)
                    draw_bargraph(data=e, filename=bar_path, x_labels=x_labels)
                
                if args.settings.growth_heatmap.avail:
                    x_labels = args.settings.growth_heatmap.x_labels
                    y_labels = [str(i) for i in range(1, args.input.num_samples+1)]
                    heatmap_path = os.path.join(ou.heatmap_dir, args.settings.growth_heatmap.name)
                    draw_heatmap(heatmap=g, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
                    bar_name = "bar_" + args.settings.growth_heatmap.name
                    bar_path = os.path.join(ou.heatmap_dir, bar_name)
                    draw_bargraph(data=g, filename=bar_path, x_labels=x_labels)

def multi_encoder_infer(args):
    """ 
    Product 데이터 추론
    """
    ou = args.output
    inn = args.input
    os.makedirs(ou.pred_dir ,exist_ok=True)
    os.makedirs(ou.heatmap_dir, exist_ok=True)
    # get data loader - [[data], [label]]
    print("Setting Dataloader...")   
    ds, input_shapes, output_shape = get_dataloader(args)
    # input parameters for lstm_inc_dec model
    args.settings.dataloader.input_shapes = input_shapes
    args.settings.dataloader.output_shape = output_shape
    
    # 모델 설정
    print("Loading Model...")
    with tf.device(args.settings.device.name):
        # model create
        model = get_model(args)
        # optimizer create
        optimizer = get_optimizer(args)
        # load weights from file
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint.restore(tf.train.latest_checkpoint(inn.checkpoint_dir)).expect_partial()

        # 추론 진행 
        print("Starting Inference...")
        for X_test, y_test in ds:
            y_pred = model(X_test, training=False)
            tensor2csv(ou.pred_dir + ou.pred_file , y_pred)
            tensor2csv(ou.pred_dir + ou.ground_file, y_test)
            draw_harvest_per_sample(y_pred, y_test, ou.pred_dir + ou.harvest_file)
        
        # 설명 그래프 그리기
        print("Drawing heatmap and bar chart for explanation...")
        if args.settings.dataloader.explain:
            if args.settings.dataloader.env_only:
                if args.settings.env_heatmap.avail:
                    e = model.explain(ds, return_heatmap=True)
                    x_labels = args.settings.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.input.seek_days, 0, -1)]
                    heatmap_path = os.path.join(ou.heatmap_dir, args.settings.env_heatmap.name)
                    draw_heatmap(heatmap=e, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
            else:
                e, g, _, _ = model.explain(ds, return_heatmap=True)
                if args.settings.env_heatmap.avail:
                    x_labels = args.settings.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.input.seek_days, 0, -1)]
                    heatmap_path = os.path.join(ou.heatmap_dir, args.settings.env_heatmap.name)
                    draw_heatmap(heatmap=e, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
                    bar_name = "bar_" + args.settings.env_heatmap.name
                    bar_path = os.path.join(ou.heatmap_dir, bar_name)
                    draw_bargraph(data=e, filename=bar_path, x_labels=x_labels)
                
                if args.settings.growth_heatmap.avail:
                    x_labels = args.settings.growth_heatmap.x_labels
                    y_labels = [str(i) for i in range(1, args.input.num_samples+1)]
                    heatmap_path = os.path.join(ou.heatmap_dir, args.settings.growth_heatmap.name)
                    draw_bargraph(data=g, filename=heatmap_path, x_labels=x_labels)
                    bar_name = "bar_" + args.settings.growth_heatmap.name
                    bar_path = os.path.join(ou.heatmap_dir, bar_name)
                    draw_bargraph(data=g, filename=bar_path, x_labels=x_labels)

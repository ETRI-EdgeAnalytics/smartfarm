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
        if self.args.predict == 'multi_encoder_train':
            multi_encoder_train(self.args)
        # jypark
        elif self.args.predict == 'rf_train':
            rf_train(self.args)
        else:
            raise ValueError("Wrong Predict Argument for Runner.train")
    
    def infer(self):
        # ihshin
        if self.args.predict == 'multi_encoder_infer':
            multi_encoder_infer(self.args)
        # jypark
        elif self.args.predict == 'rf_infer':
            rf_infer(self.args)
        else:
            raise ValueError("Wrong Predict Argument for Runner.infer")

    def run(self):
        if 'train' in self.args.predict:
            self.train()
        elif 'infer' in self.args.predict:
            self.infer()
        else:
            raise ValueError("Wrong Predict Argument for Runner.run")

def rf_train(args):
    print('rf_train')
 
    ds = get_dataloader(args)
    exp_dir = args.experiment

    print("mlp growth train")
    # First Dataset Training
    train_dataset = ds.sample(frac=0.8,random_state=0)
    test_dataset  = ds.drop(train_dataset.index)
    
    train_labels = train_dataset[['d줄기굵기(mm)','d잎길이(cm)','d잎폭(cm)','주간생육길이(cm)','dleaf_area']].copy()
    test_labels = test_dataset[['d줄기굵기(mm)','d잎길이(cm)','d잎폭(cm)','주간생육길이(cm)','dleaf_area']].copy()

    train_dataset.drop(['d줄기굵기(mm)','d잎길이(cm)','d잎폭(cm)','주간생육길이(cm)','dleaf_area'],axis=1,inplace=True)
    test_dataset.drop(['d줄기굵기(mm)','d잎길이(cm)','d잎폭(cm)','주간생육길이(cm)','dleaf_area'],axis=1,inplace=True)

    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    test_stats = test_dataset.describe()
    test_stats = test_stats.transpose()
    
    normed_train_data = (train_dataset - train_stats['mean']) / train_stats['std']
    normed_test_data = (test_dataset - test_stats['mean']) / test_stats['std']
    
    X_train = normed_train_data
    y_train = train_labels

    print(X_train.columns)

    config = [len(train_dataset.keys()),len(train_labels.keys())]
    model = get_model(args,config)

    model.fit(X_train.values, y_train.values)
    filename = args.model_file
    dump(model, exp_dir+filename)
    test_dataset.to_csv(exp_dir+args.test_data.features)
    test_labels.to_csv(exp_dir+args.test_data.labels)
    train_stats.to_csv(exp_dir+args.util.train_stats)

    print(normed_train_data.shape)
    print(train_labels.shape)
    print(normed_test_data.shape)
    print(test_labels.shape)

    rf_infer(args)

def rf_infer(args):

    def clip(input,ranges):
        for col in range(len(input.columns)):
            for row in range(len(input)):
                if input.iloc[row,col] <= ranges[col][0]:
                    input.iloc[row,col] = ranges[col][0]
                elif input.iloc[row,col] >= ranges[col][1]:
                    input.iloc[row,col] = ranges[col][1]
                else:
                    input.iloc[row,col]
        return input

    print('growth_infer')
    ds = get_dataloader(args)
    exp_dir = args.util.path
    filename = args.model_file

    test_dataset = pd.read_csv(exp_dir+args.test_data.features,index_col=0)

    test_stats = test_dataset.describe()
    test_stats = test_stats.transpose()

    normed_test_data = (test_dataset - test_stats['mean']) / test_stats['std']
    normed_test_data = normed_test_data.dropna()
    X_test = normed_test_data
    test_labels = pd.read_csv(exp_dir + args.test_data.labels,index_col=0)
    config = [len(test_dataset.keys()),len(test_labels.keys())]
    
    model = get_model(args,config)
    model = load(exp_dir+filename)

    y_pred = pd.DataFrame(model.predict(normed_test_data))
    y_test = pd.DataFrame(test_labels)

    pred = args.output.prediction
    y_pred.to_csv(exp_dir+pred)        

    score = args.output.score
    fi = args.output.feature_importance
    s = []
    for x in range(len(y_pred.columns)):
        pred = y_pred.iloc[:,x]
        grou = y_test.iloc[:,x]
        
        s.append(mean_squared_error(pred,grou,squared=True))
    pd.DataFrame(s).to_csv(exp_dir+score)

    col_sorted_by_importance=model.feature_importances_.argsort()

    feat_imp=pd.DataFrame({
        'cols':X_test.columns[col_sorted_by_importance],
        'imps':model.feature_importances_[col_sorted_by_importance]
    })

    fig = px.bar(feat_imp.sort_values(['imps'], ascending=False)[:15],
        x='cols', y='imps', labels={'cols':' ', 'imps':'feature importance'})
    fig.show()
    fig.write_image(exp_dir+fi)


    variables = ( 'Thickness(mm)', 'Leaf Length(cm)', 'Leaf Width(cm)','Grown_height(cm)', 'Leaf Area')
    ranges = [(0, 15), (5, 30), (5, 35), (0, 55), (0.0, 2.5)]             

    for x in range(len(y_pred)):
        y_pred= clip(y_pred,ranges)
        test_labels = clip(test_labels,ranges)

        fig1 = plt.figure(figsize=(6, 6))
        radar = ComplexRadar(fig1, variables, ranges)
        radar.plot(y_pred.iloc[x],'r')
        radar.plot(test_labels.iloc[x],'b')
        radar.plot(y_pred.iloc[x],color='r',marker='o')
        plt.savefig(exp_dir+"/growth_images/pred"+str(x)+".png")

        

def multi_encoder_train(args):
    #@tf.function
    def train_step(inp, targ, model, optimizer):

        with tf.GradientTape() as tape:
            out = model(inp, training=True)
            loss = keras.losses.mean_squared_error(targ, out)
        gradients =tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss, out

    def inference(inp, targ, model):
        test_loss = keras.metrics.Mean(name="test_loss")
        out = model(inp, training=False)
        loss = keras.losses.mean_squared_error(targ, out)
        test_loss(loss)

        template = 'Test Loss: {}'
        print(template.format(test_loss.result()))

        return out

    # get data loader - [[data], [label]]
    ds, input_shapes, output_shape = get_dataloader(args)
    print("input and output shape")
    print(input_shapes, output_shape)
    print("="*20)
    train_ds = ds[:-3]
    test_ds = ds[-2:-1]
    # input parameters for lstm_inc_dec model
    args.model.config.input_shapes = input_shapes
    args.model.config.output_shape = output_shape

    save_path = args.util.save_path
    
    # Select device
    with tf.device(args.device.name):
        # model create
        model = get_model(args)
        # optimizer create
        optimizer = get_optimizer(args)
        # loss metric setting
        train_loss = keras.metrics.Mean(name="train_loss")
        # model save setting
        checkpoint_dir = os.path.join(save_path, 'ckpt/ckpt')
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        # Set epochs
        EPOCHS = args.train.epochs

        # Start training
        for epoch in range(EPOCHS):
            train_loss.reset_states()

            for data, targets in train_ds:
                loss, out = train_step(data, targets, model, optimizer)

                train_loss(loss)
            
            if epoch % 10 == 0:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch+1, train_loss.result()))
            
            if epoch % 100 == 0 or epoch == (EPOCHS - 1):
                checkpoint.save(file_prefix=checkpoint_dir)
                test_data, test_targets = test_ds[-1]
                out = inference(test_data, test_targets, model)
                output_path = os.path.join(save_path, 'output.csv')
                groundtruth_path = os.path.join(save_path, 'groundtruth.csv')
                tensor2csv(output_path, out)
                tensor2csv(groundtruth_path, test_targets)
                harvest_path = os.path.join(save_path, 'harvest.png')
                draw_harvest_per_sample(out, test_targets, harvest_path)
        
        # Drawing heatmap and bar chart for explanation
        if args.model.config.explain:
            if args.model.config.env_only:
                if args.util.env_heatmap.avail:
                    e = model.explain(test_ds, return_heatmap=True)
                    x_labels = args.util.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.data.seek_days, 0, -1)]
                    heatmap_path = os.path.join(save_path, args.util.env_heatmap.name)
                    draw_heatmap(heatmap=e, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
            else:
                e, g, _, _ = model.explain(test_ds, return_heatmap=True)
                if args.util.env_heatmap.avail:
                    x_labels = args.util.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.data.seek_days, 0, -1)]
                    heatmap_path = os.path.join(save_path, args.util.env_heatmap.name)
                    draw_heatmap(heatmap=e, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
                    bar_name = "bar_" + args.util.env_heatmap.name
                    bar_path = os.path.join(save_path, bar_name)
                    draw_bargraph(data=e, filename=bar_path, x_labels=x_labels)
                
                if args.util.growth_heatmap.avail:
                    x_labels = args.util.growth_heatmap.x_labels
                    y_labels = [str(i) for i in range(1, args.data.num_samples+1)]
                    heatmap_path = os.path.join(save_path, args.util.growth_heatmap.name)
                    draw_heatmap(heatmap=g, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
                    bar_name = "bar_" + args.util.growth_heatmap.name
                    bar_path = os.path.join(save_path, bar_name)
                    draw_bargraph(data=g, filename=bar_path, x_labels=x_labels)

def multi_encoder_infer(args):
    # get data loader - [[data], [label]]
    ds, input_shapes, output_shape = get_dataloader(args)
    # input parameters for lstm_inc_dec model
    args.model.config.input_shapes = input_shapes
    args.model.config.output_shape = output_shape

    save_path = args.util.save_path
    pretrained_path = args.model.pretrained_path
    
    # Select device
    with tf.device(args.device.name):
        # model create
        model = get_model(args)
        # optimizer create
        optimizer = get_optimizer(args)
        # load weights from file
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint.restore(tf.train.latest_checkpoint(pretrained_path)).expect_partial()

        for data, targets in ds:
            out = model(data, training=False)
            output_path = os.path.join(save_path, 'output.csv')
            groundtruth_path = os.path.join(save_path, 'groundtruth.csv')
            tensor2csv(output_path, out)
            tensor2csv(groundtruth_path, targets)
            harvest_path = os.path.join(save_path, 'harvest.png')
            draw_harvest_per_sample(out, targets, harvest_path)

        # Drawing heatmap and bar chart for explanation
        if args.model.config.explain:
            if args.model.config.env_only:
                if args.util.env_heatmap.avail:
                    e = model.explain(ds, return_heatmap=True)
                    x_labels = args.util.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.data.seek_days, 0, -1)]
                    heatmap_path = os.path.join(save_path, args.util.env_heatmap.name)
                    draw_heatmap(heatmap=e, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
            else:
                e, g, _, _ = model.explain(ds, return_heatmap=True)
                if args.util.env_heatmap.avail:
                    x_labels = args.util.env_heatmap.x_labels
                    y_labels = [str(i) for i in range(args.data.seek_days, 0, -1)]
                    heatmap_path = os.path.join(save_path, args.util.env_heatmap.name)
                    draw_heatmap(heatmap=e, filename=heatmap_path, x_labels=x_labels, y_labels=y_labels)
                    bar_name = "bar_" + args.util.env_heatmap.name
                    bar_path = os.path.join(save_path, bar_name)
                    draw_bargraph(data=e, filename=bar_path, x_labels=x_labels)
                
                if args.util.growth_heatmap.avail:
                    x_labels = args.util.growth_heatmap.x_labels
                    y_labels = [str(i) for i in range(1, args.data.num_samples+1)]
                    heatmap_path = os.path.join(save_path, args.util.growth_heatmap.name)
                    draw_bargraph(data=g, filename=heatmap_path, x_labels=x_labels)
                    bar_name = "bar_" + args.util.growth_heatmap.name
                    bar_path = os.path.join(save_path, bar_name)
                    draw_bargraph(data=g, filename=bar_path, x_labels=x_labels)

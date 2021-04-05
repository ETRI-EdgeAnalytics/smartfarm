import os

import pandas as pd
import numpy as np
import tensorflow as tf

from .datautils import df2numpy

__all__ = ['dataloader4lstm_enc_dec']

def dataloader4lstm_enc_dec(args):
    train_data_list = []
    train_names = args.input.feature_data.names
    path = args.input.feature_data.path

    df_list = make_data_frame(train_names, path, label=False, seek_days=args.input.seek_days)

    for df in df_list:

        if '초장(cm)' in df.columns:
            data = df2numpy(df, args.input.num_samples, offset=args.input.num_samples, dropkey=['날짜'])
        elif '샘플번호' in df.columns:
            data = df2numpy(df, args.input.num_samples, offset=args.input.num_samples, dropkey=['날짜', '샘플번호'])
        else:
            data = df2numpy(df, args.input.seek_days, offset=7, dropkey=['날짜'])
        
        tf_data = tf.convert_to_tensor(data, dtype=tf.float32)
        train_data_list.append(tf_data)

    label_data_list = []
    label_names = args.input.label_data.names
    path = args.input.label_data.path

    df_list = make_data_frame(label_names, path, label=True)

    for df in df_list:
        
        data = df2numpy(df, args.input.num_samples, offset=args.input.num_samples, dropkey=['날짜', '샘플번호'])

        tf_data = tf.convert_to_tensor(data, dtype=tf.float32)
        label_data_list.append(data)
    
    ds = []
    env, growth, pro1, pro2 = train_data_list
    pro3, pro4 = label_data_list

    len_list = [len(env), len(growth), len(pro1), len(pro2), len(pro3), len(pro4)]
    len_list.sort()
    num_data = len_list[0]

    env = env[:num_data]
    growth = growth[:num_data]
    pro1 = pro1[:num_data]
    pro2 = pro2[:num_data]
    pro3 = pro3[:num_data]
    pro4 = pro4[:num_data]

    input_shape = [env.shape, growth.shape, pro1.shape, pro2.shape]
    
    if args.settings.dataloader.avg:
        avg_label_data_list = tf.math.divide_no_nan(pro4, pro3)
        for e, g, p1, p2, a in zip(env, growth, pro1, pro2, avg_label_data_list):
            ds.append([
                [tf.reshape(e, [1, e.shape[0], e.shape[1]]), tf.reshape(g, [1, g.shape[0], g.shape[1]]),
                tf.reshape(p1, [1, p1.shape[0], p1.shape[1]]), tf.reshape(p2, [1, p2.shape[0], p2.shape[1]])],
                tf.reshape(a, [1, a.shape[0], a.shape[1], 1])
            ])
        output_shape = [avg_label_data_list.shape[0], avg_label_data_list.shape[1], avg_label_data_list.shape[2], 1]
    else:
        for e, g, p1, p2, p3, p4 in zip(env, growth, pro1, pro2, pro3, pro4):
            p3 = tf.cast(p3, tf.float32)
            p4 = tf.cast(p4, tf.float32)
            ds.append([
                [tf.reshape(e, [1, e.shape[0], e.shape[1]]), tf.reshape(g, [1, g.shape[0], g.shape[1]]), 
                tf.reshape(p1, [1, p1.shape[0], p1.shape[1]]), tf.reshape(p2, [1, p2.shape[0], p2.shape[1]])], 
                tf.concat([tf.reshape(p3, [1, p3.shape[0], p3.shape[1], 1]), tf.reshape(p4, [1, p4.shape[0], p4.shape[1], 1])], axis=3)
            ])
        output_shape = [pro3.shape[0], pro3.shape[1], pro3.shape[2], 2]
    
    return ds, input_shape, output_shape

def dataloader4lstm_enc_dec_env(args):
    train_data_list = []
    label_data_list = []
    df_dict = dict()
    
    train_names = args.input.feature_data.names
    path = args.input.feature_data.path

    file_path = os.path.join(path, train_names)
    df = pd.ExcelFile(file_path)
    sheet_df = pd.read_excel(df, 'Sheet1')
    df_dict['env'] = sheet_df

    label_names = args.input.label_data.names
    for name in label_names:
        path = args.input.label_data.path
        file_path = os.path.join(path, name)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')

        if "product_3" in name:
            df_dict['product_3'] = sheet_df
        elif "product_4" in name:
            df_dict['product_4'] = sheet_df
        else:
            raise ValueError(name)
    
    train_df, label_dfs = make_data_frame_env(df_dict=df_dict, seek_days=args.input.seek_days, interval=args.input.interval)

    data = df2numpy(train_df, args.input.seek_days, offset=7, dropkey=['날짜'])
    
    tf_data = tf.convert_to_tensor(data, dtype=tf.float32)
    train_data_list.append(tf_data)

    label_names = args.input.label_data.names
    
    for df in label_dfs:
        data = df2numpy(df, args.input.num_samples, offset=args.input.num_samples, dropkey=['날짜', '샘플번호'])

        tf_data = tf.convert_to_tensor(data, dtype=tf.float32)
        label_data_list.append(data)
    
    ds = []
    env = train_data_list[0]
    pro3, pro4 = label_data_list

    len_list = [len(env), len(pro3), len(pro4)]
    len_list.sort()
    num_data = len_list[0]

    env = env[:num_data]
    pro3 = pro3[:num_data]
    pro4 = pro4[:num_data]

    input_shape = [env.shape]

    if args.settings.dataloader.avg:
        avg_label_data_list = tf.math.divide_no_nan(pro4, pro3)

        for e, a in zip(env, avg_label_data_list):
            ds.append([
                [tf.reshape(e, [1, e.shape[0], e.shape[1]])],
                tf.reshape(a, [1, a.shape[0], a.shape[1], 1])
            ])
        output_shape = [avg_label_data_list.shape[0], avg_label_data_list.shape[1], avg_label_data_list.shape[2], 1]
    else:
        for e, p3, p4 in zip(env, pro3, pro4):
            p3 = tf.cast(p3, tf.float32)
            p4 = tf.cast(p4, tf.float32)
            ds.append([
                [tf.reshape(e, [1, e.shape[0], e.shape[1]])],
                tf.concat([tf.reshape(p3, [1, p3.shape[0], p3.shape[1], 1]), tf.reshape(p4, [1, p4.shape[0], p4.shape[1], 1])], axis=3)
            ])
        output_shape = [pro3.shape[0], pro3.shape[1], pro3.shape[2], 2]
    
    return ds, input_shape, output_shape

def make_data_frame(file_names, path, label=False, seek_days=43):

    data = dict()
    for name in file_names:
        file_path = os.path.join(path, name)
        df = pd.ExcelFile(file_path)
        sheet_df = pd.read_excel(df, 'Sheet1')
        
        if "env" in name:
            data["env"] = sheet_df
        elif "growth" in name:
            data["growth"] = sheet_df
        elif "product_1" in name:
            data["product_1"] = sheet_df
        elif "product_2" in name:
            data["product_2"] = sheet_df
        elif "product_3" in name:
            data["product_3"] = sheet_df
        elif "product_4" in name:
            data["product_4"] = sheet_df
        else:
            raise ValueError(name)
        
    if not label:
        standard_index = 0
        date = data["product_1"].iloc[standard_index]["날짜"]
        env_index = data["env"].index[data["env"]["날짜"]==date].tolist()[0]
        num_samples = len(data["product_1"]) // len(data["product_1"]["날짜"].unique())

        while env_index < seek_days:
            standard_index += num_samples
            date = data["product_1"].iloc[standard_index]["날짜"]
            env_index = data["env"].index[data["env"]["날짜"]==date].tolist()[0]
        
        new_env_df = data["env"].iloc[env_index - seek_days + 1:]
        g_index = data["growth"].index[data["growth"]["날짜"]==date].tolist()[0]
        new_growth_df = data["growth"].iloc[g_index:]
        new_product_1_df = data["product_1"].iloc[standard_index:]
        new_product_2_df = data["product_2"].iloc[standard_index:]

        df_list = [new_env_df, new_growth_df, new_product_1_df, new_product_2_df]

    else:
        df_list = [data["product_3"], data["product_4"]]
    
    return df_list

def make_data_frame_env(df_dict, seek_days=43, interval=7):

    standard_index = 0
    date = df_dict["product_3"].iloc[standard_index]["날짜"]
    env_index = df_dict["env"].index[df_dict["env"]["날짜"]==date].tolist()[0]
    env_index = env_index - interval
    num_samples = len(df_dict["product_3"]) // len(df_dict["product_3"]["날짜"].unique())

    while env_index < seek_days:
        standard_index += num_samples
        date = df_dict["product_3"].iloc[standard_index]["날짜"]
        env_index = df_dict["env"].index[df_dict["env"]["날짜"]==date].tolist()[0]
    
    new_env_df = df_dict["env"].iloc[env_index - seek_days + 1:]
    new_product_3_df = df_dict["product_3"].iloc[standard_index:]
    new_product_4_df = df_dict["product_4"].iloc[standard_index:]

    return new_env_df, [new_product_3_df, new_product_4_df]

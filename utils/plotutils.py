import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
import plotly.graph_objects as go


__all__ = ['draw_heatmap', 'draw_bargraph', 'draw_harvest_per_sample']

'''
plot harvest prediction
'''
def draw_heatmap(heatmap, filename, x_labels, y_labels):
    fig = go.Figure(data=go.Heatmap(z=heatmap, x=x_labels, y=y_labels, hoverongaps=False))
    fig.write_image(filename)

def draw_bargraph(data, filename, x_labels=None):
    avg_data = np.mean(data, axis=0)
    df = pd.DataFrame(avg_data).T
    df.columns = x_labels
    df.sort_values(by=0, ascending=False, axis=1, inplace=True)

    drop_columns = []
    for c in df.columns:
        if not df[c][0] > 0:
            drop_columns.append(c)
    df = df.drop(columns=drop_columns)

    fig = go.Figure(data=[go.Bar(x=df.columns, y=df.values[0])])
    fig.update_traces(textposition='outside')
    fig.write_image(filename)

def autolabel(ax, rects, xpos='center'):
    ha = {'center':'center', 'right':'left', 'left':'right'}
    offset = {'center':0, 'right':1, 'left':-1}

    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate('{0:.2f}'.format(height), 
                        xy=(rect.get_x()+rect.get_width()/2, height),
                        xytext=(offset[xpos]*3, 2),
                        textcoords="offset points",
                        ha=ha[xpos], va='bottom')

def draw_harvest_per_sample(out, gt, filename):
    b, h, w, c = out.shape
    
    h, w = int(h), int(w)
    width = 0.35
    index = np.arange(w)
    
    cols = 4
    rows = int(np.ceil(h / cols))
    if c == 1:
        fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(16, 16))
        out = tf.reshape(out, [h, w])
        gt = tf.reshape(gt, [h, w])
        
        r_index, c_index, s_index = 0, 0, 1
        for o, g in zip(out, gt):
            rects1 = axes[r_index][c_index].bar(index - width/2, o, width, label='Pred')
            rects2 = axes[r_index][c_index].bar(index + width/2, g, width, label='Truth')
            axes[r_index][c_index].set_title("Sample {}".format(s_index))
            axes[r_index][c_index].legend()
            autolabel(axes[r_index][c_index], rects1, "left")
            autolabel(axes[r_index][c_index], rects2, "right")
            
            c_index += 1
            s_index += 1
            
            if c_index == cols:
                r_index += 1
                c_index = 0
        
        d_index = r_index * cols + c_index
        
        for i in range(d_index, rows * cols):
            fig.delaxes(axes.flatten()[i])
            
        fig.tight_layout()
        plt.savefig(filename)
    else:
        # fn = filename.split('.')[0]
        fn = filename
        postfix = ['_num.png', '_kg.png']

        for i, post in enumerate(postfix):
            fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(16, 16))
            out_split = tf.reshape(out[:, :, :, i], [h, w])
            gt_split = tf.reshape(gt[:, :, :, i], [h, w])
            
            r_index, c_index, s_index = 0, 0, 1
            for o, g in zip(out_split, gt_split):
                rects1 = axes[r_index][c_index].bar(index - width/2, o, width, label='Pred')
                rects2 = axes[r_index][c_index].bar(index + width/2, g, width, label='Truth')
                axes[r_index][c_index].set_title("Sample {}".format(s_index))
                axes[r_index][c_index].legend()
                autolabel(axes[r_index][c_index], rects1, "left")
                autolabel(axes[r_index][c_index], rects2, "right")
                
                c_index += 1
                s_index += 1
                
                if c_index == cols:
                    r_index += 1
                    c_index = 0
            
            d_index = r_index * cols + c_index
            
            for i in range(d_index, rows * cols):
                fig.delaxes(axes.flatten()[i])
                
            fig.tight_layout()
            plt.savefig(fn+post)
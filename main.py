from cProfile import label
from email.mime import base
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def open_file(file_name: str) -> pd.DataFrame:
    data = []
    with open(file_name) as file_in:
        for line in file_in:
            data.append(line.replace('\n', '').split(' '))
    df = pd.DataFrame(data, columns=['time', 'loss', 'step'])

    df = df.astype({'loss': 'float', 'step': 'int32'})
    df['loss'] = df['loss'][df['loss'] < df['loss'].mean() + 10 * df['loss'].std()] 

    return  df

def load_from_csv(file_name: str) ->  pd.DataFrame:
    return pd.read_csv(file_name)

def getStep(file):
    return file['step'].iloc[-1]

def plot_fitted_line(df: pd.DataFrame):
    plt.plot(df.step, df.iloc[:,1:-1], alpha = 0.35)
    plt.plot(df.step, df['mean'], c='r', label='mean')
    fitted = np.polyfit(df.step, df['mean'], 4)
    linear_model_fn=np.poly1d(fitted)
    plt.plot(df.step, linear_model_fn(df.step), c='k')
    # plt.fill_between(df.step, df['mean'] - df['stderr'], df['mean'] + df['stderr'], alpha = 0.35)
    plt.legend()
    plt.show()


def plot_all_data(df: pd.DataFrame):
    plt.plot(df.step, df.iloc[:,1:-1], alpha = 0.35)
    plt.plot(df.step, df['mean'], c='r', label='mean')
    plt.fill_between(df.step, df['mean'] - df['stderr'], df['mean'] + df['stderr'], alpha = 0.35)
    plt.legend()
    plt.show()

def compare_two_runs(df1: pd.DataFrame, df2: pd.DataFrame, title: str):
    df1_size = getStep(df1)
    df2_size = getStep(df2)
    difference = df1_size - df2_size
    if difference < 0:
        df2 = df2[df2['step'] <= df1_size]
    elif difference > 0:
        df1 = df1[df1['step'] <= df2_size]

    plt.plot(df1.step, df1['mean'], c='b', label='dreamer')
    plt.fill_between(df1.step, df1['mean'] - df1['stderr'], df1['mean'] + df1['stderr'], alpha = 0.35)
    fitted = np.polyfit(df1.step, df1['mean'], 4)
    linear_model_fn=np.poly1d(fitted)
    plt.plot(df1.step, linear_model_fn(df1.step), c='k')

    plt.plot(df2.step, df2['mean'], c='r', label='inst')
    plt.fill_between(df2.step, df2['mean'] - df2['stderr'], df2['mean'] + df2['stderr'], alpha = 0.35)
    fitted = np.polyfit(df2.step, df2['mean'], 4)
    linear_model_fn=np.poly1d(fitted)
    plt.plot(df2.step, linear_model_fn(df2.step), c='k')
    plt.title(title)
    plt.legend()
    plt.show()  

def compare_runs(dfs: List[pd.DataFrame], title:str):
    sizes = []
    for df, name in dfs:
        sizes.append(getStep(df))
    
    smallest_step = min(sizes)

    for df, name in dfs:
        df = df[df['step'] <= smallest_step]
        plt.plot(df.step, df['mean'], label=name, alpha = 0.35)
        # plt.fill_between(df.step, df['mean'] - df['stderr'], df['mean'] + df['stderr'], alpha = 0.35)
        fitted = np.polyfit(df.step, df['mean'], 4)
        linear_model_fn=np.poly1d(fitted)
        plt.plot(df.step, linear_model_fn(df.step), label=name)
    plt.title(title)
    plt.legend()
    plt.show()  

def get_data(base_path: str, input_files: List[str], metrics: str) -> pd.DataFrame:
    do_stuff = True
    full_df = None
    complete_files = [base_path + name + metrics for name in input_files]
    for file_name in complete_files:
        if do_stuff:
            full_df = open_file(file_name).drop('time', axis=1)
            full_df = full_df[['step', 'loss']]
            do_stuff = False
        else: 
            full_df = pd.merge(full_df, open_file(file_name).drop('time', axis=1), how='outer', on='step')

    full_df = full_df.sort_values(by=['step'])

    return full_df

def calc_metrics(df: pd.DataFrame):
    df['mean'] = df.iloc[:,1:].mean(axis=1)
    df['stderr'] = df.iloc[:,1:-1].std(axis=1)
    

def save_data(df: pd.DataFrame, name: str):
    df.to_csv(f'data/{name}.csv')


if __name__ == '__main__':
    base_path = 'C:\\Users\\Jelle\\Documents\\GitHub\\pydreamer_jelle\\mlruns\\0\\'
    
    conv2d_instance_normd32_files = ['02898ba8445444bdb4846efaf4553bf9',
                                    'ff62d5a41a69456aa1999681b2696a54',
                                    '14d8fd0e2ee3498ea110afdc2dffb1ee',
                                    '220c93533e824196b4b5bb0a5b8bafda',
                                    '3e949ec53ace489fb8f62185e35a5d13',
                                    ]

    conv2d__noNorm_files= ['79e2e8a5383d4e0788dc1ac261cc3e5d',
                           'b8d7ebcb9a4d4af28b352c8a3a4bdb17',
                           '95bc6e5cd84443c9836c3955e8b56bec',
                           '3f0b9ed2e5794443bbee013f04bc39cf',
                           '90607139c2c64a9096f92206391972da',
                            ]

    metrics = ['\\metrics\\_loss', '\\metrics\\agent\\policy_value', '\\metrics\\agent\\return']

    metric = metrics[2]

    # df_instance = get_data(base_path, instance_norm_files, metric) 
    # calc_metrics(df_instance)

    # Run these three lines for saving something to a csv file
    # df_conv2d = get_data(base_path, conv2d_instance_normd32_files, metric) 
    # calc_metrics(df_conv2d)
    # save_data(df_conv2d, 'conv2d_inst_d32_return')
    
    df_dreamer          = load_from_csv('data/dreamer_normal.csv')
    df_mean_instance    = load_from_csv('data/cnn_mean_instance.csv')
    df_mean_noNorm      = load_from_csv('data/cnn_mean_only.csv')
    df_2dinstance       = [load_from_csv('data/conv2d_inst_d32_loss.csv'), load_from_csv('data/conv2d_inst_d32_pv.csv'), load_from_csv('data/conv2d_inst_d32_return.csv')]
    df_2dnoNorm         = [load_from_csv('data/conv2d_noNorm_loss.csv'), load_from_csv('data/conv2d_noNorm_pv.csv'), load_from_csv('data/conv2d_noNorm_return.csv')]
    df_3dinstance       = load_from_csv('data/3d_instancenorm.csv')

    # plot_fitted_line(df_dreamer)

    compare_two_runs(df_2dinstance[0], df_2dnoNorm[0], 'Loss')
    compare_two_runs(df_2dinstance[1], df_2dnoNorm[1], 'Policy Value')
    compare_two_runs(df_2dinstance[2], df_2dnoNorm[2], 'Return')

    # compare_runs([(df_dreamer, 'dreamer'), (df_mean_instance, 'instance'), (df_mean_noNorm, 'mean only'), (df_2dinstance, '2d instance')])

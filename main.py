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

def compare_two_runs(df1: pd.DataFrame, df2: pd.DataFrame):
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
    plt.legend()
    plt.show()  

def compare_runs(dfs: List[pd.DataFrame]):
    for df, name in dfs:
        plt.plot(df.step, df['mean'], label=name, alpha = 0.35)
        # plt.fill_between(df.step, df['mean'] - df['stderr'], df['mean'] + df['stderr'], alpha = 0.35)
        fitted = np.polyfit(df.step, df['mean'], 4)
        linear_model_fn=np.poly1d(fitted)
        plt.plot(df.step, linear_model_fn(df.step), label=name)
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

    instance_norm_files = ['02898ba8445444bdb4846efaf4553bf9',
                           'ff62d5a41a69456aa1999681b2696a54',
                           '14d8fd0e2ee3498ea110afdc2dffb1ee',
                           '220c93533e824196b4b5bb0a5b8bafda',
                           '3e949ec53ace489fb8f62185e35a5d13',
                            ]

    metrics = ['\\metrics\\_loss', '\\metrics\\agent\\policy_value', '\\metrics\\agent\\return']

    metric = metrics[2]

    df_dreamer = load_from_csv('data/dreamer_normal.csv')
    df_inst = load_from_csv('data/cnn_mean_instance.csv')
    df_mean = load_from_csv('data/cnn_mean_only.csv')
    df_2dinstance = load_from_csv('data/conv2d_instance_norm.csv')
    df_3dinstance = load_from_csv('data/3d_instancenorm.csv')

    save_data(df_dreamer, 'dreamer_normal')
    save_data(df_inst, 'cnn_mean_instance')
    save_data(df_mean, 'cnn_mean_only')
    save_data(df_2dinstance, 'conv2d_instance_norm')
    save_data(df_3dinstance, '3d_instancenorm')

    plot_fitted_line(df_dreamer)
    plot_fitted_line(df_inst)
    plot_fitted_line(df_mean)
    plot_fitted_line(df_2dinstance)
    plot_fitted_line(df_3dinstance)

    compare_two_runs(df_dreamer, df_2dinstance)
    compare_two_runs(df_dreamer, df_3dinstance)
    compare_two_runs(df_3dinstance, df_2dinstance)

    compare_runs([(df_dreamer, 'dreamer'), (df_inst, 'instance'), (df_mean, 'mean only'), (df_2dinstance, '2d instance')])

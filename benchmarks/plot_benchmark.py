"""
This script reads the benchmark results from 'benchmark_results.csv' and plots
the results. The first plot compares the CVMatrix and NaiveCVMatrix models
across different preprocessing combinations. The second plot shows the CVMatrix
model across different preprocessing combinations.

The plots are saved as 'benchmark_cvmatrix_vs_naive.png' and 'benchmark_cvmatrix.png'
respectively.

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.text import Text


def plot_cvmatrix_vs_naive(df, combination_to_color_map):
    fig, ax = plt.subplots(figsize=(10, 10))
    preprocessing_combinations = []
    preprocessing_combinations.append(
        {'center_X': False, 'center_Y': False, 'scale_X': False, 'scale_Y': False}
    )
    preprocessing_combinations.append(
        {'center_X': True, 'center_Y': True, 'scale_X': False, 'scale_Y': False}
    )
    preprocessing_combinations.append(
        {'center_X': True, 'center_Y': True, 'scale_X': True, 'scale_Y': True}
    )
    N = df['N'].unique()[0]
    K = df['K'].unique()[0]
    M = df['M'].unique()[0]
    for combination in preprocessing_combinations:
        color = combination_to_color_map[str(combination)]
        fast_times = []
        naive_times = []
        Ps = []
        for P in df['P'].unique():
            fast_time = df[
                (df['model'] == 'CVMatrix') &
                (df['P'] == P) &
                (df['center_X'] == combination['center_X']) &
                (df['center_Y'] == combination['center_Y']) &
                (df['scale_X'] == combination['scale_X']) &
                (df['scale_Y'] == combination['scale_Y'])
            ]['time'].values[0]
            naive_time = df[
                (df['model'] == 'NaiveCVMatrix') &
                (df['P'] == P) &
                (df['center_X'] == combination['center_X']) &
                (df['center_Y'] == combination['center_Y']) &
                (df['scale_X'] == combination['scale_X']) &
                (df['scale_Y'] == combination['scale_Y'])
            ]['time'].values[0]
            fast_times.append(fast_time)
            naive_times.append(naive_time)
            Ps.append(P)
        label = "Fast (CVMatrix), " + ', '.join([f"{k}={v}" for k, v in combination.items()])
        ax.plot(
            Ps,
            fast_times,
            marker='D',
            color=color,
            linestyle='dashed',
            label=label
        )
        label = "Baseline, " + ', '.join(
                [f"{k}={v}" for k, v in combination.items()]
            )
        ax.plot(
            Ps,
            naive_times,
            marker='s',
            color=color,
            linestyle='dotted',
            label=label
        )
    lines = [1, 60, 3600, 86400]
    line_names = ['Second', 'Minute', 'Hour', 'Day']
    for j, line in enumerate(lines):
        ax.axhline(y=line, color='k', linestyle='--', linewidth=1)
        ax.text(
            300000, line, f"1 {line_names[j]}", fontsize=10, ha='center', va='center'
        )
    cvmatrix_version = df['version'].unique()[0]
    version_text = f'CVMatrix version: {cvmatrix_version}'
    ax.text(1, 230000, version_text, fontsize=9, ha='center', va='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    current_x_ticks, current_x_labels = plt.xticks()
    extra_x_ticks = np.array([0, 3, 5])
    extra_x_labels = [Text(0 , 0, '0'), Text(3 , 0, '3'), Text(5, 0, '5')]
    new_x_ticks = np.concatenate((current_x_ticks, extra_x_ticks))
    new_x_labels = current_x_labels + extra_x_labels
    sort_idxs = np.argsort(new_x_ticks)
    new_x_ticks = new_x_ticks[sort_idxs]
    new_x_labels = (np.array(new_x_labels)[sort_idxs]).tolist()
    start_idx = np.where(new_x_ticks == 3)[0][0]
    stop_idx = np.where(new_x_ticks == 1e5)[0][0] + 1
    new_x_ticks = new_x_ticks[start_idx:stop_idx]
    new_x_labels = new_x_labels[start_idx:stop_idx]
    ax.set_xticks(new_x_ticks)
    ax.set_xticklabels(new_x_labels)
    ax.set_title(f'Fast (CVMatrix) vs. Baseline Cross-Validation (N={N:,}, K={K}, M={M})')
    ax.set_xlabel('P (cross-validation folds)')
    ax.set_ylabel('Time (s)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('benchmark_cvmatrix_vs_naive.png')
    plt.clf()

def plot_cvmatrix(df, combination_to_color_map):
    fig, ax = plt.subplots(figsize=(10, 10))
    preprocessing_combinations = []
    center_Xs = [False, True]
    center_Ys = [False, True]
    scale_Xs = [False, True]
    scale_Ys = [False, True]
    for center_X, center_Y, scale_X, scale_Y \
        in product(center_Xs, center_Ys, scale_Xs, scale_Ys):
        preprocessing_combinations.append(
            {
                'center_X': center_X,
                'center_Y': center_Y,
                'scale_X': scale_X,
                'scale_Y': scale_Y
            }
        )
    N = df['N'].unique()[0]
    K = df['K'].unique()[0]
    M = df['M'].unique()[0]
    for combination in preprocessing_combinations:
        color = combination_to_color_map[str(combination)]
        fast_times = []
        Ps = []
        for P in df['P'].unique():
            fast_time = df[
                (df['model'] == 'CVMatrix') &
                (df['P'] == P) &
                (df['center_X'] == combination['center_X']) &
                (df['center_Y'] == combination['center_Y']) &
                (df['scale_X'] == combination['scale_X']) &
                (df['scale_Y'] == combination['scale_Y'])
            ]['time'].values[0]
            fast_times.append(fast_time)
            Ps.append(P)
        label = ', '.join([f"{k}={v}" for k, v in combination.items()])
        ax.plot(
            Ps,
            fast_times,
            marker='D',
            color=color,
            linestyle='dashed',
            label=label
        )
    lines = [1, 60]
    line_names = ['Second', 'Minute']
    for j, line in enumerate(lines):
        ax.axhline(y=line, color='k', linestyle='--', linewidth=1)
        ax.text(
            300000,line, f"1 {line_names[j]}", fontsize=10, ha='center', va='center'
        )
    cvmatrix_version = df['version'].unique()[0]
    version_text = f'CVMatrix version: {cvmatrix_version}'
    ax.text(1, 125, version_text, fontsize=9, ha='center', va='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    current_x_ticks, current_x_labels = plt.xticks()
    extra_x_ticks = np.array([0, 3, 5])
    extra_x_labels = [Text(0 , 0, '0'), Text(3 , 0, '3'), Text(5, 0, '5')]
    new_x_ticks = np.concatenate((current_x_ticks, extra_x_ticks))
    new_x_labels = current_x_labels + extra_x_labels
    sort_idxs = np.argsort(new_x_ticks)
    new_x_ticks = new_x_ticks[sort_idxs]
    new_x_labels = (np.array(new_x_labels)[sort_idxs]).tolist()
    start_idx = np.where(new_x_ticks == 3)[0][0]
    stop_idx = np.where(new_x_ticks == 1e5)[0][0] + 1
    new_x_ticks = new_x_ticks[start_idx:stop_idx]
    new_x_labels = new_x_labels[start_idx:stop_idx]
    ax.set_xticks(new_x_ticks)
    ax.set_xticklabels(new_x_labels)
    ax.set_title(f'Fast (CVMatrix) Cross-Validation (N={N:,}, K={K}, M={M})')
    ax.set_xlabel('P (cross-validation folds)')
    ax.set_ylabel('Time (s)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('benchmark_cvmatrix.png')
    plt.clf()

def get_combination_to_color_map():
    preprocessing_combinations = []
    center_Xs = [False, True]
    center_Ys = [False, True]
    scale_Xs = [False, True]
    scale_Ys = [False, True]
    for center_X, center_Y, scale_X, scale_Y \
        in product(center_Xs, center_Ys, scale_Xs, scale_Ys):
        preprocessing_combinations.append(
            {
                'center_X': center_X,
                'center_Y': center_Y,
                'scale_X': scale_X,
                'scale_Y': scale_Y
            }
        )
    # Define a list of 16 colorblind-friendly colors
    cm = colormaps.get_cmap('tab20')
    colors = [cm(i) for i in range(16)]
    combination_to_color_map = {}
    for i, combination in enumerate(preprocessing_combinations):
        combination_to_color_map[str(combination)] = colors[i]
    return combination_to_color_map

if __name__ == '__main__':
    # Set the font size, legend size, and axis label size
    plt.rcParams.update({'font.size': 12})
    df = pd.read_csv('benchmark_results.csv')
    combination_to_color_map = get_combination_to_color_map()
    plot_cvmatrix_vs_naive(df, combination_to_color_map)
    plot_cvmatrix(df, combination_to_color_map)

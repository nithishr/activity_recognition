import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import skew, kurtosis

def read_concatenate_files(filelist, usecols):
    frame_list = []
    for component_file in filelist:
        df = pd.read_csv(component_file, delimiter=',', header=0, skipinitialspace=True, usecols=usecols)
        frame_list.append(df)
    output_frame = pd.DataFrame()
    output_frame = pd.concat(frame_list)
    return output_frame
    
    
def create_sliding_windows_by_time(data_frame, col_index, window_time_period=3000):
    windows = []
    ts_windows = []
    ts_begin = None
#     window_time_period = 3000
    for row in data_frame.iterrows():
    #     print("Col1", type(row[1][1]))
    #     print("Col2", type(row[1][0]))
        if ts_begin == None:
    #         print(1)
            ts_begin = row[1][0]
            window = []
            ts_window = []
            window.append(row[1][col_index])
            ts_window.append(row[1][0])
        elif(row[1][0] <= (ts_begin + window_time_period)):
            window.append(row[1][col_index])
            ts_window.append(row[1][0])
        else:
            windows.append(window)
            ts_windows.append(ts_window)
            window = [row[1][col_index]]
            ts_window = [row[1][0]]
            ts_begin = row[1][0]
    ts_windows.append(ts_window)
    windows.append(window)
    # windows = windows[1:]
    # ts_windows = ts_windows[1:]
    # print(len(ts_windows),len(windows))
    return windows, ts_windows
    

def compute_skew_window(windows):
    skewness_windows = []
    for i in range(len(windows)):
        if len(windows[i]) <= 1:
            continue
        skewness_windows.append(skew(windows[i]))
    return skewness_windows
    
def compute_kurtosis_window(windows):
    kurtosis_windows = []
    for i in range(len(windows)):
        if len(windows[i]) <= 1:
            continue
        kurtosis_windows.append(kurtosis(windows[i]))
    return kurtosis_windows
    
def compute_sum_derivative_window(windows, ts_windows):
    sum_derivative_windows = []
    derivative_windows = []
    for i in range(len(windows)):
        if len(windows[i]) <= 1:
            continue
        derivative_window = np.gradient(windows[i], ts_windows[i])
        derivative_windows.append(derivative_window)
        sum_derivative_windows.append(sum(derivative_window))
    return sum_derivative_windows
    
    
def compute_std_deviation_window(windows):
    std_windows = []
    for i in range(len(windows)):
        if len(windows[i]) <= 1:
            continue
        std_windows.append(np.std(windows[i]))
    return std_windows

def compute_percentile_norm_window(windows, percentile=50):
    percentile_windows = []
    for i in range(len(windows)):
        if len(windows[i]) <= 1:
            continue
        window_norm = normalize(np.array(windows[i]).reshape(1,-1))
        percentile_windows.append(np.percentile(window_norm, percentile))
    return percentile_windows
    
    
def compute_inter_quartile_distance_window(windows):
    iqr_windows = []
    for i in range(len(windows)):
        if len(windows[i]) <= 1:
            continue
        q75, q25 = np.percentile(windows[i], [75 ,25])
        iqr_windows.append(q75 - q25)
    return iqr_windows

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import skew, kurtosis
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier


walking_files = ['Activity-Data/Samsung/Walk/Walk_N2/Walk_N2Pressure_clean.csv', 
				 'Activity-Data/Samsung/Walk/Walk_N_3/Walk_N_3Pressure_clean.csv', 
                 'Activity-Data/Samsung/Walk/Walk_N/Walk_NPressure_clean.csv',
                 'Activity-Data/Samsung/Walk/Walk_P1/Walk_P1Pressure_clean.csv',
                 'Activity-Data/Samsung/Walk/Walk_P2/Walk_P2Pressure_clean.csv',
                 'Activity-Data/Walking/01/Walking01Pressure_clean.csv', 
                 'Activity-Data/Walking/2/Walking2Pressure_clean.csv',
                 'Activity-Data/Walking/3/Walking3Pressure_clean.csv',
                 'Activity-Data/Walking/04/Walking04Pressure_clean.csv',
                 'Activity-Data/Walking/5/Walking5Pressure_clean.csv',
                 'Activity-Data/1912/Walk/1/Walk_1Pressure_clean.csv',
                 'Activity-Data/1912/Walk/2/Walk_8Pressure_clean.csv',
                 'Activity-Data/2012/Walk/1/Walk_1Pressure_clean.csv', 
                 'Activity-Data/2012/Walk/2/Walk_2Pressure_clean.csv',
                 'Activity-Data/2012/Walk/3/Walk_3Pressure_clean.csv',
                 'Activity-Data/2012/Walk/4/Walk_4Pressure_clean.csv',
                 'Activity-Data/2012/Walk/5/Walk_5Pressure_clean.csv', 
                 'Activity-Data/2012/Walk/6/Walk_6Pressure_clean.csv',
                 'Activity-Data/2012/Walk/7/Walk7Pressure_clean.csv',
                 'Activity-Data/2012/Walk/8/Walk_8Pressure_clean.csv',
                 'Activity-Data/2812/Walking/1/W_1Pressure.csv',
				 'Activity-Data/2812/Walking/2/W_3Pressure.csv',
				 'Activity-Data/2812/Walking/3/W_4Pressure.csv',
				 'Activity-Data/3012/Walk/1/W_0Pressure.csv']
                 
climbing_files = ['Activity-Data/Samsung/Climbing_Up/Climb_Up_1/Climb_Up_1Pressure_clean.csv',
				 'Activity-Data/Samsung/Climbing_Up/Climb_Up_2/Climb_Up_2Pressure_clean.csv',    
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_3/Climb_Up_3Pressure_clean.csv',
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_4/Climb_Up_4Pressure_clean.csv',    
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_4/Climb_Up_4Pressure_clean.csv', 
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_6/Climb_Up_6Pressure_clean.csv',
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_7/Climb_Up_7Pressure_clean.csv', 
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_8/Climb_Up_8Pressure_clean.csv',
                 'Activity-Data/Climbing_Stairs/1/Climbing_Stairs1Pressure_clean.csv', 
                 'Activity-Data/Climbing_Stairs/2/Climbing_Stairs2Pressure_clean.csv', 
                 'Activity-Data/Climbing_Stairs/3/Climbing_Stairs3Pressure_clean.csv',
                 'Activity-Data/Climbing_Stairs/5/Climbing_Stairs5Pressure_clean.csv',
                 'Activity-Data/Climbing_Stairs/6/Climbing_Stairs6Pressure_clean.csv', 
                 'Activity-Data/Climbing_Stairs/7/Climbing_Stairs7Pressure_clean.csv',
                 'Activity-Data/1912/Stairs_Up/1/Stairs_up_1Pressure_clean.csv', 
                 'Activity-Data/1912/Stairs_Up/2/Stairs_up_2Pressure_clean.csv',
                 'Activity-Data/1912/Stairs_Up/3/Stairs_up_3Pressure_clean.csv', 
                 'Activity-Data/1912/Stairs_Up/4/Stairs_up_4Pressure_clean.csv',
                 'Activity-Data/1912/Stairs_Up/5/Stairs_up_5Pressure_clean.csv', 
                 'Activity-Data/2012/Stairs_Up/1/Stairs_up_1Pressure_clean.csv', 
                 'Activity-Data/2012/Stairs_Up/2/Stairs_up_2Pressure_clean.csv', 
                 'Activity-Data/2012/Stairs_Up/3/Stairs_up_3Pressure_clean.csv', 
                 'Activity-Data/2012/Stairs_Up/4/Stairs_up_4Pressure_clean.csv', 
                 'Activity-Data/2012/Stairs_Up/5/Stairs_up_5Pressure_clean.csv',
                 'Activity-Data/2912/Stairs_up/1/Su_1Pressure.csv',
                 'Activity-Data/3012/Stairs_up/1/Su_0Pressure.csv']
                 
 
downstairs_files = ['Activity-Data/Samsung/Climbing_Down/Climb_Down_1/Climb_Down_1Pressure_clean.csv', 
					'Activity-Data/Samsung/Climbing_Down/Climb_Down_2/Climb_Down_2Pressure_clean.csv',    
                    'Activity-Data/Samsung/Climbing_Down/Climb_Down_3/Climb_Down_3Pressure_clean.csv', 
                    'Activity-Data/Samsung/Climbing_Down/Climb_Down_6/Climb_Down_6Pressure_clean.csv',
                    'Activity-Data/Samsung/Climbing_Down/Climb_Down_8/Climb_Down_8Pressure_clean.csv', 
                    'Activity-Data/Samsung/Climbing_Down/Climb_Down_9/Climb_Down_9Pressure_clean.csv',
                    'Activity-Data/Downstairs/1/Downstairs1Pressure_clean.csv', 
                    'Activity-Data/Downstairs/2/Downstairs2Pressure_clean.csv', 
                    'Activity-Data/Downstairs/3/Downstairs3Pressure_clean.csv', 
                    'Activity-Data/Downstairs/4/Downstairs4Pressure_clean.csv',
                    'Activity-Data/Downstairs/5/Downstairs5Pressure_clean.csv', 
                    'Activity-Data/Downstairs/6/Downstairs6Pressure_clean.csv',
                    'Activity-Data/1912/Stairs_Down/1/Stairs_down_1Pressure_clean.csv',
                    'Activity-Data/1912/Stairs_Down/2/Stairs_down_2Pressure_clean.csv',
                    'Activity-Data/1912/Stairs_Down/3/Stairs_down_3Pressure_clean.csv',
                    'Activity-Data/1912/Stairs_Down/4/Stairs_down_4Pressure_clean.csv',
                    'Activity-Data/1912/Stairs_Down/5/Stairs_down_5Pressure_clean.csv',
                    'Activity-Data/2012/Stairs_Down/1/Stairs_down_1Pressure_clean.csv', 
                    'Activity-Data/2012/Stairs_Down/2/Stairs_down_2Pressure_clean.csv', 
                    'Activity-Data/2012/Stairs_Down/3/Stairs_down_3Pressure_clean.csv', 
                    'Activity-Data/2012/Stairs_Down/4/Stairs_down_4Pressure_clean.csv', 
                    'Activity-Data/2012/Stairs_Down/5/Stairs_down_5Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_down/1/Sd_3Pressure.csv',
					'Activity-Data/2812/Stairs_down/2/Sd_4Pressure.csv',
					'Activity-Data/2812/Stairs_up/1/Su_1Pressure.csv',
					'Activity-Data/2812/Stairs_up/10/Sup_5Pressure.csv',
					'Activity-Data/2812/Stairs_up/11/Sup_6Pressure.csv',
					'Activity-Data/2812/Stairs_up/12/SUP_1Pressure.csv',
					'Activity-Data/2812/Stairs_up/2/Su_2Pressure.csv',
					'Activity-Data/2812/Stairs_up/3/Su_3Pressure.csv',
					'Activity-Data/2812/Stairs_up/4/Su_4Pressure.csv',
					'Activity-Data/2812/Stairs_up/5/Su_5Pressure.csv',
					'Activity-Data/2812/Stairs_up/6/Su_6Pressure.csv',
					'Activity-Data/2812/Stairs_up/7/Sup2Pressure.csv',
					'Activity-Data/2812/Stairs_up/8/Sup3Pressure.csv',
					'Activity-Data/2812/Stairs_up/9/Sup4Pressure.csv',
					'Activity-Data/2912/Stairs_down/1/Sd_2Pressure.csv',
					'Activity-Data/3012/Stairs_down/1/Sd_0Pressure.csv'
                   ]

escalator_up_files = ['Activity-Data/1912/Esc_Up/1/Esc_up_1Pressure_clean.csv',
					  'Activity-Data/1912/Esc_Up/2/Esc_up_2Pressure_clean.csv',
                      'Activity-Data/1912/Esc_Up/3/Esc_up_3Pressure_clean.csv',
                      'Activity-Data/1912/Esc_Up/4/Esc_Up_4Pressure_clean.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/Esc_Up_1/Esc_Up_1Pressure_clean.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/Esc_Up_2/Esc_Up_2Pressure_clean.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/3/Esc_Up_3Pressure_clean.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/4/Esc_up_4Pressure_clean.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/5/Esc_up_5Pressure_clean.csv', 
                      'Activity-Data/2012/Esc_Up/1/Esc_up_1Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/1/Eu_1Pressure.csv',
					  'Activity-Data/2812/Escalator_up/10/Eu_12Pressure.csv',
					  'Activity-Data/2812/Escalator_up/11/Eu_13Pressure.csv',
					  'Activity-Data/2812/Escalator_up/12/Eu_14Pressure.csv',
					  'Activity-Data/2812/Escalator_up/2/Eu_2Pressure.csv',
					  'Activity-Data/2812/Escalator_up/3/Eu_3Pressure.csv',
					  'Activity-Data/2812/Escalator_up/4/Eu_4Pressure.csv',
					  'Activity-Data/2812/Escalator_up/5/Eu_5Pressure.csv',
					  'Activity-Data/2812/Escalator_up/6/Eu_6Pressure.csv',
					  'Activity-Data/2812/Escalator_up/7/Eu_7Pressure.csv',
					  'Activity-Data/2812/Escalator_up/8/Eu_8Pressure.csv',
					  'Activity-Data/2812/Escalator_up/9/Eu_11Pressure.csv',
					  'Activity-Data/2912/Escalator_up/1/Eu_1Pressure.csv',
					  'Activity-Data/2912/Escalator_up/2/Eu_2Pressure.csv',
					  'Activity-Data/2912/Escalator_up/3/Eu_5Pressure.csv',
					  'Activity-Data/2912/Escalator_up/4/Eu_6Pressure.csv',
					  'Activity-Data/2912/Escalator_up/5/Eu_7Pressure.csv',
					  'Activity-Data/3012/Escalator_up/1/Eu_0Pressure.csv',
					  'Activity-Data/3012/Escalator_up/2/Eu_2Pressure.csv',
					  'Activity-Data/3012/Escalator_up/3/Eu_3Pressure.csv']
                      
escalator_down_files = ['Activity-Data/Samsung/061217/Esc_down/Esc_down_1/Esc_down_1Pressure_clean.csv',
						'Activity-Data/Samsung/061217/Esc_down/Esc_down_2/Esc_down_2Pressure_clean.csv',
                        'Activity-Data/Samsung/061217/Esc_down/3/Esc_d2Pressure_clean.csv',
                        'Activity-Data/2812/Escalator_down/1/Ed_1Pressure.csv', 
						'Activity-Data/2812/Escalator_down/2/Ed_2Pressure.csv',
						'Activity-Data/2812/Escalator_down/3/Ed_7Pressure.csv',
						'Activity-Data/2912/Escalator_down/1/Ed_1Pressure.csv',
						'Activity-Data/2912/Escalator_down/2/Ed_2Pressure.csv',
						'Activity-Data/2912/Escalator_down/3/Ed_3Pressure.csv',
						'Activity-Data/2912/Escalator_down/4/Ed_5Pressure.csv',
						'Activity-Data/2912/Escalator_down/5/Ed_8Pressure.csv',
						'Activity-Data/2912/Escalator_down/6/Ed_9Pressure.csv',
						'Activity-Data/3012/Escalator_down/1/Ed_0Pressure.csv']
                        
lift_up_files = ['Activity-Data/1912/Lift_Up/1/Lift_up_1Pressure_clean.csv', 
				 'Activity-Data/1912/Lift_Up/2/Lift_up_4Pressure_clean.csv',
                 'Activity-Data/1912/Lift_Up/3/Lift_up_9Pressure_clean.csv', 
                 'Activity-Data/Samsung/061217/Lift_Up/1/Lift_Up_2Pressure_clean.csv',
                 'Activity-Data/Samsung/061217/Lift_Up/2/Lift_up_5Pressure_clean.csv',
                 'Activity-Data/2812/Lift_up/1/Lu_1Pressure.csv',
				 'Activity-Data/2812/Lift_up/2/Lu_6Pressure.csv',
				 'Activity-Data/2812/Lift_up/3/Lu_10Pressure.csv',
				 'Activity-Data/2912/Lift_up/1/Lu_1Pressure.csv',
				 'Activity-Data/2912/Lift_up/2/Lu_2Pressure.csv',
				 'Activity-Data/2912/Lift_up/3/Lu_3Pressure.csv',
				 'Activity-Data/2912/Lift_up/4/Lu_5Pressure.csv']

lift_down_files = ['Activity-Data/1912/Lift_Down/1/Lift_down_9Pressure_clean.csv',
				   'Activity-Data/Samsung/061217/Lift_Down/1/Lift_Down_2Pressure_clean.csv',
                   'Activity-Data/Samsung/061217/Lift_Down/2/Lift_down_3Pressure_clean.csv',
                   'Activity-Data/Samsung/061217/Lift_Down/3/Lift_down_Pressure_clean.csv',
                   'Activity-Data/2812/Lift_down/1/Ld_1Pressure.csv',
				   'Activity-Data/2812/Lift_down/2/Ld_4Pressure.csv',
				   'Activity-Data/2812/Lift_down/3/Ld_8Pressure.csv',
				   'Activity-Data/2912/Lift_down/1/Ld_1Pressure.csv',
				   'Activity-Data/2912/Lift_down/2/Ld_3Pressure.csv',
				   'Activity-Data/2912/Lift_down/3/Ld_7Pressure.csv',
				   'Activity-Data/2912/Lift_down/4/Ld_8Pressure.csv',
				   'Activity-Data/3012/Lift_down/1/Ld_0Pressure.csv']
                 											 
                   
                   
def read_concatenate_files(filelist, usecols):
    frame_list = []
    for component_file in filelist:
        df = pd.read_csv(component_file, delimiter=',', header=0, skipinitialspace=True, usecols=usecols)
        frame_list.append(df)
    output_frame = pd.DataFrame()
    output_frame = pd.concat(frame_list)
    return output_frame
    
def visualize_pressure(input_file):
    df = pd.read_csv(input_file, delimiter=',', header=0)
    plt.figure()
    plt.title(input_file)
    plt.plot(df[' pressure'].values)

    
def remove_rows(index_list, input_file):
    df = pd.read_csv(input_file, delimiter=',', header=0)
    df1 = df.drop(index_list)
#     visualize_pressure(df1)
    output_file = input_file.split('.')[0] + '_clean.csv'
    df1.to_csv(output_file, index=False)
    return output_file

    
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
    
    
def compute_norm_window(windows):
	norm_windows = []
	for i in range(len(windows)):
		if len(windows[i]) <= 1:
			continue
		norm_windows.append(sum(windows[i]))
	return norm_windows
	
def create_window(dataframe, ts_start, window_length):
    """Return the rows in the dataframe within the timestamp range [ts_start, ts_start + window_length]"""
    return dataframe[(dataframe['timestamp'] >= ts_start) & (dataframe['timestamp'] <= ts_start + window_length)]
    

def create_sliding_windows(dataframe, sliding_window_interval, window_length):
    """Create sliding windows for a single dataframe"""
    ts_min = min(dataframe['timestamp'])
    ts_max = max(dataframe['timestamp'])
    ts_iter = ts_min
    windows = []
    while ts_iter <= ts_max:
        window = create_window(dataframe, ts_iter, window_length)
        windows.append(window)
        ts_iter += sliding_window_interval
    return windows        
    
    
def create_pressure_features_windows(windows_frames, percentile=50):
    windows_features_list = []
    for window in windows_frames:
        if len(window) <= 1:
            continue
        else:
            skew_windows = skew(window['pressure'])

            window_norm = normalize(np.array(window['pressure']).reshape(1,-1))
            percentile = np.percentile(window_norm, percentile)

            q75, q25 = np.percentile(window['pressure'], [75 ,25])
            iqr = q75 - q25

            kurtosis_w = kurtosis(window['pressure'])

            std_deviation = np.std(window['pressure'])

            #derivative = compute_sum_derivative_window(window['pressure'], window['timestamp'])
            derivative_window = np.gradient(window['pressure'], window['timestamp'])
            derivative = sum(derivative_window)
            
            median = np.median(window['pressure'].values)
            window['pressure_norm'] = window['pressure'].apply(lambda x: x-median)    
            norm = sum(window['pressure_norm'])
#             print(skew_windows, percentile, iqr, kurtosis_w, std_deviation, derivative)
            window_features = pd.DataFrame()
            window_features['skew'] = [skew_windows]
            window_features['percentile'] = [percentile]
            window_features['iqr'] = [iqr]
            window_features['kurtosis'] = [kurtosis_w]
            window_features['std_deviation'] = [std_deviation]
            window_features['derivative'] = [derivative]
#             print(window_features)
            window_features['norm'] = [norm]
            windows_features_list.append(window_features)
#     print(len(windows_features_list))
    df_features = pd.concat(windows_features_list)
    return df_features
    
def create_data_frame(input_files, sliding_window_interval, window_length, header=0, usecols=[0,1,2]):
    frame_list = []
    for i_file in input_files:
        df = pd.read_csv(i_file, delimiter=',', header = header, skipinitialspace = True, usecols = usecols)
        df_windows = create_sliding_windows(df, sliding_window_interval, window_length)
        df_features = create_pressure_features_windows(df_windows)
#     print(len(df), len(df_features))
        frame_list.append(df_features)
    out_frame = pd.DataFrame()
    out_frame = pd.concat(frame_list)
    out_frame = out_frame.reset_index(drop=True)
    return out_frame


def create_features_from_files(sliding_window_interval, window_interval, w_files = walking_files, 
                               su_files = climbing_files, sd_files = downstairs_files,
                               eu_files = escalator_up_files, ed_files = escalator_down_files,
                               lu_files = lift_up_files, ld_files = lift_down_files):
    w_frame = create_data_frame(walking_files, sliding_window_interval, window_interval)
    w_frame['label'] = 0
    su_frame = create_data_frame(su_files, sliding_window_interval, window_interval)
    su_frame['label'] = 1
    sd_frame = create_data_frame(sd_files, sliding_window_interval, window_interval)
    sd_frame['label'] = 2
    eu_frame = create_data_frame(eu_files, sliding_window_interval, window_interval)
    eu_frame['label'] = 3
    ed_frame = create_data_frame(ed_files, sliding_window_interval, window_interval)
    ed_frame['label'] = 4
    lu_frame = create_data_frame(lu_files, sliding_window_interval, window_interval)
    lu_frame['label'] = 5
    ld_frame = create_data_frame(ld_files, sliding_window_interval, window_interval)
    ld_frame['label'] = 6
    return w_frame, su_frame, sd_frame, eu_frame, ed_frame, lu_frame, ld_frame
    
    
def print_characteristics(w_frame, su_frame, sd_frame, eu_frame, ed_frame, lu_frame, ld_frame):
    print("Walking Frame")
    display(w_frame.head(), w_frame.tail())
    print("Stairs Up Frame")
    display(su_frame.head(), su_frame.tail())
    print("Stairs Down Frame")
    display(sd_frame.head(), sd_frame.tail())
    print("Escalator Up Frame")
    display(eu_frame.head(), eu_frame.tail())
    print("Escalator Down Frame")
    display(ed_frame.head(), ed_frame.tail())
    print("Lift Up Frame")
    display(lu_frame.head(), lu_frame.tail())
    print("Lift Down Frame")
    display(ld_frame.head(), ld_frame.tail())
    
    

def create_dataset_vertical_transition(w_frame, su_frame, sd_frame, eu_frame, ed_frame, lu_frame, ld_frame):
    w_frame['label_vertical'] = 0
    v_frame = pd.concat([su_frame, sd_frame, eu_frame, ed_frame, lu_frame, ld_frame])
    v_frame['label_vertical'] = 1
    v_frame = v_frame.reset_index(drop = True)
    return w_frame, v_frame    

    
def visualize_vertical_transition_features(walking_frame, vertical_frame):
    v_features_array = vertical_frame.as_matrix(columns=vertical_frame.columns)
    w_features_array = walking_frame.as_matrix(columns=walking_frame.columns)
    X = np.concatenate([w_features_array, v_features_array])
    print(X.shape)
    
    windows_map = { 0:{'values':X[:,0], 'legend': 'skewness'}, 1:{'values':X[:,5], 'legend': 'gradient'},
               2:{'values':X[:,3], 'legend':'kurtosis'}, 3:{'values':X[:,4], 'legend':'std deviation'},
               4:{'values':X[:,1], 'legend':'percentile_windows'}, 5:{'values':X[:,2], 'legend': 'iqr'},
               6:{'values':X[:,6], 'legend':'norm'}, 7:{'values':X[:,8], 'legend':'labels'}}

    combos = list(combinations(list(range(len(windows_map.keys())-1)), 2))
#     print(len(combos))

    for combo in combos:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(windows_map[combo[0]]['values'][:len(walking_frame)], windows_map[combo[1]]['values'][:len(walking_frame)], c='r', alpha=0.5, label='walking')
        ax1.scatter(windows_map[combo[0]]['values'][len(walking_frame):], windows_map[combo[1]]['values'][len(walking_frame):], c='b', alpha=0.5, label='vertical')
        ax = plt.subplot()
        ax.set_xlabel(windows_map[combo[0]]['legend'])
        ax.set_ylabel(windows_map[combo[1]]['legend'])
        ax.legend()
        ax.grid(True)
        plt.show()
        

def create_vertical_transition_dataset(walking_frame, vertical_frame):
    v_features_array = vertical_frame.as_matrix(columns=vertical_frame.columns)
    w_features_array = walking_frame.as_matrix(columns=walking_frame.columns)
    X = np.concatenate([w_features_array, v_features_array])
    Y = X[:,8]
    X = X[:,:7]
    return X, Y
    

def classify_logistic_regression(XTrain, XTest, YTrain, YTest):
    YPred = LogisticRegressionCV(solver = 'liblinear').fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1    

    
def classify_sgd(XTrain, XTest, YTrain, YTest):
    YPred = SGDClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1    
    

def classify_svm(XTrain, XTest, YTrain, YTest):
    YPred = svm.SVC().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    

def classify_linearSVM(XTrain, XTest, YTrain, YTest):
    YPred = svm.LinearSVC().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    

def classify_knn(XTrain, XTest, YTrain, YTest):
    YPred = KNeighborsClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    

def classify_gaussian_process_classifier(XTrain, XTest, YTrain, YTest):
    YPred = GaussianProcessClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    
    
def classify_naive_bayes(XTrain, XTest, YTrain, YTest):
    YPred = GaussianNB().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    

def classify_decision_tree(XTrain, XTest, YTrain, YTest):
    YPred = tree.DecisionTreeClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    

def classify_random_forest(XTrain, XTest, YTrain, YTest):
    YPred = RandomForestClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    

def classify_ada_boost(XTrain, XTest, YTrain, YTest):
    YPred = AdaBoostClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    

def classify_gradient_boost(XTrain, XTest, YTrain, YTest):
    YPred = GradientBoostingClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    

def classify_neural_network(XTrain, XTest, YTrain, YTest):
    YPred = MLPClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    return acc,f1   
    

def stratified_KFoldVal(X, Y, classify, n_folds=10):
    accuracies = []
    f1_scores = []
    skf = StratifiedKFold(n_folds)
    for train, test in skf.split(X,Y):
        XTrain, XTest, YTrain, YTest = X[train], X[test], Y[train], Y[test]
        acc, f1 = classify(XTrain, XTest, YTrain, YTest)
        f1_scores.append(f1)
        accuracies.append(acc)
    return f1_scores, accuracies
    

def KFoldVal(X, Y, classify, n_folds=10):
    accuracies = []
    f1_scores = []
    kf = KFold(n_folds)
    for train, test in kf.split(X):
        XTrain, XTest, YTrain, YTest = X[train], X[test], Y[train], Y[test]
        acc, f1 = classify(XTrain, XTest, YTrain, YTest)
        f1_scores.append(f1)
        accuracies.append(acc)
    return f1_scores, accuracies
    



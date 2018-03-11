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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
                 'Activity-Data/2812/Walking/1/W_1Pressure_clean.csv',
                 'Activity-Data/2812/Walking/2/W_3Pressure_clean.csv',
                 'Activity-Data/2812/Walking/3/W_4Pressure_clean.csv',
                 'Activity-Data/3012/Walk/1/W_0Pressure_clean.csv']
                 
climbing_files = ['Activity-Data/Samsung/Climbing_Up/Climb_Up_1/Climb_Up_1Pressure_clean.csv',
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_2/Climb_Up_2Pressure_clean.csv',    
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_3/Climb_Up_3Pressure_clean.csv',
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
                 'Activity-Data/2912/Stairs_up/1/Su_1Pressure_clean.csv',
                 'Activity-Data/3012/Stairs_up/1/Su_0Pressure_clean.csv',
                 'Activity-Data/3112/Stairs_up/1/Su_0Pressure_clean.csv',
                 'Activity-Data/3112/Stairs_up/2/Su_1Pressure_clean.csv',
                 'Activity-Data/3112/Stairs_up/3/Su_2Pressure_clean.csv',
                 'Activity-Data/3112/Stairs_up/4/Su_3Pressure_clean.csv',
                 'Activity-Data/3112/Stairs_up/5/Su_4Pressure_clean.csv',
                 'Activity-Data/3112/Stairs_up/6/Su_5Pressure_clean.csv',
                 'Activity-Data/3112/Stairs_up/7/Su_6Pressure_clean.csv',
                 'Activity-Data/3112/Stairs_up/8/Su_7Pressure_clean.csv',
                 'Activity-Data/3112/Stairs_up/9/Su_8Pressure_clean.csv',
                 'Activity-Data/3212/Stairs_up/1/Su_2Pressure_clean.csv',
                 'Activity-Data/3212/Stairs_up/2/Su_8Pressure_clean.csv',
                 'Activity-Data/3212/Stairs_up/3/Su_11Pressure_clean.csv',
                 'Activity-Data/3312/Stairs_up/1/Su_0Pressure_clean.csv',
                 'Activity-Data/3312/Stairs_up/2/Su_1Pressure_clean.csv',
                 'Activity-Data/3312/Stairs_up/3/Su_6Pressure_clean.csv',
                 'Activity-Data/3312/Stairs_up/4/Su_8Pressure_clean.csv',
                 'Activity-Data/3312/Stairs_up/5/Su_9Pressure_clean.csv',
                 'Activity-Data/3312/Stairs_up/6/Su_10Pressure_clean.csv',
                 'Activity-Data/3312/Stairs_up/7/Su_18Pressure_clean.csv',
                 'Activity-Data/3312/Stairs_up/8/Sup_4Pressure_clean.csv',
                 'Activity-Data/3412/Stairs_up/1/Su_0Pressure_clean.csv',
                 'Activity-Data/3412/Stairs_up/2/Su_1Pressure_clean.csv',
                 'Activity-Data/3412/Stairs_up/3/Su_2Pressure_clean.csv',
                 'Activity-Data/3412/Stairs_up/4/Su_3Pressure_clean.csv',
                 'Activity-Data/3412/Stairs_up/5/Su_4Pressure_clean.csv',
                 'Activity-Data/3412/Stairs_up/6/Su_5Pressure_clean.csv',
                 'Activity-Data/3412/Stairs_up/7/Su_6Pressure_clean.csv',
                 'Activity-Data/3512/Stairs_up/1/Su_0Pressure_clean.csv',
                 'Activity-Data/3512/Stairs_up/2/Su_2Pressure_clean.csv',
                 'Activity-Data/3512/Stairs_up/3/Su_0Pressure_clean.csv']
                 
 
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
                    'Activity-Data/2812/Stairs_down/1/Sd_3Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_down/2/Sd_4Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/1/Su_1Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/10/Sup_5Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/11/Sup_6Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/12/SUP_1Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/2/Su_2Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/3/Su_3Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/4/Su_4Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/5/Su_5Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/6/Su_6Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/7/Sup2Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/8/Sup3Pressure_clean.csv',
                    'Activity-Data/2812/Stairs_up/9/Sup4Pressure_clean.csv',
                    'Activity-Data/2912/Stairs_down/1/Sd_2Pressure_clean.csv',
                    'Activity-Data/3012/Stairs_down/1/Sd_0Pressure_clean.csv',
                    'Activity-Data/3112/Stairs_down/1/Sd_0Pressure_clean.csv',
                    'Activity-Data/3112/Stairs_down/2/Sd_1Pressure_clean.csv',
                    'Activity-Data/3112/Stairs_down/3/Sd_2Pressure_clean.csv',
                    'Activity-Data/3112/Stairs_down/4/Sd_3Pressure_clean.csv',
                    'Activity-Data/3112/Stairs_down/5/Sd_5Pressure_clean.csv',
                    'Activity-Data/3112/Stairs_down/6/Sd_6Pressure_clean.csv',
                    'Activity-Data/3112/Stairs_down/7/Sd_3Pressure_clean.csv',
                    'Activity-Data/3212/Stairs_down/1/Sd_7Pressure_clean.csv',
                    'Activity-Data/3212/Stairs_down/2/Sd_9Pressure_clean.csv',
                    'Activity-Data/3212/Stairs_down/3/Sd_10Pressure_clean.csv',
                    'Activity-Data/3212/Stairs_down/4/Sd_15Pressure_clean.csv',
                    'Activity-Data/3312/Stairs_down/1/Sd_1Pressure_clean.csv',
                    'Activity-Data/3312/Stairs_down/2/Sd_4Pressure_clean.csv',
                    'Activity-Data/3312/Stairs_down/3/Sd_8Pressure_clean.csv',
                    'Activity-Data/3312/Stairs_down/4/Sd_10Pressure_clean.csv',
                    'Activity-Data/3312/Stairs_down/5/Sd_16Pressure_clean.csv',
                    'Activity-Data/3312/Stairs_down/6/Sd_18Pressure_clean.csv',
                    'Activity-Data/3312/Stairs_down/7/Sd_20Pressure_clean.csv',
                    'Activity-Data/3412/Stairs_down/1/Sd_0Pressure_clean.csv',
                    'Activity-Data/3412/Stairs_down/2/Sd_1Pressure_clean.csv',
                    'Activity-Data/3412/Stairs_down/3/Sd_2Pressure_clean.csv',
                    'Activity-Data/3412/Stairs_down/4/Sd_3Pressure_clean.csv',
                    'Activity-Data/3512/Stairs_down/1/Sd_0Pressure_clean.csv',
                    'Activity-Data/3512/Stairs_down/2/Sd_1Pressure_clean.csv',
                    'Activity-Data/3512/Stairs_down/3/Sd_2Pressure_clean.csv',
                    'Activity-Data/3512/Stairs_down/4/Sd_3Pressure_clean.csv',
                    'Activity-Data/3512/Stairs_down/5/Sd_4Pressure_clean.csv',
                    'Activity-Data/3512/Stairs_down/6/Sd_0Pressure_clean.csv',
                    'Activity-Data/3512/Stairs_down/7/Sd_1Pressure_clean.csv'
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
                      'Activity-Data/2812/Escalator_up/1/Eu_1Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/10/Eu_12Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/11/Eu_13Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/12/Eu_14Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/2/Eu_2Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/3/Eu_3Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/4/Eu_4Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/5/Eu_5Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/6/Eu_6Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/7/Eu_7Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/8/Eu_8Pressure_clean.csv',
                      'Activity-Data/2812/Escalator_up/9/Eu_11Pressure_clean.csv',
                      'Activity-Data/2912/Escalator_up/1/Eu_1Pressure_clean.csv',
                      'Activity-Data/2912/Escalator_up/2/Eu_2Pressure_clean.csv',
                      'Activity-Data/2912/Escalator_up/3/Eu_5Pressure_clean.csv',
                      'Activity-Data/2912/Escalator_up/4/Eu_6Pressure_clean.csv',
                      'Activity-Data/2912/Escalator_up/5/Eu_7Pressure_clean.csv',
                      'Activity-Data/3012/Escalator_up/1/Eu_0Pressure_clean.csv',
                      'Activity-Data/3012/Escalator_up/2/Eu_2Pressure_clean.csv',
                      'Activity-Data/3012/Escalator_up/3/Eu_3Pressure_clean.csv',
                      'Activity-Data/3112/Esc_up/1/Eu_0Pressure_clean.csv',
                      'Activity-Data/3112/Esc_up/2/Eu_2Pressure_clean.csv',
                      'Activity-Data/3112/Esc_up/3/Eu_5Pressure_clean.csv',
                      'Activity-Data/3112/Esc_up/4/Eu_6Pressure_clean.csv',
                      'Activity-Data/3412/Esc_up/1/Eu_0Pressure_clean.csv',
                      'Activity-Data/3412/Esc_up/2/Eu_1Pressure_clean.csv',
                      'Activity-Data/3412/Esc_up/3/Eu_2Pressure_clean.csv',
                      'Activity-Data/3412/Esc_up/4/Eu_3Pressure_clean.csv',
                      'Activity-Data/3412/Esc_up/5/Eu_4Pressure_clean.csv',
                      'Activity-Data/3412/Esc_up/6/Eu_5Pressure_clean.csv']
                      
escalator_down_files = ['Activity-Data/Samsung/061217/Esc_down/Esc_down_1/Esc_down_1Pressure_clean.csv',
                        'Activity-Data/Samsung/061217/Esc_down/Esc_down_2/Esc_down_2Pressure_clean.csv',
                        'Activity-Data/Samsung/061217/Esc_down/3/Esc_d2Pressure_clean.csv',
                        'Activity-Data/2812/Escalator_down/1/Ed_1Pressure_clean.csv', 
                        'Activity-Data/2812/Escalator_down/2/Ed_2Pressure_clean.csv',
                        'Activity-Data/2812/Escalator_down/3/Ed_7Pressure_clean.csv',
                        'Activity-Data/2912/Escalator_down/1/Ed_1Pressure_clean.csv',
                        'Activity-Data/2912/Escalator_down/2/Ed_2Pressure_clean.csv',
                        'Activity-Data/2912/Escalator_down/3/Ed_3Pressure_clean.csv',
                        'Activity-Data/2912/Escalator_down/4/Ed_5Pressure_clean.csv',
                        'Activity-Data/2912/Escalator_down/5/Ed_8Pressure_clean.csv',
                        'Activity-Data/2912/Escalator_down/6/Ed_9Pressure_clean.csv',
                        'Activity-Data/3012/Escalator_down/1/Ed_0Pressure_clean.csv',
                        'Activity-Data/3112/Esc_down/1/Ed_0Pressure_clean.csv',
                        'Activity-Data/3112/Esc_down/2/Ed_1Pressure_clean.csv',
                        'Activity-Data/3112/Esc_down/3/Ed_5Pressure_clean.csv',
                        'Activity-Data/3212/Esc_down/1/Ed_3Pressure_clean.csv',
                        'Activity-Data/3212/Esc_down/2/Ed_4Pressure_clean.csv',
                        'Activity-Data/3212/Esc_down/3/Ed_8Pressure_clean.csv',
                        'Activity-Data/3212/Esc_down/4/Ed_13Pressure_clean.csv',
                        'Activity-Data/3212/Esc_down/5/Ed_15Pressure_clean.csv',
                        'Activity-Data/3212/Esc_down/6/Ed_18Pressure_clean.csv',
                        'Activity-Data/3312/Esc_down/1/Ed_0Pressure_clean.csv',
                        'Activity-Data/3312/Esc_down/2/Ed_3Pressure_clean.csv',
                        'Activity-Data/3312/Esc_down/3/Ed_5Pressure_clean.csv',
                        'Activity-Data/3312/Esc_down/4/Ed_7Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/1/Ed_0Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/10/Ed_9Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/2/Ed_1Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/3/Ed_2Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/4/Ed_3Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/5/Ed_4Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/6/Ed_5Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/7/Ed_6Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/8/Ed_7Pressure_clean.csv',
                        'Activity-Data/3412/Esc_down/9/Ed_8Pressure_clean.csv',
                        'Activity-Data/3512/Esc_down/1/Ed_0Pressure_clean.csv',
                        'Activity-Data/3512/Esc_down/2/Ed_1Pressure_clean.csv']
                        
lift_up_files = ['Activity-Data/1912/Lift_Up/1/Lift_up_1Pressure_clean.csv', 
                 'Activity-Data/1912/Lift_Up/2/Lift_up_4Pressure_clean.csv',
                 'Activity-Data/1912/Lift_Up/3/Lift_up_9Pressure_clean.csv', 
                 'Activity-Data/Samsung/061217/Lift_Up/1/Lift_Up_2Pressure_clean.csv',
                 'Activity-Data/Samsung/061217/Lift_Up/2/Lift_up_5Pressure_clean.csv',
                 'Activity-Data/2812/Lift_up/1/Lu_1Pressure_clean.csv',
                 'Activity-Data/2812/Lift_up/2/Lu_6Pressure_clean.csv',
                 'Activity-Data/2812/Lift_up/3/Lu_10Pressure_clean.csv',
                 'Activity-Data/2912/Lift_up/1/Lu_1Pressure_clean.csv',
                 'Activity-Data/2912/Lift_up/2/Lu_2Pressure_clean.csv',
                 'Activity-Data/2912/Lift_up/3/Lu_3Pressure_clean.csv',
                 'Activity-Data/2912/Lift_up/4/Lu_5Pressure_clean.csv',
                 'Activity-Data/3112/Lift_up/1/Lu_0Pressure_clean.csv',
                 'Activity-Data/3112/Lift_up/2/Lu_1Pressure_clean.csv',
                 'Activity-Data/3112/Lift_up/3/Lu_0Pressure_clean.csv',
                 'Activity-Data/3112/Lift_up/4/Lu_1Pressure_clean.csv',
                 'Activity-Data/3112/Lift_up/5/Lu_3Pressure_clean.csv',
                 'Activity-Data/3112/Lift_up/6/Lu_7Pressure_clean.csv',
                 'Activity-Data/3212/Lift_up/1/Lu_0Pressure_clean.csv',
                 'Activity-Data/3212/Lift_up/2/Lu_1Pressure_clean.csv',
                 'Activity-Data/3212/Lift_up/3/Lu_5Pressure_clean.csv',
                 'Activity-Data/3212/Lift_up/4/Lu_7Pressure_clean.csv',
                 'Activity-Data/3212/Lift_up/5/Lu_10Pressure_clean.csv',
                 'Activity-Data/3312/Lift_up/1/Lu_0Pressure_clean.csv',
                 'Activity-Data/3312/Lift_up/2/Lu_11Pressure_clean.csv',
                 'Activity-Data/3412/Lift_up/1/Lu_0Pressure_clean.csv',
                 'Activity-Data/3412/Lift_up/2/Lu-1Pressure_clean.csv',
                 'Activity-Data/3512/Lift_up/1/Lu_0Pressure_clean.csv',
                 'Activity-Data/3512/Lift_up/2/Lu_1Pressure_clean.csv',
                 'Activity-Data/3512/Lift_up/3/Lu_3Pressure_clean.csv',
                 'Activity-Data/3512/Lift_up/4/Lu_4Pressure_clean.csv']

lift_down_files = ['Activity-Data/1912/Lift_Down/1/Lift_down_9Pressure_clean.csv',
                   'Activity-Data/Samsung/061217/Lift_Down/1/Lift_Down_2Pressure_clean.csv',
                   'Activity-Data/Samsung/061217/Lift_Down/2/Lift_down_3Pressure_clean.csv',
                   'Activity-Data/Samsung/061217/Lift_Down/3/Lift_down_Pressure_clean.csv',
                   'Activity-Data/2812/Lift_down/1/Ld_1Pressure_clean.csv',
                   'Activity-Data/2812/Lift_down/2/Ld_4Pressure_clean.csv',
                   'Activity-Data/2812/Lift_down/3/Ld_8Pressure_clean.csv',
                   'Activity-Data/2912/Lift_down/1/Ld_1Pressure_clean.csv',
                   'Activity-Data/2912/Lift_down/2/Ld_3Pressure_clean.csv',
                   'Activity-Data/2912/Lift_down/3/Ld_7Pressure_clean.csv',
                   'Activity-Data/2912/Lift_down/4/Ld_8Pressure_clean.csv',
                   'Activity-Data/3012/Lift_down/1/Ld_0Pressure_clean.csv',
                   'Activity-Data/3112/Lift_down/1/Ld_0Pressure_clean.csv',
                   'Activity-Data/3112/Lift_down/2/Ld_1Pressure_clean.csv',
                   'Activity-Data/3112/Lift_down/3/Ld_3Pressure_clean.csv',
                   'Activity-Data/3212/Lift_down/1/Ld_0Pressure_clean.csv',
                   'Activity-Data/3212/Lift_down/2/Ld_1Pressure_clean.csv',
                   'Activity-Data/3312/Lift_down/1/Ld_0Pressure_clean.csv',
                   'Activity-Data/3312/Lift_down/2/Ld_2Pressure_clean.csv',
                   'Activity-Data/3412/Lift_down/1/Ld_0Pressure_clean.csv',
                   'Activity-Data/3412/Lift_down/2/Ld_1Pressure_clean.csv',
                   'Activity-Data/3412/Lift_down/3/Ld_3Pressure_clean.csv',
                   'Activity-Data/3412/Lift_down/4/Ld_4Pressure_clean.csv',
                   'Activity-Data/3412/Lift_down/5/Ld-2Pressure_clean.csv',
                   'Activity-Data/3512/Lift_down/1/Ld_0Pressure_clean.csv',
                   'Activity-Data/3512/Lift_down/2/Ld_1Pressure_clean.csv',
                   'Activity-Data/3512/Lift_down/3/Ld_2Pressure_clean.csv',
                   'Activity-Data/3512/Lift_down/4/Ld_3Pressure_clean.csv',
                   'Activity-Data/3512/Lift_down/5/Ld_4Pressure_clean.csv',
                   'Activity-Data/3512/Lift_down/6/Ld_5Pressure_clean.csv',
                   'Activity-Data/3512/Lift_down/7/Ld_6Pressure_clean.csv',
                   'Activity-Data/3512/Lift_down/8/Ld_7Pressure_clean.csv']
                                                             
                   
                   
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
            window_features['ts_max'] = max(window['timestamp'])
            window_features['ts_min'] = min(window['timestamp'])
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
    w_frame = create_data_frame(w_files, sliding_window_interval, window_interval)
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


def create_dataset_direction(w_frame, su_frame, sd_frame, eu_frame, ed_frame, lu_frame, ld_frame):
    u_frame = pd.concat([su_frame, eu_frame, lu_frame])
    u_frame['label_direction'] = 0
    d_frame = pd.concat([sd_frame, ed_frame, ld_frame])
    d_frame['label_direction'] = 1
    return u_frame, d_frame
    
    
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
        
        
def visualize_direction_features(up_frame, down_frame):
    u_features_array = up_frame.as_matrix(columns=up_frame.columns)
    d_features_array = down_frame.as_matrix(columns=down_frame.columns)
    X = np.concatenate([u_features_array, d_features_array])
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
        ax1.scatter(windows_map[combo[0]]['values'][:len(up_frame)], windows_map[combo[1]]['values'][:len(up_frame)], c='r', alpha=0.5, label='up')
        ax1.scatter(windows_map[combo[0]]['values'][len(up_frame):], windows_map[combo[1]]['values'][len(up_frame):], c='b', alpha=0.5, label='down')
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
    
    
def create_direction_dataset(up_frame, down_frame):
    u_features_array = up_frame.as_matrix(columns=up_frame.columns)
    d_features_array = down_frame.as_matrix(columns=down_frame.columns)
    X = np.concatenate([u_features_array, d_features_array])
    Y = X[:,8]
    X = X[:,:7]
    return X, Y
    

def classify_logistic_regression(XTrain, XTest, YTrain, YTest):
    YPred = LogisticRegressionCV(solver = 'liblinear').fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall

    
def classify_sgd(XTrain, XTest, YTrain, YTest):
    YPred = SGDClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    

def classify_svm(XTrain, XTest, YTrain, YTest):
    YPred = svm.SVC().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    

def classify_linearSVM(XTrain, XTest, YTrain, YTest):
    YPred = svm.LinearSVC().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    

def classify_knn(XTrain, XTest, YTrain, YTest):
    YPred = KNeighborsClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    

def classify_gaussian_process_classifier(XTrain, XTest, YTrain, YTest):
    YPred = GaussianProcessClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    
    
def classify_naive_bayes(XTrain, XTest, YTrain, YTest):
    YPred = GaussianNB().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    

def classify_decision_tree(XTrain, XTest, YTrain, YTest):
    YPred = tree.DecisionTreeClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    

def classify_random_forest(XTrain, XTest, YTrain, YTest):
    YPred = RandomForestClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall 
    

def classify_ada_boost(XTrain, XTest, YTrain, YTest):
    YPred = AdaBoostClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    

def classify_gradient_boost(XTrain, XTest, YTrain, YTest):
    YPred = GradientBoostingClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    

def classify_neural_network(XTrain, XTest, YTrain, YTest):
    YPred = MLPClassifier().fit(XTrain, YTrain).predict(XTest)
    acc = accuracy_score(YTest, YPred)
    f1 = f1_score(YTest, YPred)
    prec = precision_score(YTest, YPred)
    recall = recall_score(YTest, YPred)
    return acc,f1, prec, recall
    

def stratified_KFoldVal(X, Y, classify, n_folds=10):
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    skf = StratifiedKFold(n_folds)
    for train, test in skf.split(X,Y):
        XTrain, XTest, YTrain, YTest = X[train], X[test], Y[train], Y[test]
        acc, f1, prec, recall = classify(XTrain, XTest, YTrain, YTest)
        f1_scores.append(f1)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(recall)
    return f1_scores, accuracies, precisions, recalls
    

def KFoldVal(X, Y, classify, n_folds=10):
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    kf = KFold(n_folds)
    for train, test in kf.split(X):
        XTrain, XTest, YTrain, YTest = X[train], X[test], Y[train], Y[test]
        acc, f1, prec, recall = classify(XTrain, XTest, YTrain, YTest)
        f1_scores.append(f1)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(recall)
    return f1_scores, accuracies, precisions, recalls


def evaluate_stratified_ml_algorithms(X, Y, folds=10):
    acc_skf_lr, f_skf_lr, p_skf_lr, r_skf_lr = stratified_KFoldVal(X, Y, classify_logistic_regression, folds)
    acc_skf_sgd, f_skf_sgd, p_skf_sgd, r_skf_sgd = stratified_KFoldVal(X, Y, classify_sgd, folds)
    acc_skf_svm, f_skf_svm, p_skf_svm, r_skf_svm = stratified_KFoldVal(X, Y, classify_svm, folds)
    acc_skf_l_svm, f_skf_l_svm, p_skf_l_svm, r_skf_l_svm = stratified_KFoldVal(X, Y, classify_linearSVM, folds)
    acc_skf_knn, f_skf_knn, p_skf_knn, r_skf_knn = stratified_KFoldVal(X, Y, classify_knn, folds)
    acc_skf_gpc, f_skf_gpc, p_skf_gpc, r_skf_gpc = stratified_KFoldVal(X, Y, classify_gaussian_process_classifier, folds)
    acc_skf_nb, f_skf_nb, p_skf_nb, r_skf_nb = stratified_KFoldVal(X, Y, classify_naive_bayes, folds)
    acc_skf_dt, f_skf_dt, p_skf_dt, r_skf_dt = stratified_KFoldVal(X, Y, classify_decision_tree, folds)
    acc_skf_rf, f_skf_rf, p_skf_rf, r_skf_rf = stratified_KFoldVal(X, Y, classify_random_forest, folds)
    acc_skf_ab, f_skf_ab, p_skf_ab, r_skf_ab = stratified_KFoldVal(X, Y, classify_ada_boost, folds)
    acc_skf_gb, f_skf_gb, p_skf_gb, r_skf_gb = stratified_KFoldVal(X, Y, classify_gradient_boost, folds)
    acc_skf_nn, f_skf_nn, p_skf_nn, r_skf_nn = stratified_KFoldVal(X, Y, classify_neural_network, folds)
    
    acc_list = [acc_skf_lr, acc_skf_sgd, acc_skf_svm, acc_skf_l_svm, acc_skf_knn, acc_skf_gpc, acc_skf_nb, 
                acc_skf_dt, acc_skf_rf, acc_skf_ab, acc_skf_gb, acc_skf_nn]
    f_list = [f_skf_lr, f_skf_sgd, f_skf_svm, f_skf_l_svm, f_skf_knn, f_skf_gpc, f_skf_nb, 
                f_skf_dt, f_skf_rf, f_skf_ab, f_skf_gb, f_skf_nn]
    p_list = [p_skf_lr, p_skf_sgd, p_skf_svm, p_skf_l_svm, p_skf_knn, p_skf_gpc, p_skf_nb, 
                p_skf_dt, p_skf_rf, p_skf_ab, p_skf_gb, p_skf_nn]
    r_list = [r_skf_lr, r_skf_sgd, r_skf_svm, r_skf_l_svm, r_skf_knn, r_skf_gpc, r_skf_nb, 
                r_skf_dt, r_skf_rf, r_skf_ab, r_skf_gb, r_skf_nn]
    avg_acc_list = list(map(avg, acc_list))
    avg_f_list = list(map(avg, f_list))
    avg_p_list = list(map(avg, p_list))
    avg_r_list = list(map(avg, r_list))
    labels_list = ["LR", "SGD", "SVM", "L-SVM", "KNN", "GPC", "NB", 
                "DT", "RF", "AB", "GB", "NN"]
    return avg_acc_list, avg_f_list, labels_list, avg_p_list, avg_r_list
    
        
def avg(lst):
    return sum(lst)/len(lst)
    
    
def plot_performance(avg_acc, avg_f, l_list):
    N = len(avg_f)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.4       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, avg_acc, width, color='r')

    rects2 = ax.bar(ind + width, avg_f, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Scores by algorithm')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(l_list)

    ax.legend((rects1[0], rects2[0]), ('Accuracy', 'F-Score'), loc='lower right')

    plt.show()


def plot_performance_all(avg_acc, avg_f, l_list, avg_p, avg_r):
    N = len(avg_f)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.2       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, avg_acc, width, color='r')

    rects2 = ax.bar(ind + width, avg_f, width, color='g')
    rects3 = ax.bar(ind + 2*width, avg_p, width, color='b')
    rects4 = ax.bar(ind + 3*width, avg_r, width, color='y')

    # add some text for labels, title and axes ticks
#     ax.set_yticks(70,100)
    ax.set_ylabel('Scores')
    ax.set_title('Scores by algorithm')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(l_list)
    ax.set_ylim([0.7, 1.0])

    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Accuracy', 'F-Score', 'Precision', 'Recall'), loc='lower right')
    plt.grid(True)
    plt.show()

climbing_files_acc = ['Activity-Data/Samsung/Climbing_Up/Climb_Up_1/Climb_Up_1Acceleration.csv',
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_2/Climb_Up_2Acceleration.csv',    
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_3/Climb_Up_3Acceleration.csv',
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_4/Climb_Up_4Acceleration.csv',    
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_6/Climb_Up_6Acceleration.csv',
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_7/Climb_Up_7Acceleration.csv', 
                 'Activity-Data/Samsung/Climbing_Up/Climb_Up_8/Climb_Up_8Acceleration.csv',
                 'Activity-Data/Climbing_Stairs/1/Climbing_Stairs1Acceleration.csv', 
                 'Activity-Data/Climbing_Stairs/2/Climbing_Stairs2Acceleration.csv', 
                 'Activity-Data/Climbing_Stairs/3/Climbing_Stairs3Acceleration.csv',
                 'Activity-Data/Climbing_Stairs/5/Climbing_Stairs5Acceleration.csv',
                 'Activity-Data/Climbing_Stairs/6/Climbing_Stairs6Acceleration.csv', 
                 'Activity-Data/Climbing_Stairs/7/Climbing_Stairs7Acceleration.csv',
                 'Activity-Data/1912/Stairs_Up/1/Stairs_up_1Acceleration.csv', 
                 'Activity-Data/1912/Stairs_Up/2/Stairs_up_2Acceleration.csv',
                 'Activity-Data/1912/Stairs_Up/3/Stairs_up_3Acceleration.csv', 
                 'Activity-Data/1912/Stairs_Up/4/Stairs_up_4Acceleration.csv',
                 'Activity-Data/1912/Stairs_Up/5/Stairs_up_5Acceleration.csv', 
                 'Activity-Data/2012/Stairs_Up/1/Stairs_up_1Acceleration.csv', 
                 'Activity-Data/2012/Stairs_Up/2/Stairs_up_2Acceleration.csv', 
                 'Activity-Data/2012/Stairs_Up/3/Stairs_up_3Acceleration.csv', 
                 'Activity-Data/2012/Stairs_Up/4/Stairs_up_4Acceleration.csv', 
                 'Activity-Data/2012/Stairs_Up/5/Stairs_up_5Acceleration.csv',
                 'Activity-Data/2912/Stairs_up/1/Su_1Acceleration.csv',
                 'Activity-Data/3012/Stairs_up/1/Su_0Acceleration.csv',
                 'Activity-Data/3112/Stairs_up/1/Su_0Acceleration.csv',
                 'Activity-Data/3112/Stairs_up/2/Su_1Acceleration.csv',
                 'Activity-Data/3112/Stairs_up/3/Su_2Acceleration.csv',
                 'Activity-Data/3112/Stairs_up/4/Su_3Acceleration.csv',
                 'Activity-Data/3112/Stairs_up/5/Su_4Acceleration.csv',
                 'Activity-Data/3112/Stairs_up/6/Su_5Acceleration.csv',
                 'Activity-Data/3112/Stairs_up/7/Su_6Acceleration.csv',
                 'Activity-Data/3112/Stairs_up/8/Su_7Acceleration.csv',
                 'Activity-Data/3112/Stairs_up/9/Su_8Acceleration.csv',
                 'Activity-Data/3212/Stairs_up/1/Su_2Acceleration.csv',
                 'Activity-Data/3212/Stairs_up/2/Su_8Acceleration.csv',
                 'Activity-Data/3212/Stairs_up/3/Su_11Acceleration.csv',
                 'Activity-Data/3312/Stairs_up/1/Su_0Acceleration.csv',
                 'Activity-Data/3312/Stairs_up/2/Su_1Acceleration.csv',
                 'Activity-Data/3312/Stairs_up/3/Su_6Acceleration.csv',
                 'Activity-Data/3312/Stairs_up/4/Su_8Acceleration.csv',
                 'Activity-Data/3312/Stairs_up/5/Su_9Acceleration.csv',
                 'Activity-Data/3312/Stairs_up/6/Su_10Acceleration.csv',
                 'Activity-Data/3312/Stairs_up/7/Su_18Acceleration.csv',
                 'Activity-Data/3312/Stairs_up/8/Sup_4Acceleration.csv',
                 'Activity-Data/3412/Stairs_up/1/Su_0Acceleration.csv',
                 'Activity-Data/3412/Stairs_up/2/Su_1Acceleration.csv',
                 'Activity-Data/3412/Stairs_up/3/Su_2Acceleration.csv',
                 'Activity-Data/3412/Stairs_up/4/Su_3Acceleration.csv',
                 'Activity-Data/3412/Stairs_up/5/Su_4Acceleration.csv',
                 'Activity-Data/3412/Stairs_up/6/Su_5Acceleration.csv',
                 'Activity-Data/3412/Stairs_up/7/Su_6Acceleration.csv',
                 'Activity-Data/3512/Stairs_up/1/Su_0Acceleration.csv',
                 'Activity-Data/3512/Stairs_up/2/Su_2Acceleration.csv',
                 'Activity-Data/3512/Stairs_up/3/Su_0Acceleration.csv']
                 
downstairs_files_acc = ['Activity-Data/Samsung/Climbing_Down/Climb_Down_1/Climb_Down_1Acceleration.csv', 
                    'Activity-Data/Samsung/Climbing_Down/Climb_Down_2/Climb_Down_2Acceleration.csv',    
                    'Activity-Data/Samsung/Climbing_Down/Climb_Down_3/Climb_Down_3Acceleration.csv', 
                    'Activity-Data/Samsung/Climbing_Down/Climb_Down_6/Climb_Down_6Acceleration.csv',
                    'Activity-Data/Samsung/Climbing_Down/Climb_Down_8/Climb_Down_8Acceleration.csv', 
                    'Activity-Data/Samsung/Climbing_Down/Climb_Down_9/Climb_Down_9Acceleration.csv',
                    'Activity-Data/Downstairs/1/Downstairs1Acceleration.csv', 
                    'Activity-Data/Downstairs/2/Downstairs2Acceleration.csv', 
                    'Activity-Data/Downstairs/3/Downstairs3Acceleration.csv', 
                    'Activity-Data/Downstairs/4/Downstairs4Acceleration.csv',
                    'Activity-Data/Downstairs/5/Downstairs5Acceleration.csv', 
                    'Activity-Data/Downstairs/6/Downstairs6Acceleration.csv',
                    'Activity-Data/1912/Stairs_Down/1/Stairs_down_1Acceleration.csv',
                    'Activity-Data/1912/Stairs_Down/2/Stairs_down_2Acceleration.csv',
                    'Activity-Data/1912/Stairs_Down/3/Stairs_down_3Acceleration.csv',
                    'Activity-Data/1912/Stairs_Down/4/Stairs_down_4Acceleration.csv',
                    'Activity-Data/1912/Stairs_Down/5/Stairs_down_5Acceleration.csv',
                    'Activity-Data/2012/Stairs_Down/1/Stairs_down_1Acceleration.csv', 
                    'Activity-Data/2012/Stairs_Down/2/Stairs_down_2Acceleration.csv', 
                    'Activity-Data/2012/Stairs_Down/3/Stairs_down_3Acceleration.csv', 
                    'Activity-Data/2012/Stairs_Down/4/Stairs_down_4Acceleration.csv', 
                    'Activity-Data/2012/Stairs_Down/5/Stairs_down_5Acceleration.csv',
                    'Activity-Data/2812/Stairs_down/1/Sd_3Acceleration.csv',
                    'Activity-Data/2812/Stairs_down/2/Sd_4Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/1/Su_1Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/10/Sup_5Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/11/Sup_6Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/12/SUP_1Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/2/Su_2Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/3/Su_3Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/4/Su_4Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/5/Su_5Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/6/Su_6Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/7/Sup2Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/8/Sup3Acceleration.csv',
                    'Activity-Data/2812/Stairs_up/9/Sup4Acceleration.csv',
                    'Activity-Data/2912/Stairs_down/1/Sd_2Acceleration.csv',
                    'Activity-Data/3012/Stairs_down/1/Sd_0Acceleration.csv',
                    'Activity-Data/3112/Stairs_down/1/Sd_0Acceleration.csv',
                    'Activity-Data/3112/Stairs_down/2/Sd_1Acceleration.csv',
                    'Activity-Data/3112/Stairs_down/3/Sd_2Acceleration.csv',
                    'Activity-Data/3112/Stairs_down/4/Sd_3Acceleration.csv',
                    'Activity-Data/3112/Stairs_down/5/Sd_5Acceleration.csv',
                    'Activity-Data/3112/Stairs_down/6/Sd_6Acceleration.csv',
                    'Activity-Data/3112/Stairs_down/7/Sd_3Acceleration.csv',
                    'Activity-Data/3212/Stairs_down/1/Sd_7Acceleration.csv',
                    'Activity-Data/3212/Stairs_down/2/Sd_9Acceleration.csv',
                    'Activity-Data/3212/Stairs_down/3/Sd_10Acceleration.csv',
                    'Activity-Data/3212/Stairs_down/4/Sd_15Acceleration.csv',
                    'Activity-Data/3312/Stairs_down/1/Sd_1Acceleration.csv',
                    'Activity-Data/3312/Stairs_down/2/Sd_4Acceleration.csv',
                    'Activity-Data/3312/Stairs_down/3/Sd_8Acceleration.csv',
                    'Activity-Data/3312/Stairs_down/4/Sd_10Acceleration.csv',
                    'Activity-Data/3312/Stairs_down/5/Sd_16Acceleration.csv',
                    'Activity-Data/3312/Stairs_down/6/Sd_18Acceleration.csv',
                    'Activity-Data/3312/Stairs_down/7/Sd_20Acceleration.csv',
                    'Activity-Data/3412/Stairs_down/1/Sd_0Acceleration.csv',
                    'Activity-Data/3412/Stairs_down/2/Sd_1Acceleration.csv',
                    'Activity-Data/3412/Stairs_down/3/Sd_2Acceleration.csv',
                    'Activity-Data/3412/Stairs_down/4/Sd_3Acceleration.csv',
                    'Activity-Data/3512/Stairs_down/1/Sd_0Acceleration.csv',
                    'Activity-Data/3512/Stairs_down/2/Sd_1Acceleration.csv',
                    'Activity-Data/3512/Stairs_down/3/Sd_2Acceleration.csv',
                    'Activity-Data/3512/Stairs_down/4/Sd_3Acceleration.csv',
                    'Activity-Data/3512/Stairs_down/5/Sd_4Acceleration.csv',
                    'Activity-Data/3512/Stairs_down/6/Sd_0Acceleration.csv',
                    'Activity-Data/3512/Stairs_down/7/Sd_1Acceleration.csv'
                   ]
                   
escalator_up_files_acc = ['Activity-Data/1912/Esc_Up/1/Esc_up_1Acceleration.csv',
                      'Activity-Data/1912/Esc_Up/2/Esc_up_2Acceleration.csv',
                      'Activity-Data/1912/Esc_Up/3/Esc_up_3Acceleration.csv',
                      'Activity-Data/1912/Esc_Up/4/Esc_Up_4Acceleration.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/Esc_Up_1/Esc_Up_1Acceleration.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/Esc_Up_2/Esc_Up_2Acceleration.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/3/Esc_Up_3Acceleration.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/4/Esc_up_4Acceleration.csv',
                      'Activity-Data/Samsung/061217/Esc_Up/5/Esc_up_5Acceleration.csv', 
                      'Activity-Data/2012/Esc_Up/1/Esc_up_1Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/1/Eu_1Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/10/Eu_12Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/11/Eu_13Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/12/Eu_14Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/2/Eu_2Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/3/Eu_3Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/4/Eu_4Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/5/Eu_5Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/6/Eu_6Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/7/Eu_7Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/8/Eu_8Acceleration.csv',
                      'Activity-Data/2812/Escalator_up/9/Eu_11Acceleration.csv',
                      'Activity-Data/2912/Escalator_up/1/Eu_1Acceleration.csv',
                      'Activity-Data/2912/Escalator_up/2/Eu_2Acceleration.csv',
                      'Activity-Data/2912/Escalator_up/3/Eu_5Acceleration.csv',
                      'Activity-Data/2912/Escalator_up/4/Eu_6Acceleration.csv',
                      'Activity-Data/2912/Escalator_up/5/Eu_7Acceleration.csv',
                      'Activity-Data/3012/Escalator_up/1/Eu_0Acceleration.csv',
                      'Activity-Data/3012/Escalator_up/2/Eu_2Acceleration.csv',
                      'Activity-Data/3012/Escalator_up/3/Eu_3Acceleration.csv',
                      'Activity-Data/3112/Esc_up/1/Eu_0Acceleration.csv',
                      'Activity-Data/3112/Esc_up/2/Eu_2Acceleration.csv',
                      'Activity-Data/3112/Esc_up/3/Eu_5Acceleration.csv',
                      'Activity-Data/3112/Esc_up/4/Eu_6Acceleration.csv',
                      'Activity-Data/3412/Esc_up/1/Eu_0Acceleration.csv',
                      'Activity-Data/3412/Esc_up/2/Eu_1Acceleration.csv',
                      'Activity-Data/3412/Esc_up/3/Eu_2Acceleration.csv',
                      'Activity-Data/3412/Esc_up/4/Eu_3Acceleration.csv',
                      'Activity-Data/3412/Esc_up/5/Eu_4Acceleration.csv',
                      'Activity-Data/3412/Esc_up/6/Eu_5Acceleration.csv']
                      
escalator_down_files_acc = ['Activity-Data/Samsung/061217/Esc_down/Esc_down_1/Esc_down_1Acceleration.csv',
                        'Activity-Data/Samsung/061217/Esc_down/Esc_down_2/Esc_down_2Acceleration.csv',
                        'Activity-Data/Samsung/061217/Esc_down/3/Esc_d2Acceleration.csv',
                        'Activity-Data/2812/Escalator_down/1/Ed_1Acceleration.csv', 
                        'Activity-Data/2812/Escalator_down/2/Ed_2Acceleration.csv',
                        'Activity-Data/2812/Escalator_down/3/Ed_7Acceleration.csv',
                        'Activity-Data/2912/Escalator_down/1/Ed_1Acceleration.csv',
                        'Activity-Data/2912/Escalator_down/2/Ed_2Acceleration.csv',
                        'Activity-Data/2912/Escalator_down/3/Ed_3Acceleration.csv',
                        'Activity-Data/2912/Escalator_down/4/Ed_5Acceleration.csv',
                        'Activity-Data/2912/Escalator_down/5/Ed_8Acceleration.csv',
                        'Activity-Data/2912/Escalator_down/6/Ed_9Acceleration.csv',
                        'Activity-Data/3012/Escalator_down/1/Ed_0Acceleration.csv',
                        'Activity-Data/3112/Esc_down/1/Ed_0Acceleration.csv',
                        'Activity-Data/3112/Esc_down/2/Ed_1Acceleration.csv',
                        'Activity-Data/3112/Esc_down/3/Ed_5Acceleration.csv',
                        'Activity-Data/3212/Esc_down/1/Ed_3Acceleration.csv',
                        'Activity-Data/3212/Esc_down/2/Ed_4Acceleration.csv',
                        'Activity-Data/3212/Esc_down/3/Ed_8Acceleration.csv',
                        'Activity-Data/3212/Esc_down/4/Ed_13Acceleration.csv',
                        'Activity-Data/3212/Esc_down/5/Ed_15Acceleration.csv',
                        'Activity-Data/3212/Esc_down/6/Ed_18Acceleration.csv',
                        'Activity-Data/3312/Esc_down/1/Ed_0Acceleration.csv',
                        'Activity-Data/3312/Esc_down/2/Ed_3Acceleration.csv',
                        'Activity-Data/3312/Esc_down/3/Ed_5Acceleration.csv',
                        'Activity-Data/3312/Esc_down/4/Ed_7Acceleration.csv',
                        'Activity-Data/3412/Esc_down/1/Ed_0Acceleration.csv',
                        'Activity-Data/3412/Esc_down/10/Ed_9Acceleration.csv',
                        'Activity-Data/3412/Esc_down/2/Ed_1Acceleration.csv',
                        'Activity-Data/3412/Esc_down/3/Ed_2Acceleration.csv',
                        'Activity-Data/3412/Esc_down/4/Ed_3Acceleration.csv',
                        'Activity-Data/3412/Esc_down/5/Ed_4Acceleration.csv',
                        'Activity-Data/3412/Esc_down/6/Ed_5Acceleration.csv',
                        'Activity-Data/3412/Esc_down/7/Ed_6Acceleration.csv',
                        'Activity-Data/3412/Esc_down/8/Ed_7Acceleration.csv',
                        'Activity-Data/3412/Esc_down/9/Ed_8Acceleration.csv',
                        'Activity-Data/3512/Esc_down/1/Ed_0Acceleration.csv',
                        'Activity-Data/3512/Esc_down/2/Ed_1Acceleration.csv']
                        
lift_up_files_acc = ['Activity-Data/1912/Lift_Up/1/Lift_up_1Acceleration.csv', 
                 'Activity-Data/1912/Lift_Up/2/Lift_up_4Acceleration.csv',
                 'Activity-Data/1912/Lift_Up/3/Lift_up_9Acceleration.csv', 
                 'Activity-Data/Samsung/061217/Lift_Up/1/Lift_Up_2Acceleration.csv',
                 'Activity-Data/Samsung/061217/Lift_Up/2/Lift_up_5Acceleration.csv',
                 'Activity-Data/2812/Lift_up/1/Lu_1Acceleration.csv',
                 'Activity-Data/2812/Lift_up/2/Lu_6Acceleration.csv',
                 'Activity-Data/2812/Lift_up/3/Lu_10Acceleration.csv',
                 'Activity-Data/2912/Lift_up/1/Lu_1Acceleration.csv',
                 'Activity-Data/2912/Lift_up/2/Lu_2Acceleration.csv',
                 'Activity-Data/2912/Lift_up/3/Lu_3Acceleration.csv',
                 'Activity-Data/2912/Lift_up/4/Lu_5Acceleration.csv',
                 'Activity-Data/3112/Lift_up/1/Lu_0Acceleration.csv',
                 'Activity-Data/3112/Lift_up/2/Lu_1Acceleration.csv',
                 'Activity-Data/3112/Lift_up/3/Lu_0Acceleration.csv',
                 'Activity-Data/3112/Lift_up/4/Lu_1Acceleration.csv',
                 'Activity-Data/3112/Lift_up/5/Lu_3Acceleration.csv',
                 'Activity-Data/3112/Lift_up/6/Lu_7Acceleration.csv',
                 'Activity-Data/3212/Lift_up/1/Lu_0Acceleration.csv',
                 'Activity-Data/3212/Lift_up/2/Lu_1Acceleration.csv',
                 'Activity-Data/3212/Lift_up/3/Lu_5Acceleration.csv',
                 'Activity-Data/3212/Lift_up/4/Lu_7Acceleration.csv',
                 'Activity-Data/3212/Lift_up/5/Lu_10Acceleration.csv',
                 'Activity-Data/3312/Lift_up/1/Lu_0Acceleration.csv',
                 'Activity-Data/3312/Lift_up/2/Lu_11Acceleration.csv',
                 'Activity-Data/3412/Lift_up/1/Lu_0Acceleration.csv',
                 'Activity-Data/3412/Lift_up/2/Lu-1Acceleration.csv',
                 'Activity-Data/3512/Lift_up/1/Lu_0Acceleration.csv',
                 'Activity-Data/3512/Lift_up/2/Lu_1Acceleration.csv',
                 'Activity-Data/3512/Lift_up/3/Lu_3Acceleration.csv',
                 'Activity-Data/3512/Lift_up/4/Lu_4Acceleration.csv']

lift_down_files_acc = ['Activity-Data/1912/Lift_Down/1/Lift_down_9Acceleration.csv',
                   'Activity-Data/Samsung/061217/Lift_Down/1/Lift_Down_2Acceleration.csv',
                   'Activity-Data/Samsung/061217/Lift_Down/2/Lift_down_3Acceleration.csv',
                   'Activity-Data/Samsung/061217/Lift_Down/3/Lift_down_Acceleration.csv',
                   'Activity-Data/2812/Lift_down/1/Ld_1Acceleration.csv',
                   'Activity-Data/2812/Lift_down/2/Ld_4Acceleration.csv',
                   'Activity-Data/2812/Lift_down/3/Ld_8Acceleration.csv',
                   'Activity-Data/2912/Lift_down/1/Ld_1Acceleration.csv',
                   'Activity-Data/2912/Lift_down/2/Ld_3Acceleration.csv',
                   'Activity-Data/2912/Lift_down/3/Ld_7Acceleration.csv',
                   'Activity-Data/2912/Lift_down/4/Ld_8Acceleration.csv',
                   'Activity-Data/3012/Lift_down/1/Ld_0Acceleration.csv',
                   'Activity-Data/3112/Lift_down/1/Ld_0Acceleration.csv',
                   'Activity-Data/3112/Lift_down/2/Ld_1Acceleration.csv',
                   'Activity-Data/3112/Lift_down/3/Ld_3Acceleration.csv',
                   'Activity-Data/3212/Lift_down/1/Ld_0Acceleration.csv',
                   'Activity-Data/3212/Lift_down/2/Ld_1Acceleration.csv',
                   'Activity-Data/3312/Lift_down/1/Ld_0Acceleration.csv',
                   'Activity-Data/3312/Lift_down/2/Ld_2Acceleration.csv',
                   'Activity-Data/3412/Lift_down/1/Ld_0Acceleration.csv',
                   'Activity-Data/3412/Lift_down/2/Ld_1Acceleration.csv',
                   'Activity-Data/3412/Lift_down/3/Ld_3Acceleration.csv',
                   'Activity-Data/3412/Lift_down/4/Ld_4Acceleration.csv',
                   'Activity-Data/3412/Lift_down/5/Ld-2Acceleration.csv',
                   'Activity-Data/3512/Lift_down/1/Ld_0Acceleration.csv',
                   'Activity-Data/3512/Lift_down/2/Ld_1Acceleration.csv',
                   'Activity-Data/3512/Lift_down/3/Ld_2Acceleration.csv',
                   'Activity-Data/3512/Lift_down/4/Ld_3Acceleration.csv',
                   'Activity-Data/3512/Lift_down/5/Ld_4Acceleration.csv',
                   'Activity-Data/3512/Lift_down/6/Ld_5Acceleration.csv',
                   'Activity-Data/3512/Lift_down/7/Ld_6Acceleration.csv',
                   'Activity-Data/3512/Lift_down/8/Ld_7Acceleration.csv']
                   
walking_files_acc = ['Activity-Data/Samsung/Walk/Walk_N2/Walk_N2Acceleration.csv', 
                 'Activity-Data/Samsung/Walk/Walk_N_3/Walk_N_3Acceleration.csv', 
                 'Activity-Data/Samsung/Walk/Walk_N/Walk_NAcceleration.csv',
                 'Activity-Data/Samsung/Walk/Walk_P1/Walk_P1Acceleration.csv',
                 'Activity-Data/Samsung/Walk/Walk_P2/Walk_P2Acceleration.csv',
                 'Activity-Data/Walking/01/Walking01Acceleration.csv', 
                 'Activity-Data/Walking/2/Walking2Acceleration.csv',
                 'Activity-Data/Walking/3/Walking3Acceleration.csv',
                 'Activity-Data/Walking/04/Walking04Acceleration.csv',
                 'Activity-Data/Walking/5/Walking5Acceleration.csv',
                 'Activity-Data/1912/Walk/1/Walk_1Acceleration.csv',
                 'Activity-Data/1912/Walk/2/Walk_8Acceleration.csv',
                 'Activity-Data/2012/Walk/1/Walk_1Acceleration.csv', 
                 'Activity-Data/2012/Walk/2/Walk_2Acceleration.csv',
                 'Activity-Data/2012/Walk/3/Walk_3Acceleration.csv',
                 'Activity-Data/2012/Walk/4/Walk_4Acceleration.csv',
                 'Activity-Data/2012/Walk/5/Walk_5Acceleration.csv', 
                 'Activity-Data/2012/Walk/6/Walk_6Acceleration.csv',
                 'Activity-Data/2012/Walk/7/Walk7Acceleration.csv',
                 'Activity-Data/2012/Walk/8/Walk_8Acceleration.csv',
                 'Activity-Data/2812/Walking/1/W_1Acceleration.csv',
                 'Activity-Data/2812/Walking/2/W_3Acceleration.csv',
                 'Activity-Data/2812/Walking/3/W_4Acceleration.csv',
                 'Activity-Data/3012/Walk/1/W_0Acceleration.csv']
                        
def print_acc_characteristics(su_frame, sd_frame, eu_frame, ed_frame, lu_frame, ld_frame, w_frame):
    print("Stairs Up Frame")
    display(su_frame.head(), su_frame.tail())
    print("Stairs Down Frame")
    display(sd_frame.head(), sd_frame.tail())
    print("Escalator Up Frame")
    display(eu_frame.head(), eu_frame.tail())
    print("Escalator Down Frame")
    display(ed_frame.head(), ed_frame.tail())
    print("Elevator Up Frame")
    display(lu_frame.head(), lu_frame.tail())
    print("Elevator Down Frame")
    display(ld_frame.head(), ld_frame.tail())
    print("Walking Frame")
    display(w_frame.head(), w_frame.tail())
    
def create_acc_features_windows(windows_frames, percentile=50):
    windows_features_list = []
    for window in windows_frames:
        if len(window) <= 1:
            continue
        else:
            ## X Components
            skew_windows_x = skew(window['accelerationX'])

            window_norm_x = normalize(np.array(window['accelerationX']).reshape(1,-1))
            percentile_x = np.percentile(window_norm_x, percentile)

            q75_x, q25_x = np.percentile(window['accelerationX'], [75 ,25])
            iqr_x = q75_x - q25_x

            kurtosis_x = kurtosis(window['accelerationX'])

            std_deviation_x = np.std(window['accelerationX'])

            #derivative = compute_sum_derivative_window(window['pressure'], window['timestamp'])
            derivative_window_x = np.gradient(window['accelerationX'], window['timestamp'])
            derivative_x = sum(derivative_window_x)
            
            median_x = np.median(window['accelerationX'].values)
            window['acc_norm_x'] = window['accelerationX'].apply(lambda x: x-median_x)    
            norm_x = sum(window['acc_norm_x'])
#             print(skew_windows, percentile, iqr, kurtosis_w, std_deviation, derivative)
            window_features = pd.DataFrame()
            window_features['skewX'] = [skew_windows_x]
            window_features['percentileX'] = [percentile_x]
            window_features['iqrX'] = [iqr_x]
            window_features['kurtosisX'] = [kurtosis_x]
            window_features['std_deviationX'] = [std_deviation_x]
            window_features['derivativeX'] = [derivative_x]
#             print(window_features)
            window_features['normX'] = [norm_x]
    
            ## Y Components
            skew_windows_y = skew(window['accelerationY'])

            window_norm_y = normalize(np.array(window['accelerationY']).reshape(1,-1))
            percentile_y = np.percentile(window_norm_y, percentile)

            q75_y, q25_y = np.percentile(window['accelerationY'], [75 ,25])
            iqr_y = q75_y - q25_y

            kurtosis_y = kurtosis(window['accelerationY'])

            std_deviation_y = np.std(window['accelerationY'])

            #derivative = compute_sum_derivative_window(window['pressure'], window['timestamp'])
            derivative_window_y = np.gradient(window['accelerationY'], window['timestamp'])
            derivative_y = sum(derivative_window_y)
            
            median_y = np.median(window['accelerationY'].values)
            window['acc_norm_y'] = window['accelerationY'].apply(lambda x: x-median_y)    
            norm_y = sum(window['acc_norm_y'])
#             print(skew_windows, percentile, iqr, kurtosis_w, std_deviation, derivative)
#             window_features = pd.DataFrame()
            window_features['skewY'] = [skew_windows_y]
            window_features['percentileY'] = [percentile_y]
            window_features['iqrY'] = [iqr_y]
            window_features['kurtosisY'] = [kurtosis_y]
            window_features['std_deviationY'] = [std_deviation_y]
            window_features['derivativeY'] = [derivative_y]

            window_features['normY'] = [norm_y]
            
            ## Z Components
            skew_windows_z = skew(window['accelerationZ'])

            window_norm_z = normalize(np.array(window['accelerationZ']).reshape(1,-1))
            percentile_z = np.percentile(window_norm_z, percentile)

            q75_z, q25_z = np.percentile(window['accelerationZ'], [75 ,25])
            iqr_z = q75_z - q25_z

            kurtosis_z = kurtosis(window['accelerationZ'])

            std_deviation_z = np.std(window['accelerationZ'])

            #derivative = compute_sum_derivative_window(window['pressure'], window['timestamp'])
            derivative_window_z = np.gradient(window['accelerationZ'], window['timestamp'])
            derivative_z = sum(derivative_window_z)
            
            median_z = np.median(window['accelerationZ'].values)
            window['acc_norm_z'] = window['accelerationZ'].apply(lambda x: x-median_z)    
            norm_z = sum(window['acc_norm_z'])
#             print(skew_windows, percentile, iqr, kurtosis_w, std_deviation, derivative)
#             window_features = pd.DataFrame()
            window_features['skewZ'] = [skew_windows_z]
            window_features['percentileZ'] = [percentile_z]
            window_features['iqrZ'] = [iqr_z]
            window_features['kurtosisZ'] = [kurtosis_z]
            window_features['std_deviationZ'] = [std_deviation_z]
            window_features['derivativeZ'] = [derivative_z]
#             print(window_features)
            window_features['normZ'] = [norm_z]    
            window_features['ts_max'] = max(window['timestamp'])
            window_features['ts_min'] = min(window['timestamp'])

#             print(window_features)
            windows_features_list.append(window_features)
#     print(len(windows_features_list))
    df_features = pd.concat(windows_features_list)
    return df_features
    
def create_acc_data_frame(input_files, sliding_window_interval, window_length, header=0, usecols=[0,1,2,3,5]):
    frame_list = []
    for i_file in input_files:
        df = pd.read_csv(i_file, delimiter=',', header = header, skipinitialspace = True, usecols = usecols)
        df_windows = create_sliding_windows(df, sliding_window_interval, window_length)
        df_features = create_acc_features_windows(df_windows)
#     print(len(df), len(df_features))
        frame_list.append(df_features)
    out_frame = pd.DataFrame()
    out_frame = pd.concat(frame_list)
    out_frame = out_frame.reset_index(drop=True)
    return out_frame
    
def create_acc_features_from_files(sliding_window_interval, window_interval, 
                               su_files = climbing_files_acc, sd_files = downstairs_files_acc,
                               eu_files = escalator_up_files_acc, ed_files = escalator_down_files_acc,
                               lu_files = lift_up_files_acc, ld_files = lift_down_files_acc, w_files = walking_files_acc):
    su_frame = create_acc_data_frame(su_files, sliding_window_interval, window_interval)
    su_frame['label'] = 1
    sd_frame = create_acc_data_frame(sd_files, sliding_window_interval, window_interval)
    sd_frame['label'] = 2
    eu_frame = create_acc_data_frame(eu_files, sliding_window_interval, window_interval)
    eu_frame['label'] = 3
    ed_frame = create_acc_data_frame(ed_files, sliding_window_interval, window_interval)
    ed_frame['label'] = 4
    lu_frame = create_acc_data_frame(lu_files, sliding_window_interval, window_interval)
    lu_frame['label'] = 5
    ld_frame = create_acc_data_frame(ld_files, sliding_window_interval, window_interval)
    ld_frame['label'] = 6
    w_frame = create_acc_data_frame(w_files, sliding_window_interval, window_interval)
    w_frame['label'] = 0
    return su_frame, sd_frame, eu_frame, ed_frame, lu_frame, ld_frame, w_frame
    
def create_dataset_esc_stairs(su_frame, sd_frame, eu_frame, ed_frame):
    s_frame = pd.concat([su_frame, sd_frame])
    s_frame['label_es'] = 0
    e_frame = pd.concat([eu_frame, ed_frame])
    e_frame['label_es'] = 1
    return s_frame, e_frame
    
def visualize_esc_stairs_features(s_frame, e_frame, param):
    s_features_array = s_frame.as_matrix(columns=s_frame.columns)
    e_features_array = e_frame.as_matrix(columns=e_frame.columns)
    X = np.concatenate([s_features_array, e_features_array])
    print(X.shape)
    if param == 'x':
        windows_map = { 0:{'values':X[:,0], 'legend': 'skewness'}, 1:{'values':X[:,5], 'legend': 'gradient'},
               2:{'values':X[:,3], 'legend':'kurtosis'}, 3:{'values':X[:,4], 'legend':'std deviation'},
               4:{'values':X[:,1], 'legend':'percentile_windows'}, 5:{'values':X[:,2], 'legend': 'iqr'},
               6:{'values':X[:,6], 'legend':'norm'}, 7:{'values':X[:,22], 'legend':'labels'}}
#     print(len(combos))
    elif param == 'y':
        windows_map = { 0:{'values':X[:,7], 'legend': 'skewness'}, 1:{'values':X[:,12], 'legend': 'gradient'},
               2:{'values':X[:,10], 'legend':'kurtosis'}, 3:{'values':X[:,11], 'legend':'std deviation'},
               4:{'values':X[:,8], 'legend':'percentile_windows'}, 5:{'values':X[:,9], 'legend': 'iqr'},
               6:{'values':X[:,13], 'legend':'norm'}, 7:{'values':X[:,22], 'legend':'labels'}}
    elif param == 'z':
        windows_map = { 0:{'values':X[:,14], 'legend': 'skewness'}, 1:{'values':X[:,19], 'legend': 'gradient'},
               2:{'values':X[:,17], 'legend':'kurtosis'}, 3:{'values':X[:,18], 'legend':'std deviation'},
               4:{'values':X[:,15], 'legend':'percentile_windows'}, 5:{'values':X[:,16], 'legend': 'iqr'},
               6:{'values':X[:,20], 'legend':'norm'}, 7:{'values':X[:,22], 'legend':'labels'}}
    combos = list(combinations(list(range(len(windows_map.keys())-1)), 2))
    for combo in combos:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(windows_map[combo[0]]['values'][:len(s_frame)], windows_map[combo[1]]['values'][:len(s_frame)], c='r', alpha=0.5, label='stairs')
        ax1.scatter(windows_map[combo[0]]['values'][len(s_frame):], windows_map[combo[1]]['values'][len(s_frame):], c='b', alpha=0.5, label='escalator')
        ax = plt.subplot()
        ax.set_xlabel(windows_map[combo[0]]['legend'])
        ax.set_ylabel(windows_map[combo[1]]['legend'])
        ax.legend()
        ax.grid(True)
        plt.show()

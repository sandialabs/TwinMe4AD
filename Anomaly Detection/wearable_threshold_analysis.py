"""
Developer: Methun K.
Date: Nov, 2023
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
from scipy.stats import norm, skew, kurtosis,mode
import altair as alt
from collections import defaultdict
import os
import sys
sys.path.insert(0, '../Include/')

from filehandler import FileReader
from dataqueue import DataProcessingQueue
from TimeSeriesAnomaly import *
from common import *
from Multiprocess import *
from confusion_matrix import *

import argparse
from argparse import ArgumentParser
import json
import time

myFmt = mdates.DateFormatter('%H:%M')
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment=None
pd.options.display.max_rows=None
pd.options.display.max_columns=None
np.random.seed(1)

# python wearable_threshold_analysis.py --healthyRHR 100 --rangeRHR "[90, 110]" --window "[60, 120]" --fraction "[0.5, 1.0]" --nCols "[2,3,4]" --resultpath "./result/Dec10" --no-multiprocess --dttype 'real' --dpi 800 --event "UPLOTEB" --datapath '../../../Data/COVID-19-Wearables/data' 

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--datapath', type=str,
                        required=True,
                        default='./', 
                        help='Location of data files')
    parser.add_argument('--rangeRHR', type=json.loads,
                        required=True,
                        default="[90, 110]", 
                        help='RHR range for a sick person')
    parser.add_argument('--window', type=json.loads,
                        required=True,
                        default="[60, 120]", 
                        help='List of window size')
    parser.add_argument('--fraction', type=json.loads,
                        required=True,
                        default="[0.5, 1.0]", 
                        help='Sliding window: List of fraction of windows')
    parser.add_argument('--event', type=str,
                        required=True,
                        default='RPD', 
                        help='RPD:Create refined data and profile data, CHD: Compute Heillinger distance')
    parser.add_argument('--healthyRHR', type=int,
                        required=False,
                        default=100, 
                        help='Resting hear rate of a healthy person')
    parser.add_argument('--nCols', type=json.loads,
                        required=False,
                        default="[2]", 
                        help='Number of columns to consider for analysis')
    parser.add_argument('--resultpath', type=str,
                        required=False,
                        default=f'./result/{time.strftime("%b%d")}', 
                        help='Location of data files')
    parser.add_argument('--multiprocess', type=bool,
                        required=False,
                        default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Pass --no-multiprocess to stop, --multiprocess to start')
    parser.add_argument('--perturb', type=bool,
                        required=False,
                        default=False,
                        action=argparse.BooleanOptionalAction,
                        help='Pass --no-perturb to stop, --perturb to start')
    parser.add_argument('--epsilon', type=json.loads,
                        required=False,
                        default="[0.1, 0.5]", 
                        help='List of epsilons (noise)')
    parser.add_argument('--healthy', type=str, nargs='+',
                        required=False,
                        default="abc def", 
                        help='List of healthy users')
    parser.add_argument('--sick', type=str, nargs='+',
                        required=False,
                        default="abc def",  
                        help='List of sick users')
    
    
    args = parser.parse_args()
    print(args)
    
    return args

if __name__=="__main__":
    
    if len(sys.argv)<2:
        print('Need to pass arguments:\n\t1. ', flush=True)
        
    args = parse_arguments()
    print(args)
    
    DATA_PATH = args.datapath
    healthyRHR = args.healthyRHR
    rangeRHR = args.rangeRHR
    windows = args.window
    fractions = args.fraction
    action = args.event
    n_cols = args.nCols
    write_dir = args.resultpath
    is_mp = args.multiprocess
    is_purturb = args.perturb
    hUser = args.healthy
    sUser = args.sick
    epsList = args.epsilon
    
    rangeRHR[-1]=rangeRHR[-1]+1
    
    print(DATA_PATH, type(DATA_PATH) ,flush=True)
    print(healthyRHR, type(healthyRHR) ,flush=True)
    print(rangeRHR, type(rangeRHR) ,flush=True)
    print(windows, type(windows) ,flush=True)
    print(fractions, type(fractions) ,flush=True)
    print(action, type(action) ,flush=True)
    print(n_cols, type(n_cols) ,flush=True)
    print(write_dir, type(write_dir) ,flush=True)
    print(is_mp, type(is_mp) ,flush=True)
    print(is_purturb, type(is_purturb) ,flush=True)
    print(hUser, type(hUser), flush=True)
    print(sUser, type(sUser), flush=True)
    print(epsList, type(epsList), flush=True)

    if is_purturb:
        users = {'Healthy':hUser, 'Sick':sUser}
        epsilon = [f"epsilon_{str(x)[:4]}" for x in epsList]
        
        print(users, flush=True)
        print(epsilon, flush=True)
    
    # Run this command at first
    # RPD stands for creating Refined data (descriptive statistics using window and sliding) from raw data
    # and Profile data. Profile data contains list of user and the status of the user (sick or healthy). 
    #If the max(average RHR) > healthy RHR: the patient is sick, otherwise, healthy.
    # Command: python wearable_threshold_analysis.py --rangeRHR "[80, 120]" --window "[60, 120]" --fraction "[0.5, 1.0]" --no-multiprocess --event "RPD" --datapath '/Users/mkamruz/Public/SNL/Projects/Anomaly_detection/Code/ipredictome/Data/COVID-19-Wearables/data/'
    if action == "RPD":
        if is_purturb:
            data_paths = []
            for ut in users:
                for u in users[ut]:
                    if len(u)>0:
                        for eps in epsilon:
                            data_paths.append(os.path.join(DATA_PATH, ut, u, eps,'Data'))
                        
            run_refined_and_profile_data_purterbation(data_paths, 
                                                      rhr_range=rangeRHR, 
                                                      freq_range=windows, 
                                                      frac_range=fractions,
                                                      is_mp=is_mp)
        else:
            run_refined_and_profile_data(DATA_PATH, 
                                         rhr_range=rangeRHR, 
                                         freq_range=windows, 
                                         frac_range=fractions,
                                         is_mp=is_mp)
    # Run this command at second
    # CHD stands for Computing Hellinger Distance.
    # Command: python wearable_threshold_analysis.py --rangeRHR "[90, 110]" --window "[60]" --fraction "[0.5]" --nCols "[2]" --resultpath "./result/Feb29" --multiprocess --event "CHD" --datapath '/Users/mkamruz/Public/SNL/Projects/Anomaly_detection/Code/ipredictome/Data/COVID-19-Wearables/data/'
    elif action == "CHD":
        if is_purturb:
            data_paths = []
            write_paths = []
            for ut in users:
                for u in users[ut]:
                    if len(u)>0:
                        for eps in epsilon:
                            data_paths.append(os.path.join(DATA_PATH, ut, u, eps,'Data'))
                            write_paths.append(os.path.join(write_dir, ut, u, eps))
                            
            run_hellinger_distance_process_purterbation(rhr_range=rangeRHR,
                                           freqList=windows, 
                                           fracList=fractions, 
                                           colList=n_cols, 
                                           data_file_dirs=data_paths, 
                                           write_file_dirs=write_paths,
                                           is_mp=is_mp)
        else:
            run_hellinger_distance_process(rhr_range=rangeRHR,
                                       freqList=windows, 
                                       fracList=fractions, 
                                       colList=n_cols, 
                                       data_file_dir=DATA_PATH, 
                                       write_file_dir=write_dir,
                                       is_mp=is_mp)
        
    # Run this command to compute confusion matrix
    elif action == "CCM":
        # python wearable_threshold_analysis.py --healthyRHR 100 --rangeRHR "[90, 110]" --window "[60, 120]" --fraction "[0.5, 1.0]" --nCols "[2,3,4]" --resultpath "./result/Dec10" --no-multiprocess --event "CCM" --datapath '../../../Data/COVID-19-Wearables/data'
        thresholdRange = [round(x, 3) for x in np.arange(0.005, 0.021, 0.001)]
        reader = FileReader(filePath=DATA_PATH, 
                    freqInMinute=windows[0], 
                    slidingInMinute=int(windows[0]*fractions[0]), 
                    healthy_RHR_threshold=healthyRHR)
        user_list = reader.get_all_user()
        del [reader] 

        real_user = []
        synthetic_user = []
        for x in user_list:
            if "_Syn" in x:
                synthetic_user.append(x)
            else:
                real_user.append(x)

        real_syn_user = []
        for x in user_list:
            if x in real_user and x+"_Syn" in synthetic_user:
                real_syn_user.append(x)
                real_syn_user.append(x+"_Syn")

        cm = ConfusionMatrix(write_dir)

        if len(real_user)>0:
            print('Processing real data ...')
            df_confusion_real = cm.compute_population_confusion_matrix(windows, fractions, n_cols, rangeRHR, thresholdRange, real_user)
            output_dir = create_path(os.path.join(write_dir, 'result', 'confusion_matrix'))
            df_confusion_real.to_csv(os.path.join(output_dir, 
                                                  f'Confusion_matrix_Threshold_all_RHR_{rangeRHR[0]}_{rangeRHR[1]}_real.csv'), 
                                     index=False)
            
        if len(synthetic_user)>0:
            print('Processing synthetic data ...')
            df_confusion_syn = compute_population_confusion_matrix(resultpath, window, fraction, nCols, rhrRange, thresholdRange, synthetic_user)
            output_dir = create_path(os.path.join(resultpath, 'result', 'confusion_matrix'))
            df_confusion_syn.to_csv(os.path.join(output_dir, 
                                                 f'Confusion_matrix_Threshold_all_RHR_{rangeRHR[0]}_{rangeRHR[1]}_syn.csv'), 
                                    index=False)
            
        if len(real_syn_user)>0:
            print('Processing real and synthetic data ...')
            df_confusion_real_syn = compute_population_confusion_matrix(resultpath, window, fraction, nCols, rhrRange, thresholdRange, real_syn_user)
            output_dir = create_path(os.path.join(resultpath, 'result','confusion_matrix'))
            df_confusion_real_syn.to_csv(os.path.join(output_dir, 
                                                      f'Confusion_matrix_Threshold_all_RHR_{rangeRHR[0]}_{rangeRHR[1]}_real_syn.csv'), 
                                         index=False)
        
        print(flush=True)
        print(df_confusion_real.shape, df_confusion_syn.shape, df_confusion_real_syn.shape)
        print('Done')
        

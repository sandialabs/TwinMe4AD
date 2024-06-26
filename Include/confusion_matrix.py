"""
Developer: Methun K.
Date: May, 2024
"""

import pandas as pd
import numpy as np
import scipy.stats as st 
import statsmodels.stats.api as sms
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator
import matplotlib.dates as mdates
import seaborn as sns
import warnings
from scipy.stats import norm, skew, kurtosis,mode
import altair as alt
from collections import defaultdict
import os
import sys

from filehandler import FileReader
from dataqueue import DataProcessingQueue
from TimeSeriesAnomaly import *
from common import *
from Multiprocess import *

from IPython.display import clear_output
from matplotlib.transforms import ScaledTranslation

myFmt = mdates.DateFormatter('%H:%M')
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment=None
pd.options.display.max_rows=None
pd.options.display.max_columns=None
np.random.seed(1)

plt.rcParams['font.family'] = 'Times New Roman'

class ConfusionMatrix:
    def __init__(self, file_directory):
        self.read_dir = file_directory
        
    def getMetrics(self, tp, tn, fp, fn, precision=3):
        accuracy = 0 if (tp+tn)==0 else (tp+tn)/(tp+tn+fp+fn)
        precision = 0 if tp==0 else tp/(tp+fp)
        recall = 0 if tp==0 else tp/(tp+fn)
        f1_score = 0 if (precision*recall)==0 else 2 * (precision*recall) / (precision+recall)
        mcc_score = 0 if (tp*tn-fp*fn)==0 else np.sqrt(((tp*tn-fp*fn)**2)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        tpr = recall
        fnr = 0 if fn==0 else fn/(fn+tp)
        fpr = 0 if fp==0 else fp/(fp+tn)
        
        return [tp, tn, fp, fn, (tp+tn+fp+fn), accuracy, precision, recall, f1_score, mcc_score, tpr, fnr, fpr]
    
    def compute_confusion_matrix_for_W_S_C(self, freq, sliding, col, rhr, threshold, user_group):
        f = os.path.join(self.read_dir, f"{freq}", f"{sliding}", f"{col}")
        
        TP,TN,FP,FN=0,0,0,0
        
        all_users = []
        if os.path.exists(f):
            all_users = os.listdir(f)
            
        if len(all_users)>0:
            for fn in all_users:
                if f"_{rhr}.csv" in fn:
                    u = fn[:len(fn)-len(f"_{rhr}.csv")]
                    if u in user_group:
                        data = pd.read_csv(os.path.join(f, fn))
                        
                        rhr_type = data.Type.unique()[0]
                        th_type = 'H' if data.hellingerDistance.max()<=threshold else 'S'
                        
                        if (rhr_type, th_type)==('S','S'):
                            TP+=1
                        elif (rhr_type, th_type)==('S','H'):
                            FN+=1
                        elif (rhr_type, th_type)==('H','S'):
                            FP+=1
                        elif (rhr_type, th_type)==('H','H'):
                            TN+=1
                        
            
        return self.getMetrics(TP,TN,FP,FN)

    def compute_confusion_matrix_for_W_S_C_each_obs(self, freq, sliding, col, rhr, threshold, user_group):
        f = os.path.join(self.read_dir, f"{freq}", f"{sliding}", f"{col}")
        
        TP,TN,FP,FN=0,0,0,0
        
        all_users = []
        if os.path.exists(f):
            all_users = os.listdir(f)
            
        if len(all_users)>0:
            for fn in all_users:
                if f"_{rhr}.csv" in fn:
                    u = fn[:len(fn)-len(f"_{rhr}.csv")]
                    if u in user_group:
                        data = pd.read_csv(os.path.join(f, fn))
                        
                        TP += data[np.logical_and(data.avgRHR>rhr, data.hellingerDistance>threshold)].shape[0]
                        TN += data[np.logical_and(data.avgRHR<=rhr, data.hellingerDistance<=threshold)].shape[0]
                        FP += data[np.logical_and(data.avgRHR<=rhr, data.hellingerDistance>threshold)].shape[0]
                        FN += data[np.logical_and(data.avgRHR>rhr, data.hellingerDistance<=threshold)].shape[0]
                        
                        '''
                        rhr_type = data.Type.unique()[0]
                        th_type = 'H' if data.hellingerDistance.max()<threshold else 'S'
                        
                        if (rhr_type, th_type)==('S','S'):
                            TP+=1
                        elif (rhr_type, th_type)==('S','H'):
                            FN+=1
                        elif (rhr_type, th_type)==('H','S'):
                            FP+=1
                        elif (rhr_type, th_type)==('H','H'):
                            TN+=1
                        '''
            
        #print(TP, TN, FP, FN)
        return self.getMetrics(TP,TN,FP,FN)
    
    def compute_population_confusion_matrix(self, window, fraction, nCols, rhrRange, thresholdRange, user_group):
        tmpdf = []
        
        for freq in window:
            for frac in fraction:
                sliding = int(freq*frac)
                for col in nCols:
                    for rhr in rhrRange:
                        for th in thresholdRange:
                            t = [freq, sliding, col, rhr, th]
                            perfVal = self.compute_confusion_matrix_for_W_S_C(freq, sliding, col, rhr, th, user_group)
                            #perfVal = self.compute_confusion_matrix_for_W_S_C_each_obs(freq, sliding, col, rhr, th, user_group)
                            t.extend(perfVal)
                            tmpdf.append(t)
                        print(f"Freq={freq}, Sliding={sliding}, RHR={rhr} done", flush=True)
                        
        cols = ['Freq','Sliding','nCol','RHR','THRESHOLD','TP','TN','FP','FN',
                'Total','Accuracy','Precision','Recall','F1 score','MCC score', 
                'TPR','FNR','FPR']
        return pd.DataFrame(tmpdf, columns=cols)

    def compute_confusion_matrix_for_epsilon(self, freq, sliding, col, rhr, threshold, users, epsilons):
        
        TP,TN,FP,FN=0,0,0,0
        
        for ut in users:
            for u in users[ut]:
                f = os.path.join(self.read_dir, f"{ut}", f"{u}", f"{epsilons}", f"{freq}", f"{sliding}", f"{col}")
    
                all_users = []
                if os.path.exists(f):
                    all_users = os.listdir(f)
    
                if len(all_users)>0:
                    for fn in all_users:
                        if f"_{rhr}.csv" in fn:
                            data = pd.read_csv(os.path.join(f, fn))
    
                            rhr_type = data.Type.unique()[0]
                            th_type = 'H' if data.hellingerDistance.max()<=threshold else 'S'
    
                            if (rhr_type, th_type)==('S','S'):
                                TP+=1
                            elif (rhr_type, th_type)==('S','H'):
                                FN+=1
                            elif (rhr_type, th_type)==('H','S'):
                                FP+=1
                            elif (rhr_type, th_type)==('H','H'):
                                TN+=1
                        #else:
                        #    print(f"_{rhr}.csv")
                        #    print('No Data for', os.path.join(f, fn), 'in', fn)
                    
                else:
                    print(f'No record for {f}', flush=True)
                    
                #print(f'{ut}-{u} done.')
                        
            
        return self.getMetrics(TP,TN,FP,FN)
    
    def compute_population_confusion_matrix_for_epsilon(self, window, fraction, nCols, rhrRange, thresholdRange, users, epsilons):
        tmpdf = []
        
        for freq in window:
            for frac in fraction:
                sliding = int(freq*frac)
                for col in nCols:
                    for rhr in rhrRange:
                        for th in thresholdRange:
                            t = [freq, sliding, col, rhr, th]
                            perfVal = self.compute_confusion_matrix_for_epsilon(freq, sliding, col, rhr, th, users, epsilons)
                            t.extend(perfVal)
                            tmpdf.append(t)
                            
                        print(f"Freq={freq}, Sliding={sliding}, RHR={rhr} done", flush=True)
                        
        cols = ['Freq','Sliding','nCol','RHR','THRESHOLD','TP','TN','FP','FN',
                'Total','Accuracy','Precision','Recall','F1 score','MCC score', 'TPR','FNR','FPR']
        return pd.DataFrame(tmpdf, columns=cols)

print('Done')
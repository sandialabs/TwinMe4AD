"""
Developer: Methun K.
Date: Aug, 2023
"""

import multiprocessing as mp
from multiprocessing import Process

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
import json

import sys

from filehandler import FileReader
from TimeSeriesAnomaly import *
from common import *
from dataqueue import *

warnings.filterwarnings('ignore')
np.random.seed(1)

## ////////////////////////////////////////////////////////////////////////////////////////////////////////
def get_proc_count():
    return mp.cpu_count()

def get_curr_proc_id():
    return os.getpid()

def get_job_dist(jc):
    np = get_proc_count()
    print('Total process=', np,flush=True)
    print('Total jobs=', jc,flush=True)

    jobs_per_process = 1
    if np>1 and jc+1 >= np:
        jobs_per_process = round(jc/(np-1))
    return jobs_per_process

def load_from(file):
    with open(file) as f:
        data = json.loads(f.read())
    return data
## ////////////////////////////////////////////////////////////////////////////////////////////////////////
## ++++++++++++++++++++ Create refined and profile data from raw data file ++++++++++++++++++++++++++++++++
## ////////////////////////////////////////////////////////////////////////////////////////////////////////
def create_refined_and_profile_data(data_path, job_list):
    for healthyRHR, freq, sliding in job_list:
         
        print(f'Current process={get_curr_proc_id()}, RHR={healthyRHR}, Freq={freq}, Sliding={sliding}',flush=True)
        
        reader = FileReader(filePath=data_path, 
                            freqInMinute=freq, 
                            slidingInMinute=sliding, 
                            healthy_RHR_threshold=healthyRHR)
        reader.getLimitsByUserType()
        
        del [reader]
        
def create_refined_and_profile_data_perturbation(job_list):
    for data_path, healthyRHR, freq, sliding in job_list:
         
        print(f'Current process={get_curr_proc_id()}, DATA={data_path}, RHR={healthyRHR}, Freq={freq}, Sliding={sliding}',flush=True)
        
        reader = FileReader(filePath=data_path, 
                            freqInMinute=freq, 
                            slidingInMinute=sliding, 
                            healthy_RHR_threshold=healthyRHR)
        reader.getLimitsByUserType()
        
        del [reader]

def run_refined_and_profile_data(data_path, rhr_range=[], freq_range=[], frac_range=[], is_mp=False):
    job_list = []
    
    for rhr in range(rhr_range[0], rhr_range[1]):
        for freq in freq_range:
            for frac in frac_range:
                sliding = int(freq*frac)
                job_list.append((rhr, freq, sliding))
                
    if is_mp==True:
        np = get_proc_count()
        jobs_per_process = get_job_dist(len(job_list))
        print('Jobs per process=', jobs_per_process,flush=True)
        
        proc = []
        total_job = len(job_list)
        si = 0

        for _ in range(np):
            ei = min(si+jobs_per_process, len(job_list))

            if si>=ei:
                break

            p = Process(target=create_refined_and_profile_data, 
                        args=(data_path, job_list[si:ei]))
            si = ei
            proc.append(p)
            p.start()

        for p in proc:
            p.join()
    else:
        create_refined_and_profile_data(data_path, job_list)
        
def run_refined_and_profile_data_purterbation(data_paths, rhr_range=[], freq_range=[], frac_range=[], is_mp=False):
    job_list = []
    
    for data_path in data_paths:
        for rhr in range(rhr_range[0], rhr_range[1]):
            for freq in freq_range:
                for frac in frac_range:
                    sliding = int(freq*frac)
                    job_list.append((data_path, rhr, freq, sliding))
                
    if is_mp==True:
        np = get_proc_count()
        jobs_per_process = get_job_dist(len(job_list))
        print('Jobs per process=', jobs_per_process,flush=True)
        
        proc = []
        total_job = len(job_list)
        si = 0

        for _ in range(np):
            ei = min(si+jobs_per_process, len(job_list))

            if si>=ei:
                break
            
            p = Process(target=create_refined_and_profile_data_perturbation, 
                        args=([job_list[si:ei]]))
            si = ei
            proc.append(p)
            p.start()

        for p in proc:
            p.join()
    else:
        create_refined_and_profile_data_perturbation(job_list)


## ////////////////////////////////////////////////////////////////////////////////////////////////////////
## +++++++++++++++++++++++++++++++++++ Compute Heillinger distance  +++++++++++++++++++++++++++++++++++++++
## ////////////////////////////////////////////////////////////////////////////////////////////////////////
# Compute hellinger distance
def compute_hellinger_distance(filereader, user, uType, 
                               write_dir='./', 
                               ncol=2, 
                               rhr=100,
                               read_from_zip_file=False, 
                               base_size=128,
                               base_X=[]):
    
    fpath = os.path.join(write_dir,
                     f'{filereader.freqInMinute}', 
                     f'{filereader.slidingInMinute}', 
                     f'{ncol}')
    #print('1', flush=True)
    
    if not os.path.exists(fpath):
        fpath = create_path(fpath)

    saved_file_path = os.path.join(fpath, f'{user}_{rhr}.csv')
    if os.path.exists(saved_file_path):
        #print('Already exists: ',saved_file_path)
        return
    #print('2', flush=True)
    
    #limits = filereader.getLimitsByUserType('H')
    #print('21', limits[:ncol], flush=True)

    comp = Measurement()
    #print('22', flush=True)
    #base_X = comp.getRandomUniformValues(limits[:ncol], size=base_size)
    #print('23', flush=True)
    base_fmm, base_svm = comp.compute_FMM_SVM(base_X)
    #print("3", flush=True)
    
    df = filereader.getRHRData(user, read_from_zip_file)
    q = DataProcessingQueue(filereader.freqInMinute, filereader.slidingInMinute)
    tmp = []
    cols = ['minRHR', 'avgRHR', 'medRHR', 'maxRHR']
    #print("4", flush=True)
    
    for i in df.index:
        result = q.addRecord(list(df.iloc[i, :].values))
        if len(result)>0:
            #print(result)
            datetime = result.pop(0)
            #print(result[:ncol])
            newX = np.append(base_X, [result[:ncol]], axis=0)
            try:
                fmm, svm = comp.compute_FMM_SVM(newX)
                result = [user, uType, datetime] + result[:ncol]
                result.append(np.round(comp.compute_hellinger_distance(base_fmm, fmm), 4))
                result.append(np.round(comp.compute_angle_change(base_svm, svm), 4))
                tmp.append(result)
            except Exception as e:
                print(f'Error for {saved_file_path}', flush=True)
                print(f'Error for {uType} user {user}, at {datetime}, values=[{result[:ncol]}]', flush=True)
    
    #print("5", flush=True)
    df = pd.DataFrame(tmp, columns=['User', 'Type', 'datetime']+cols[:ncol]+['hellingerDistance', 'angleDistance'])
    
    #lock = mp.Lock()
    #lock.acquire()
    df.to_csv(saved_file_path, index=False)
    #lock.release()

    #print(f'Finished process={os. getpid()}, {self.freqInMinute}, {self.slidingInMinute}, {ncol}, {user}, {uType}', flush=True)
    #print("6", flush=True)
    #print(flush=True)
    
def hellinger_distance_process(job_list, read_dir, write_dir, obj_base_X):
    
    for freq, sliding, col, healthyRHR in job_list:
        #for healthyRHR in range(rhr_range[0], rhr_range[1]):
        
        reader = FileReader(filePath=read_dir, 
                            freqInMinute=freq, 
                            slidingInMinute=sliding, 
                            healthy_RHR_threshold=healthyRHR)

        data = reader.get_profile_data()
        nObs = np.array(obj_base_X[f"{healthyRHR}"])
        base_X = nObs[:int(len(nObs)*1.0),]

        for u, ut in zip(data.User, data.Type):
            try:
                compute_hellinger_distance(filereader=reader, 
                                           user=u, 
                                           uType=ut, 
                                           write_dir=write_dir, 
                                           ncol=col,
                                           rhr=healthyRHR, 
                                           base_X=base_X)
            except Exception as e:
                print(f'In hellinger_distance_process => Error for ({u}, {ut}) is: {e}', flush=True)

        print(f'Done for pid={os. getpid()}, window={freq}, sliding={sliding}, rhr={healthyRHR}', flush=True)

        #del[reader]
        
def run_hellinger_distance_process(rhr_range, freqList, fracList, colList, data_file_dir, write_file_dir, is_mp=False):
    
    job_list = []
    obj_base_X = load_from('../../../Data/COVID-19-Wearables/base_x_v3.json')
    #base_X = np.array(obj_base_X[f"{healthy_rhr}"])
    
    
    for freq in freqList:
        for frac in fracList:
            sliding = int(freq*frac)
            for col in colList:
                for healthyRHR in range(rhr_range[0], rhr_range[1]):
                    job_list.append((freq, sliding, col, healthyRHR))
                
    print('Multiprocerss=', is_mp)
    if is_mp == True:
        np = get_proc_count()
        jobs_per_process = get_job_dist(len(job_list))
        print('Jobs per process=', jobs_per_process,flush=True)
        
        proc = []
        total_job = len(job_list)
        si = 0
        pc = 1
        
        for _ in range(np):
            ei = min(si+jobs_per_process, len(job_list))
            
            if si>=ei:
                break

            p = Process(target=hellinger_distance_process, 
                        args=(job_list[si:ei], data_file_dir, write_file_dir, obj_base_X))
            
            print(f"Process {pc}, start: {si}, end={ei}")
            si = ei
            pc+=1
            
            proc.append(p)
            p.start()

        for p in proc:
            p.join()
    else:
        hellinger_distance_process(job_list, data_file_dir, write_file_dir, obj_base_X)

def hellinger_distance_process_purterbation(job_list, rhr_range, obj_base_X):
    
    for read_dir, write_dir, freq, sliding, col in job_list:
        for healthyRHR in range(rhr_range[0], rhr_range[1]):
        
            reader = FileReader(filePath=read_dir, 
                                freqInMinute=freq, 
                                slidingInMinute=sliding, 
                                healthy_RHR_threshold=healthyRHR)

            data = reader.get_profile_data()
            nObs = np.array(obj_base_X[f"{healthyRHR}"])
            base_X = nObs[:int(len(nObs)*1.0),]

            for u, ut in zip(data.User, data.Type):
                try:
                    compute_hellinger_distance(filereader=reader, 
                                               user=u, 
                                               uType=ut, 
                                               write_dir=write_dir, 
                                               ncol=col,
                                               rhr=healthyRHR,
                                               base_X=base_X)
                except Exception as e:
                    print(f'In hellinger_distance_process_purterbation => Error for ({u}, {ut}) is: {e}', flush=True)
                    
            print(f'Done for pid={os. getpid()}, window={freq}, sliding={sliding}, rhr={healthyRHR}', flush=True)

            del[reader]
            
def run_hellinger_distance_process_purterbation(rhr_range, freqList, fracList, colList, data_file_dirs, write_file_dirs, is_mp=False):
    
    job_list = []
    obj_base_X = load_from('../../../Data/COVID-19-Wearables/base_x_v3.json')
    #base_X = np.array(obj_base_X[f"{healthy_rhr}"])
    
    for i,d in enumerate(data_file_dirs):
        for freq in freqList:
            for frac in fracList:
                sliding = int(freq*frac)
                for col in colList:
                    job_list.append((d, write_file_dirs[i], freq, sliding, col))
                
    print('Multiprocerss=', is_mp)
    if is_mp == True:
        np = get_proc_count()
        jobs_per_process = get_job_dist(len(job_list))
        print('Jobs per process=', jobs_per_process,flush=True)
        
        proc = []
        total_job = len(job_list)
        si = 0
        pc = 1
        
        for _ in range(np):
            ei = min(si+jobs_per_process, len(job_list))
            
            if si>=ei:
                break

            p = Process(target=hellinger_distance_process_purterbation, 
                        args=(job_list[si:ei], rhr_range, obj_base_X))
            
            print(f"Process {pc}, start: {si}, end={ei}")
            si = ei
            pc+=1
            
            proc.append(p)
            p.start()

        for p in proc:
            p.join()
    else:
        hellinger_distance_process_purterbation(job_list, rhr_range, obj_base_X)
        
## ////////////////////////////////////////////////////////////////////////////////////////////////////////
##                                        Depricated methods 
## ////////////////////////////////////////////////////////////////////////////////////////////////////////

def splitHealthyAndSick(data_file_path, write_file_path, freq, healthyMaxRHR, sickMinRHR):
    
    if os.path.exists(os.path.join(write_file_path, f'user_profile_{freq}.csv')): 
        print(freq, 'already exists')
        return
        
    file_reader = FileReader(data_file_path, freq=freq)
    allUsers = file_reader.get_all_user()
    tmp = []

    for u in allUsers:
        data = file_reader.read_file(u)
        if data is None or data.shape[0]==0:
            continue

        minAvgRHR, maxAvgRHR = data.avgHR[data.totalSTEPS==0].min(), data.avgHR[data.totalSTEPS==0].max()
        minMedRHR, maxMedRHR = data.minHR[data.totalSTEPS==0].max(), data.medHR[data.totalSTEPS==0].max()
        
        if maxAvgRHR < healthyMaxRHR:
            tmp.append([u, 'healthy', minAvgRHR, maxAvgRHR, minMedRHR, maxMedRHR])
            
        elif maxAvgRHR > sickMinRHR:
            tmp.append([u, 'sick', minAvgRHR, maxAvgRHR, minMedRHR, maxMedRHR])
            
        else:
            tmp.append([u, 'Risky', minAvgRHR, maxAvgRHR, minMedRHR, maxMedRHR])
            
        del [data]

    data = pd.DataFrame(tmp, columns=['User', 'Type', 'minAvgRHR', 'maxAvgRHR', 'minMedRHR', 'maxMedRHR'])
    data.to_csv(os.path.join(write_file_path, f'user_profile_{freq}.csv'), index=False)
    
    del [tmp, data, file_reader, allUsers]
    print(freq, 'Done')

def run_splitHealthyAndSick(data_file_path, write_file_path, freqList, healthyMaxRHR, sickMinRHR):
    proc = []
    for freq in freqList:
        p = Process(target=splitHealthyAndSick, 
                    args=(data_file_path, write_file_path, freq, healthyMaxRHR, sickMinRHR))
        proc.append(p)
        p.start()
                
    for p in proc:
        p.join()
        
def analyzeHealthyAndSick(data_file_path, user_profile_path, write_path, freq, col_index, ci, colName, ylbl_list, title_list, base_size):
    print(f'Process={os. getpid()}, started freq={freq} and #columns={ci+1}', flush=True)
    
    data = pd.read_csv(os.path.join(user_profile_path, f'user_profile_{freq}.csv'))
    t = data.groupby('Type').apply(lambda x: 
                                   [[x['minAvgRHR'].min(), 
                                   x['maxAvgRHR'].max()],
                                   [x['minMedRHR'].min(), 
                                   x['maxMedRHR'].max()]]
                                  )
    data_limits = t['healthy']

    healthy = data.User[data.Type=='healthy'].values
    sick = data.User[data.Type=='sick'].values
    risky = data.User[data.Type=='Risky'].values
    
    print(freq, len(healthy), len(risky), len(sick))
    
    limits = []
    cNames = []
    yl = []
    title = []
    for i in range(ci+1):
        limits.append(data_limits[col_index[i]])
        cNames.append(colName[col_index[i]])
        yl.append(ylbl_list[col_index[i]])
        title.append(title_list[col_index[i]])
    
    comp = Measurement()
    file_reader = FileReader(data_file_path, freq=freq)
    
    base_X = comp.getRandomUniformValues(limits, size=base_size) #getRandomNormalValues(limits, rvs=100)
    base_fmm, base_svm = comp.compute_FMM_SVM(base_X)
    
    import matplotlib.pyplot as plt
    
    for utype, ulist in [('Healthy', healthy), ('Sick', sick), ('Risky', risky)]:
        print(f'Process={os. getpid()}, {utype}', flush=True)
        for u in ulist:
            
            if os.path.exists(os.path.join(write_path, freq, f'{len(cNames)}', f'{utype}_{u}.csv')) and os.path.exists(os.path.join(write_path, freq, f'{len(cNames)}', f'{utype}_{u}.png')):
                continue
                
            data = file_reader.read_file(u)
            data = data.loc[data.totalSTEPS==0.0 ,colName]
            data.reset_index(inplace=True, drop=True)

            X = data[cNames].values

            A=[]
            D=[]

            for i in range(X.shape[0]):
                newX = np.append(base_X, [X[i,:].tolist()], axis=0)
                fmm, svm = comp.compute_FMM_SVM(newX)
                D.append(np.round(comp.compute_hellinger_distance(base_fmm, fmm),4))
                A.append(np.round(comp.compute_angle_change(base_svm, svm), 3))

            #print(X.shape, len(D), len(A))

            fig, ax = plt.subplots(2+X.shape[1],1,figsize=(25,20), constrained_layout=True, dpi=200, sharex=True)    
            axi = 0

            ax[axi].plot(np.arange(0, X.shape[0]), np.array(D))
            ax[axi].set_ylabel('Hellinger Distance', fontsize=24)
            ax[axi].tick_params(axis="x", labelsize=20)
            ax[axi].tick_params(axis="y", labelsize=20)
            ax[axi].set_title(f'HD-{u}', fontsize=24)
            ax[axi].grid(True, color = 'green')
            axi+=1

            ax[axi].plot(np.arange(0, X.shape[0]), np.array(A))
            ax[axi].set_ylabel('Angle', fontsize=24)
            ax[axi].tick_params(axis="x", labelsize=20)
            ax[axi].tick_params(axis="y", labelsize=20)
            ax[axi].set_title(f'Angle', fontsize=24)
            ax[axi].grid(True, color = 'green')
            axi+=1

            while axi<X.shape[1]+2:
                #print(axi)
                ax[axi].plot(np.arange(0, X.shape[0]), X[:, axi-2])
                if axi==X.shape[1]+1: ax[axi].set_xlabel('Window Length', fontsize=24)
                ax[axi].set_ylabel(yl[axi-2], fontsize=24)
                ax[axi].tick_params(axis="x", labelsize=20)
                ax[axi].tick_params(axis="y", labelsize=20)
                ax[axi].set_title(title[axi-2], fontsize=24)
                ax[axi].grid(True, color = 'green')
                ax[axi].axhline(y = limits[axi-2][1], color = 'r', linestyle = '-')
                axi+=1

            tmp = pd.DataFrame({'avgRHR':X[:,0], 'HD':D, 'Angle':A})
            tmp.to_csv(os.path.join(write_path, freq, f'{X.shape[1]}', f'{utype}_{u}.csv'), index=False)
            del [tmp]

            plt.savefig(os.path.join(write_path, freq, f'{X.shape[1]}', f'{utype}_{u}.png'))
            plt.close()
            #plt.show()
            
    print(f'Process={os. getpid()}, finished freq={freq} and #columns={ci+1}', flush=True)

        
        
def run_analyzeHealthyAndSick(data_file_path, user_profile_path, freqList, write_path, 
                              col_index = [0,1,0,1,0,1], 
                              colName = ['avgHR', 'medHR'],
                              ylbl_list = ['Avg RHR', 'Median RHR'],
                              title_list = ['Avg RHR', 'Median RHR'],
                              base_size=94
                             ):
    
    proc = []
    for freq in freqList:
        for ci in range(1, len(col_index)):
            p = Process(target=analyzeHealthyAndSick, 
                        args=(data_file_path, user_profile_path, write_path, freq, col_index, ci, colName, ylbl_list, title_list, base_size))
            proc.append(p)
            p.start()
            
    for p in proc:
        p.join()
            
    print('Done')
    


    
def pfunc(a,b,c):
    print(a,b,c)
    

def run(ra=1, rb=1, rc=1):
    proc = []
    for a in range(ra):
        for b in range(rb):
            for c in range(rc):
                p = Process(target=pfunc, args=(a,b,c))  
                p.start()
                proc.append(p)
                
    for p in proc:
        p.join()
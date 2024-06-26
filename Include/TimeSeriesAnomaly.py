"""
Developer: Methun K.
Date: Aug, 2023
"""

import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm, mode
import os
import itertools
from sklearn.manifold import MDS

class TimeSeries():
    """Time series class to calculate moment tensor"""
    def __init__(self, time_series_data):
        self.time_series_data = time_series_data
        self.rescale()
        self.num_data_pts, self.num_features = self.time_series_data.shape
        
    def rescale(self):
        self.time_series_data = self.time_series_data/np.max(self.time_series_data, axis=0)
        
    def expect(self):
        return np.mean(self.time_series_data, axis=0)
    
    def covariance(self):
        return np.cov(self.time_series_data, romvar=False)
    
    def coskewness(self):
        data_minus_mean = self.time_series_data - np.mean(self.time_series_data, axis=0)
        coskewness = np.zeros(np.repeat(self.num_features,4))
        for i in range(self.num_features):
            for j in range(self.num_features):
                for k in range(self.num_features):
                    coskewness[i,j,k] = np.mean(np.prod(data_minus_mean[:,[i,j,k]], axis=1), axis=0)
        return coskewness            
    
    def cokurtosis(self):
        """Calculates the cokurtosis_tensor"""
        #data_minus_mean = (self.time_series_data - np.mean(self.time_series_data, axis=0))/np.max(self.time_series_data, axis=0)
        data_minus_mean = self.time_series_data - np.mean(self.time_series_data, axis=0)
        cokurtosis = np.zeros(np.repeat(self.num_features,4))
        for i in range(self.num_features):
            for j in range(self.num_features):
                for k in range(self.num_features):
                    for l in range(self.num_features):
                        cokurtosis[i,j,k,l] = np.mean(np.prod(data_minus_mean[:,[i,j,k,l]], axis=1), axis=0)\
                                            -np.mean(np.prod(data_minus_mean[:,[i,j]],axis=1),axis=0) * np.mean(np.prod(data_minus_mean[:,[k,l]],axis=1),axis=0)\
                                            -np.mean(np.prod(data_minus_mean[:,[i,k]],axis=1),axis=0) * np.mean(np.prod(data_minus_mean[:,[j,l]],axis=1),axis=0)\
                                            -np.mean(np.prod(data_minus_mean[:,[i,l]],axis=1),axis=0) * np.mean(np.prod(data_minus_mean[:,[j,k]],axis=1),axis=0)
                                            
        return cokurtosis
    
class AnomalyDetection():   
    def __init__(self, cokurtosis):
        self.cokurtosis = cokurtosis
        self.num_features = cokurtosis.shape[0]
        
    def hosvd(self):
        kurtosis_matrix = self.cokurtosis.reshape((self.num_features,self.num_features**3))
        u,s,vh = np.linalg.svd(kurtosis_matrix, full_matrices=True)
        return s,u
    
    def feature_moment_metric(self):
        sing_vals, sing_vecs = self.hosvd()
        #print(sing_vals, sing_vecs)
        feature_metric_num = np.multiply(sing_vals.repeat(self.num_features,).reshape(self.num_features,self.num_features),\
                                         sing_vecs*sing_vecs).sum(axis=1)
        feature_metric_den = sing_vals.sum()
        return feature_metric_num/feature_metric_den
    
class Measurement:
    def compute_FMM_SVM(self, X):
        T = TimeSeries(X)
        model = AnomalyDetection(T.cokurtosis())       
        fmm = list(model.feature_moment_metric())
        _,sing_vecs = model.hosvd()
        svm = sing_vecs[0,:].tolist()

        return fmm, svm
    
    def compute_hellinger_distance(self, FMM_last, FMM_this):
        return np.sqrt(np.sum(np.square(np.sqrt(FMM_this) - np.sqrt(FMM_last))))/np.sqrt(2)
    
    def compute_angle_change(self, svm_last, svm_this):
        return np.dot(svm_last,svm_this)/(np.linalg.norm(svm_last)*np.linalg.norm(svm_this))
    
    def getRandomNormalValues(self, limits, size=10000):
        ncols = len(limits)
        low, high = limits[0][0], limits[0][1]
        mu = round((high+low)/2)
        X = np.zeros((size, ncols))
        
        for i in range(ncols):
            X[:, i] = self.getRandomValuesFromNormalDist(limits[i], size)
            
        return X
    
    def getRandomValuesFromNormalDist(self, val_range, size=10000):
        a, b = val_range[0], val_range[1]
        if b<a:
            a,b=b,a
        if a==b:
            if -1<b<1:
                b+=0.1
            else:
                b+=1

        mu = a+(b-a)/2
        k = 0.5
        minD = float('+inf')
        optK = -1
        optV = []
        l =k*np.sqrt(np.std([a,b]))

        for _ in range(500):

            for _ in range(100):
                v = np.round(np.random.normal(loc=mu, scale=l, size=size), 2)

                ld = abs(min(v)-a)
                rd = abs(max(v)-b)

                if ld+rd<minD:
                    minD = ld+rd
                    optK = k
                    optV = list(v)

                if abs(min(v)-a)<1 and abs(max(v)-b)<1:
                    break

            v = np.array(optV)

            if abs(min(v)-a)<1 and abs(max(v)-b)<1:
                break
            else:
                k+=0.1

            l = k*np.sqrt(np.std(v))

        #print(minD, optK)
        #print(min(optV), max(optV))

        #if min(optV)>a:
        #    optV.append(a)

        #if max(optV)<b:
        #    optV.append(b)

        np.random.shuffle(optV)

        return optV
    
    def getRandomUniformValues(self, limits, epsilon=0.5, size=100):
        ncols = len(limits)
        
        X = np.zeros((size, ncols))
        for i in range(ncols):
            low, high = limits[i][0], limits[i][1]
            if low>high: 
                low, high=high, low

            high+=5.
            
            step = (high-low)/int(size*0.8)
            d = np.arange(low, high, step).tolist()
            d.extend(np.random.uniform(low=low, high=high, size=size-len(d)).tolist())
            np.random.shuffle(d)
            #print(len(d))
            X[:, i] = d
        
        X = np.round(X, 2)
        
        return X
    
    def get_healthy_observations(self, limits, offset=1):
        el = [[x for x in range(l[0], l[1]+1+max(offset, 1))] for l in limits]
        X = sorted([x for x in itertools.product(*el)])
        
        X = np.round(X, 2)
        
        return X

def get_hellinger_distance(M):
    dist = []
    i = 0
    F_i_n_1 = M[i,:]
    while i < M.shape[0]-1:
        #F_i_n_1 = M[i,:]
        F_i_n = M[i+1,:]
        val = np.sqrt(np.sum(np.square(np.sqrt(F_i_n) - np.sqrt(F_i_n_1))))/np.sqrt(2)
        dist.append(val)
        i = i+1
    return dist

def get_angle_change(M):
    angle = []
    i = 0
    a = M[i,:]
    while i < M.shape[0]-1:
        #a = M[i,:]
        b = M[i+1,:]
        val = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        angle.append(val)
        i = i+1
    return angle

def get_Dice_distance(M):
    dist = []
    i = 0
    while i < M.shape[0]-1:
        F_i_n_1 = M[i,:]
        F_i_n = M[i+1,:]
        if i<3:
            print(F_i_n_1)
            print(F_i_n)
        #val = np.sqrt(np.sum(np.square(np.sqrt(F_i_n) - np.sqrt(F_i_n_1))))/np.sqrt(2)
        val = similarity_metric_source_target_domains(F_i_n.T, F_i_n_1.T)
        dist.append(val)
        i = i+1
    return dist

def similarity_metric_source_target_domains(samps_s, samps_t, dim=1):
    # mean and variance of source data
    mu1, std1 = norm.fit(samps_s)
    var1 = np.power(std1, 2)
    mu1 = mu1/np.sqrt(dim)
 
    # mean and variance of target data
    mu2, std2 = norm.fit(samps_t)
    var2 = np.power(std2, 2)
    mu2 = mu2/np.sqrt(dim)
 
    numerator = 2.0 * np.sqrt(np.abs(var1 * var2))
    denominator = np.sqrt(np.abs(0.5 * (var1 + var2))) * (np.sqrt(np.abs(var1)) + np.sqrt(np.abs(var2)))
 
    sim_met = (numerator / denominator) * np.exp(-0.5 * (mu1 - mu2) * (1.0 / (var1 + var2)) * (mu1 - mu2).transpose())
    return sim_met

def hellinger_distance_plot(data, title='',user='',htype='', img_path='./', window_size=96, saveToFile=True):
    feature_matrix = []
    singular_vec_matrix = []
    start_idx = window_size
    X = data.values

    for idx in range(start_idx, X.shape[0]+1):
        T = TimeSeries(X[:idx,:])
        model = AnomalyDetection(T.cokurtosis())
        feature_matrix.append(list(model.feature_moment_metric()))
        _,sing_vecs = model.hosvd()
        singular_vec_matrix.append(sing_vecs[0,:].tolist())
        
    feature_matrix = np.array(feature_matrix)
    singular_vec_matrix = np.array(np.array(feature_matrix)) 
    A = get_angle_change(singular_vec_matrix)
    D = get_hellinger_distance(feature_matrix)
    #D = get_Dice_distance(feature_matrix)
    
    fig, ax = plt.subplots(2,1,figsize=(25,20), constrained_layout=True, dpi=200, sharex=True)

    ax[0].plot(np.arange(start_idx, X.shape[0]), np.array(D))
    ax[0].set_xlabel('Window Length', fontsize=24)
    ax[0].set_ylabel('Hellinger Distance', fontsize=24)
    ax[0].tick_params(axis="x", labelsize=20)
    ax[0].tick_params(axis="y", labelsize=20)
    ax[0].set_title(f'HD {title}', fontsize=24)
    ax[0].grid(True)
        
    ax[1].plot(np.arange(start_idx, X.shape[0]), np.array(A))
    ax[1].set_xlabel('Window Length', fontsize=24)
    ax[1].set_ylabel('Angle', fontsize=24)
    ax[1].tick_params(axis="x", labelsize=20)
    ax[1].tick_params(axis="y", labelsize=20)
    ax[1].set_title(f'Angle {title}', fontsize=24)
    ax[1].grid(True)
    
    if saveToFile:
        plt.savefig(os.path.join(img_path, f'{htype}_{user}.png'))
        plt.close()
    else:
        plt.show()
        
    tmp = pd.DataFrame({'Index':np.arange(start_idx, X.shape[0]), 'HD':np.array(D), 'Angle':np.array(A)})
    tmp.to_csv(os.path.join(img_path, f'{htype}_{user}_HD_Angle.csv'), index=False)
    
    if saveToFile == False:
        return tmp
    
def hellinger_distance_plot_V2(data, user='', htype='', img_path='./', saveToFile=True, maxWindow=5, threshold=90):
    feature_matrix = []
    singular_vec_matrix = []
    
    fmm,svm,_ = getInitialCKValue(data = data, maxWindow=maxWindow, threshold=threshold)
    feature_matrix.append(fmm)
    singular_vec_matrix.append(svm)
    
    X = data.values
    windows = []
    
    # Computation using available data
    start_idx = maxWindow
    for idx in range(start_idx, X.shape[0]+1):
        T = TimeSeries(X[:idx,:])
        model = AnomalyDetection(T.cokurtosis())
        feature_matrix.append(list(model.feature_moment_metric()))
        _,sing_vecs = model.hosvd()
        singular_vec_matrix.append(sing_vecs[0,:].tolist())
        windows.append((si, ei))
    
    feature_matrix = np.array(feature_matrix)
    singular_vec_matrix = np.array(np.array(feature_matrix)) 
    A = get_angle_change(singular_vec_matrix)
    D = get_hellinger_distance(feature_matrix)
    
    fig, ax = plt.subplots(2,1,figsize=(25,20), constrained_layout=True, dpi=200, sharex=True)

    ax[0].plot(np.arange(start_idx, X.shape[0]), np.array(D))
    ax[0].set_xlabel('Window Length', fontsize=24)
    ax[0].set_ylabel('Hellinger Distance', fontsize=24)
    ax[0].tick_params(axis="x", labelsize=20)
    ax[0].tick_params(axis="y", labelsize=20)
    ax[0].set_title(f'Hellinger Distance', fontsize=24)
    ax[0].grid(True)
        
    ax[1].plot(np.arange(start_idx, X.shape[0]), np.array(A))
    ax[1].set_xlabel('Window Length', fontsize=24)
    ax[1].set_ylabel('Angle', fontsize=24)
    ax[1].tick_params(axis="x", labelsize=20)
    ax[1].tick_params(axis="y", labelsize=20)
    ax[1].set_title(f'Angle change', fontsize=24)
    ax[1].grid(True)
    
    if saveToFile:
        plt.savefig(os.path.join(img_path, f'{htype}_{user}.png'))
        plt.close()
    else:
        plt.show()
        
    tmp = pd.DataFrame({'Idx':np.arange(start_idx, X.shape[0]), 'HD':list(D), 'Angle':list(A)})
    
    if saveToFile:
        tmp.to_csv(os.path.join(img_path, f'{htype}_{user}_HD_Angle.csv'), index=False)
    else:
        return tmp
    
def plotVarAdjWindows(data, HD_data, index=None, user='',htype='', img_path='./', saveToFile=False):
    data = data.reset_index(drop=True)
    
    def printData(i):
        now_df = data.iloc[:i, :]
        #print(now_df.shape)

        prev_df = data.iloc[:i-1, :]
        #print(prev_df.shape)

        fig, ax = plt.subplots(2,3,figsize=(25,10), constrained_layout=True, dpi=200, sharex=True)

        ax[0,0].plot(now_df['beginHR'], color='red', label='Cur Begin')
        ax[0,0].plot(prev_df['beginHR'], color='blue', label='Prev Begin')
        ax[0,0].set_title(f'Window Size: {i-1}-{i}')
        ax[0,0].legend()

        ax[0,1].plot(now_df['minHR'], color='red', label='Cur Min')
        ax[0,1].plot(prev_df['minHR'], color='blue', label='Prev Min')
        ax[0,1].legend()

        ax[0,2].plot(now_df['maxHR'], color='red', label='Cur Max')
        ax[0,2].plot(prev_df['maxHR'], color='blue', label='Prev Max')
        ax[0,2].legend()

        ax[1,0].plot(now_df['avgHR'], color='red', label='Cur Avg')
        ax[1,0].plot(prev_df['avgHR'], color='blue', label='Prev Avg')
        ax[1,0].legend()

        ax[1,1].plot(now_df['stdHR'], color='red', label='Cur Std')
        ax[1,1].plot(prev_df['stdHR'], color='blue', label='Prev Std')
        ax[1,1].legend()

        ax[1,2].plot(now_df['endHR'], color='red', label='Cur End')
        ax[1,2].plot(prev_df['endHR'], color='blue', label='Prev End')
        ax[1,2].legend()

        if saveToFile:
            plt.savefig(os.path.join(img_path, f'{htype}_{user}_{i-1}-{i}.png'))
            plt.close()
        else:
            plt.show()
    
    if index is None or index<0 or index>=data.shape[0]:
        for i in HD_data.Index:
            printData(i)
            
    else:
        printData(index)

def getMDS(X, transformed_dim=2):
    embedding = MDS(n_components=transformed_dim, normalized_stress='auto')
    X_transformed = embedding.fit_transform(X)
    return X_transformed

def getInitialCKValue(X, index, window_size, threshold):
    i=0
    cnt=0
    idv = []
    while cnt<window_size and i<X.shape[0]:
        if X[i, index]<=threshold:
            idv.append(i)
            cnt+=1
            
        i+=1
    
    T = TimeSeries(X[idv,:])
    model = AnomalyDetection(T.cokurtosis())
    fmm = list(model.feature_moment_metric())
    _,sing_vecs = model.hosvd()
    svm = sing_vecs[0,:].tolist()
    
    return fmm, svm, cnt, mode(X[idv,index])[0]

def findMaxIntervalWithHealthyHR(data, minWindow=5, maxWindow=5, threshold=90):
    maxSoFar = [(0,0)]
    
    def sub(data, maxVal, minWindow, maxWindow, threshold, searchStartIdx):
        if searchStartIdx < len(data.index):
            i,index = searchStartIdx,data.index

            # Find the min index
            while i<len(index):
                if data.avgHR[index[i]]<=threshold:
                    break
                i+=1

            j = i+1 
            wl = 1
            while j<len(index):
                if data.avgHR[index[j]]<=threshold:
                    wl+=1
                    j+=1
                    #print(wl)
                    maxVal[0] = (i,j)
                    if wl>=maxWindow:
                        break
                else:
                    break

            #print(i,j, j-i+1, maxVal)
            if j-i+1<minWindow:
                # Save the last largest window
                if j-i > maxVal[0][1]-maxVal[0][0]:
                    maxVal[0] = (i,j)
                sub(data, maxVal, minWindow, maxWindow, threshold, j)

            #return (i,j)
            #print(i,j, j-i+1, maxVal)
    
    sub(data, maxSoFar, minWindow, maxWindow, threshold)
    return maxSoFar[0]
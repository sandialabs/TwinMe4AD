"""
Developer: Methun K.
Date: Aug, 2023
"""

import pandas as pd
import numpy as np
import zipfile
import tarfile
import os
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from dataqueue import DataProcessingQueue
from common import *
import multiprocessing as mp
        
class ZipFileReader:
    def __init__(self, zip_folder='./'):
        self.zipFolder = zip_folder
        self.folder = zip_folder.split('/')[-1].split('.')[0]
        
    def read_from_zip_folder(self, fileName='', datetime_col='datetime', parse_date_cols=[], index_cols=[]):
        data = None
        with tarfile.open(self.zipFolder, 'r:bz2') as tar:
            with tar.extractfile(f'{self.folder}/{fileName}') as file:
                if len(parse_date_cols)==0 and len(index_cols)==0:
                    data = pd.read_csv(file)
                elif len(index_cols)==0:
                    data = pd.read_csv(file, parse_dates=[datetime_col])
                elif len(parse_date_cols)==0:
                    data = pd.read_csv(file, index_col=[datetime_col])
                else:
                    data = pd.read_csv(file, parse_dates=[datetime_col], index_col=[datetime_col])
                
        return data
    
    def getData(self, fileName=''):
        data = None
        
        with tarfile.open(self.zipFolder, 'r:bz2') as tar:
            if fileName in tar.getnames():
                with tar.extractfile(f'{self.folder}/{fileName}') as file:
                    data = pd.read_csv(file)
                    
        return data
    
    
class FileReader(ZipFileReader):
    # Class constructor
    def __init__(self, filePath='./', sick_filePath='./', freqInMinute=60, slidingInMinute=30, healthy_RHR_threshold=100):
        while filePath[-1]=='/': filePath=filePath[:-1]
        super().__init__(filePath) 
        self.file_path = filePath
        self.freqInMinute = int(freqInMinute)
        self.slidingInMinute = int(slidingInMinute)
        self.healthy_RHR = healthy_RHR_threshold
        self.sick_filePath = sick_filePath
        
        plist = filePath.split('/')
        
        try:
            self.saved_file_path = create_path(os.path.join('/'.join(plist[:-1]), 'refined'))
        except Exception as e:
            print("Error to create file: ",e)
        finally:
            pass
        
        try:
            self.profile_path = create_path(os.path.join('/'.join(plist[:-1]), 'profile'))
        except Exception as e:
            print("Error to create file: ",e)
        finally:
            pass
        
        self.user_profile_file = os.path.join(self.profile_path, f'user_profile_{self.freqInMinute}_{self.slidingInMinute}_{self.healthy_RHR}.csv')
        
        #print(self.user_profile_file, flush=True)
    
    # If data already exists then read it and return otherwise return None
    def getData(self, file):
        data = None
        f = os.path.join(self.saved_file_path, file)
        if os.path.exists(f):
            #print(f, flush=True)
            data = pd.read_csv(f)
            
        return data
    
    # Compute the [min, max] values of each column
    def get_limit(self, df):
        return df.apply(lambda x: pd.Series([x.min(), x.max()])).T.values.tolist()
    
    def get_profile_data(self):
        return pd.read_csv(self.user_profile_file)
        
    
    # Get all the user name in the file
    def get_all_user(self):
        ulist = []
        
        # Read each file name in the directory
        for f in os.listdir(self.file_path):
            # If this is not a hidden file and the extension of the file is .csv
            if f[0]!='.' and f[-4:]=='.csv':
                tu = []
                # File name could be: ABC_1_hr.csv
                # f[:-4] = ABC_1_hr
                # split('_'): [ABC, 1, hr]
                # Concate name until any of [hr, steps, sleep] found
                # tu = [ABC, 1]
                # After '_'.join(tu) = ABC_1
                for x in f[:-4].split('_'):
                    if x in ['hr','sleep','steps']:
                        break
                    tu.append(x)
                s = '_'.join(tu)
                ulist.append(s)
                
        return sorted(list(set(ulist)))
    
    def get_real_sick_user(self):
        def getValidPatId(x):
            if '_' not in x:
                return x

            t = x.split('_')
            if len(t)>2:
                return t[0]+'_'+t[-1]

            return t[0]
        
        def parse_datetime(x):
            t = []
            for y in x.split("'"):
                if 'Timestamp' in y or ')]' in y:
                    continue
                y = pd.to_datetime(y, format="%Y-%m-%d %H:%M:%S")
                t.append(y)
            return t
        
        sick_pat = []
        
        if not os.path.exists(os.path.join(self.profile_path, 'real_sick_user.csv')):
        
            # Illness reported based on symptom like COVID or other illness 
            # Read from sheet 3, start from row 4
            ftmp = ''
            if os.path.exists(os.path.join(self.sick_filePath, '41551_2020_640_MOESM3_ESM.xlsx')):
                ftmp = os.path.join(self.sick_filePath, '41551_2020_640_MOESM3_ESM.xlsx')
            
            if len(ftmp)>0:
                df = pd.read_excel(ftmp, 
                                   sheet_name=2, 
                                   skiprows=3, 
                                   usecols=['ParticipantID','Symptom_dates'])

                tf = []
                for i in df.index:
                    pid = df.ParticipantID[i]
                    x = df.Symptom_dates[i]

                    t = df.Symptom_dates[i]
                    if 'Timestamp' in t:
                        for x in parse_datetime(t):
                            tf.append([getValidPatId(pid), x])

                # Self reported illness
                # Read from sheet 4, start from row 5
                df = pd.read_excel(ftmp, 
                                   sheet_name=3, 
                                   skiprows=4, 
                                   usecols=['ParticipantID', 'shift_sick_start', 'shift_sick_end', 'sick_feel'])

                df.shift_sick_start = pd.to_datetime(df.shift_sick_start)
                df.shift_sick_end = pd.to_datetime(df.shift_sick_end)

                tf2 = []
                for i in df.index:
                    pid = getValidPatId(df.ParticipantID[i])
                    feel = df.sick_feel[i]
                    s = df.shift_sick_start[i]
                    e = df.shift_sick_end[i]

                    if pd.isna(e):
                        continue

                    if 'required medical attention' in feel:
                        while s<=e:
                            tf2.append([pid, s])
                            s = s + timedelta(days=1)

                # Daily health status report
                # Read from sheet 5, start from row 5
                df = pd.read_excel(ftmp, 
                                   sheet_name=4, 
                                   skiprows=4, 
                                   usecols=['ParticipantID','Date','overall_feel'])
                df = df[df.overall_feel.isin(['Experiencing symptoms of beginning illness', 
                                              'Currently ill'])]


                df.Date = pd.to_datetime(df.Date)
                df = df[df.overall_feel.isin(['Experiencing symptoms of beginning illness', 
                                                      'Currently ill'])]

                tf3 = []
                for i in df.index:
                    pid = getValidPatId(df.ParticipantID[i])
                    t = df.Date[i]

                    tf3.append([pid, t])

                dtf = pd.DataFrame(tf+tf2+tf3, columns=['User', 'Sick_dates'])
                dtf = dtf.drop_duplicates()
                lock = mp.Lock()
                lock.acquire()
                dtf.to_csv(os.path.join(self.profile_path, 'real_sick_user.csv'), index=False)
                lock.release()
                del [tf,tf2,tf3,dtf]
            
        
        dtf = pd.read_csv(os.path.join(self.profile_path, 'real_sick_user.csv'))
        
        return dtf.User.unique()
    
    def get_sick_users_based_on_RHR(self):
        users = self.get_all_user()
        
        su = []
        for u in users:
            data = self.read_file(u)
    
            if data is None or data.shape[0]==0:
                continue
            
            #print(u, data.avgRHR.max())
            
            if data.avgRHR.max() >= self.healthy_RHR:
                su.append(u)
                
        return su
         
    def get_raw_data(self, user, zipped):
        if len(user)==0:
            return None
        
        if zipped:
            df_hr = super().read_from_zip_folder(user+'_hr.csv', parse_date_cols=['datetime'], index_cols=['datetime'])
        else:
            df_hr = pd.read_csv(os.path.join(self.file_path, user+'_hr.csv'), parse_dates=['datetime'], index_col=['datetime'])
            
        if df_hr is None or df_hr.shape[0]==0:
            return None
        
        df_hr = df_hr[['heartrate']]
        
        if zipped:
            df_steps = super().read_from_zip_folder(user+'_steps.csv', parse_date_cols=['datetime'], index_cols=['datetime'])
        else:
            df_steps = pd.read_csv(os.path.join(self.file_path, user+'_steps.csv'), parse_dates=['datetime'], index_col=['datetime'])
            
        if df_steps is None or df_steps.shape[0]==0:
            return None
        
        df_steps = df_steps[['steps']]
        
        # Merge data
        df = df_hr.merge(df_steps, left_index=True, right_index=True)
        del[df_hr, df_steps]
        df = df.dropna()
        df.reset_index(inplace=True)
        
        return df[['datetime', 'steps', 'heartrate']]
            
    
    # Read and merge both heart rate and steps files
    # Read the data and return the data for when step size=0
    def getRHRData(self, user, zipped):
        if len(user)==0:
            return None
        
        df = self.get_raw_data(user, zipped)
        
        if df is None:
            return None
        
        df = df.loc[df.steps==0, ['datetime', 'heartrate']]
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    # This method will use to process data from wearable device
    # Process data within the interval and sliding window using data processing queue
    def read_steraming_data(self, user='', split_dt=False, read_from_zip_file=False, remove_unique_columns=False):
        if len(user)==0:
            return None
        
        final_file_name = f'{user}_{self.freqInMinute}_{self.slidingInMinute}.csv'
        
        df = self.getData(final_file_name)
                
        if df is None:
            print(final_file_name)
            df = self.getRHRData(user, read_from_zip_file)
            q = DataProcessingQueue(self.freqInMinute, self.slidingInMinute)
            for i in df.index:
                q.addRecord(list(df.iloc[i, :].values))

            df = q.getData()
            if df is None:
                return None

            if split_dt:
                df['Date'] = df.datetime.dt.date
                df['Year'] = df.datetime.dt.year
                df['Month'] = df.datetime.dt.month
                df['Day'] = df.datetime.dt.day
                df['DOY'] = df.datetime.dt.dayofyear
                df['Weekday'] = df.datetime.dt.day_name()
                df['Time'] = df.datetime.dt.time
                df['Hour'] = df.datetime.dt.hour
                df['Minute'] = df.datetime.dt.minute
                df['Second'] = df.datetime.dt.second

            # Remove columns with unique value
            if remove_unique_columns:
                l = []
                for c in df.columns:
                    #print(c, df[c].dtype)
                    if df[c].dtype in ['float64','int32']:
                        if df[c].sum()==0: l.append(c)
                #print(l)
                df = df.drop(l, axis=1)
                df = df.reset_index(drop=True)
            
            lock = mp.Lock()
            lock.acquire()
            df.to_csv(os.path.join(self.saved_file_path, final_file_name), index=False)
            lock.release()

        return df
        
    # Process data within the interval and sliding window using data processing queue
    def read_file(self, user, split_dt=False, read_from_zip_file=False, remove_unique_columns=False):
        if len(user)==0:
            return None
        
        final_file_name = f'{user}_{self.freqInMinute}_{self.slidingInMinute}.csv'
        
        df = self.getData(final_file_name)
                
        if df is None:
            print(f'User={user}', flush=True)
            
            df = self.getRHRData(user, read_from_zip_file)
            q = DataProcessingQueue(self.freqInMinute, self.slidingInMinute)
            for i in df.index:
                q.addRecord(list(df.iloc[i, :].values))
                
            df = q.getData()
            if df is None:
                return None

            # Remove columns with unique value
            if remove_unique_columns:
                l = []
                for c in df.columns:
                    #print(c, df[c].dtype)
                    if df[c].dtype in ['float64','int32']:
                        if df[c].sum()==0: l.append(c)
                #print(l)
                df = df.drop(l, axis=1)
                df = df.reset_index(drop=True)
            
            lock = mp.Lock()
            lock.acquire()
            df.to_csv(os.path.join(self.saved_file_path, final_file_name), index=False)
            lock.release()
        
        if split_dt==True:
            df.datetime = pd.to_datetime(df.datetime)
            df['Date'] = df.datetime.dt.date
            df['Year'] = df.datetime.dt.year
            df['Month'] = df.datetime.dt.month
            df['Day'] = df.datetime.dt.day
            df['DOY'] = df.datetime.dt.dayofyear
            df['DOW'] = df.datetime.dt.dayofweek
            df['WOY'] = df.datetime.dt.isocalendar().week
            df['Weekday'] = df.datetime.dt.day_name()
            df['Time'] = df.datetime.dt.time
            df['Hour'] = df.datetime.dt.hour
            df['Minute'] = df.datetime.dt.minute
            df['Second'] = df.datetime.dt.second
            

        return df
    
    # Create limit of each feature values
    def getLimitsByUserType(self, ptype='H'):
        
        default_range = [30, self.healthy_RHR]
        pf = self.user_profile_file
        #print(pf)
        
        if not os.path.exists(pf):
        
            user_list = self.get_all_user()
            print(f"Total users={len(user_list)}", flush=True)
            
            tmp = []

            for u in user_list:
                #print(f'User={u}', flush=True)
                data = self.read_file(u)
                if data is None or data.shape[0]==0:
                    continue

                
                if data.avgRHR.max() < self.healthy_RHR+1:
                    t = [u, 'H']

                else:
                    t = [u, 'S']

                t.extend([data.minRHR.min(), 
                          data.minRHR.max(), 
                          data.avgRHR.min(), 
                          data.avgRHR.max(), 
                          data.medRHR.min(), 
                          data.medRHR.max(), 
                          data.maxRHR.min(), 
                          data.maxRHR.max()]) 
                
                #, data.stdRHR.min(), data.stdRHR.max(), data.skewRHR.min(), 
                # data.skewRHR.max(), data.KurtosisRHR.min(), data.KurtosisRHR.max()])

                tmp.append(t)

                del [data]
            
            data = pd.DataFrame(tmp, columns=['User', 'Type', 'lowMinRHR', 'highMinRHR', 'lowAvgRHR', 
                                              'highAvgRHR', 'lowMedRHR', 'highMedRHR', 'lowMaxRHR', 'highMaxRHR']) 
            
            #, 'rType', 'lowStdRHR', 'highStdRHR', 'lowSkewRHR', 'highSkewRHR', 
            # 'lowKurtosisRHR', 'highKurtosisRHR'])
            
            #lock = mp.Lock()
            #lock.acquire()
            data.to_csv(pf, index=False)
            #lock.release()

            del [tmp, data, user_list]
            
        data = pd.read_csv(pf)
        #print("Before, ",data.shape, flush=True)
        
        if len(ptype)>0 and ptype in ['H', 'S']:
            data = data[data['Type']==ptype]
        
        #print("After, ",data.shape, flush=True)
        
        minRHRRange = default_range
        avgRHRRange = default_range
        medRHRRange = default_range
        maxRHRRange = default_range
        
        if data.shape[0]>0:
            #print(data.head(), flush=True)
            
            if data['lowMinRHR'].min() < data['highMinRHR'].max():
                minRHRRange = [data['lowMinRHR'].min(), data['highMinRHR'].max()]
                
            if data['lowAvgRHR'].min() < data['highAvgRHR'].max():
                avgRHRRange = [data['lowAvgRHR'].min(), data['highAvgRHR'].max()]
                
            if data['lowMedRHR'].min() < data['highMedRHR'].max():
                medRHRRange = [data['lowMedRHR'].min(), data['highMedRHR'].max()]
                
            if data['lowMaxRHR'].min() < data['highMaxRHR'].max():
                maxRHRRange = [data['lowMaxRHR'].min(), data['highMaxRHR'].max()]

        #stdRHRRange = [data['lowStdRHR'].min(), data['highStdRHR'].max()]
        #skewRHRRange = [data['lowSkewRHR'].min(), data['highSkewRHR'].max()]
        #KurtosisRHRRange = [data['lowKurtosisRHR'].min(), data['highKurtosisRHR'].max()]
        
        #print('Done')

        return [minRHRRange, avgRHRRange, medRHRRange, maxRHRRange] #, stdRHRRange, skewRHRRange,KurtosisRHRRange]

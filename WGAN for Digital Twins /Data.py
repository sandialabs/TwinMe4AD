"""
Developer: Methun K.
Date: Aug, 2023
"""

import pandas as pd
import numpy as np
import tarfile
import os
from datetime import timedelta
from collections import deque

class DataProcessingQueue:
    def __init__(self, freqInMinute, slidingInMinute):
        self.freqInMinute = freqInMinute
        self.slidingInMinute = slidingInMinute
        self.queue = deque()
        self.data = []
        
    def __getStatistics__(self):
        tmp = pd.DataFrame(list(self.queue), columns=['datetime', 'heartrate'])
        return [self.queue[-1][0],
                int(np.min(tmp['heartrate'])),
                int(np.mean(tmp['heartrate'])),
                int(np.median(tmp['heartrate'])),
                int(np.max(tmp['heartrate']))
               ]
        
    def __processRecord__(self):
        if len(self.queue)<2:
            return []
        
        #print(self.queue)
        
        t1 = self.queue[0][0]
        t2 = self.queue[-1][0]
        t = (t2-t1).total_seconds()//60
        #print(t1, t2, t)
        
        # Queue length covers the freq in minutes
        result = []
        if t>=self.freqInMinute:
            self.data.append(self.__getStatistics__())
            result = self.data[-1]
            
            # Non-overlapping windows
            if t<=self.slidingInMinute:
                self.queue.clear()
                return result

            # Remove the oldest record
            self.queue.popleft()

            # If queue is empty, nothing to extract and compare
            if len(self.queue)==0:
                return result

            t2 = self.queue[0][0]
            t = (t2-t1).total_seconds()//60

            # Move t2 to right so that t2-t1 <= sliding window
            while t1<t2 and t<=self.slidingInMinute:
                self.queue.popleft()

                # if queue is empty, nothing to extract and compare
                if len(self.queue)==0:
                    break

                t2 = self.queue[0][0]
                t = (t2-t1).total_seconds()//60
                
        return result
        
    def addRecord(self, record):
        self.queue.append(record)
        return self.__processRecord__()
        
    def getData(self):
        if len(self.data)==0:
            return None
        
        return pd.DataFrame(self.data, columns=['datetime','minRHR','avgRHR','medRHR','maxRHR'])
        
        
class ZipFileReader:
    def __init__(self, zip_folder='./'):
        self.zipFolder = zip_folder
        self.folder = zip_folder.split('/')[-1].split('.')[0]
        
    def read_from_zip_folder(self, fileName='', parse_date_cols=[], index_cols=[]):
        data = None
        with tarfile.open(self.zipFolder, 'r:bz2') as tar:
            with tar.extractfile(f'{self.folder}/{fileName}') as file:
                if len(parse_date_cols)==0 and len(index_cols)==0:
                    data = pd.read_csv(file)
                elif len(index_cols)==0:
                    data = pd.read_csv(file, parse_dates=['datetime'])
                elif len(parse_date_cols)==0:
                    data = pd.read_csv(file, index_col=['datetime'])
                else:
                    data = pd.read_csv(file, parse_dates=['datetime'], index_col=['datetime'])
                
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
    def __init__(self, filePath='./', freqInMinute=60, slidingInMinute=30, healthy_RHR_threshold=100):
        while filePath[-1]=='/': filePath=filePath[:-1]
        super().__init__(filePath) 
        self.file_path = filePath
        self.freqInMinute = int(freqInMinute)
        self.slidingInMinute = int(slidingInMinute)
        self.healthy_RHR = healthy_RHR_threshold
        
        plist = filePath.split('/')
        self.saved_file_path = os.path.join('/'.join(plist[:-1]), 'refined')
        self.profile_path = os.path.join('/'.join(plist[:-1]), 'profile')
        
        try:
            self.__create_path__(self.saved_file_path)
            self.__create_path__(self.profile_path)
        except Exception as e:
            print("Error to create file: ",e)
        finally:
            pass
        
        self.user_profile_file = os.path.join(self.profile_path, f'user_profile_{self.freqInMinute}_{self.slidingInMinute}_{self.healthy_RHR}.csv')
        
    # Create folder along the path if does not exists
    def __create_path__(self, fpath):
        if os.path.exists(fpath):
            return
        
        os.makedirs(fpath)
    
    # If data already exists then read it and return otherwise return None
    def getData(self, file=''):
        data = None
        
        if os.path.exists(os.path.join(self.saved_file_path, file)):
            data = pd.read_csv(os.path.join(self.saved_file_path, file))
            
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
    
    def get_sick_user(self):
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
        
        # sick_pat = []
        
        if not os.path.exists(os.path.join(self.profile_path, 'real_sick_user.csv')):
        
            # Illness reported based on symptom like COVID or other illness 
            # Read from sheet 3, start from row 4
            df = pd.read_excel(os.path.join(self.file_path, '41551_2020_640_MOESM3_ESM.xlsx'), 
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
            df = pd.read_excel(os.path.join(self.file_path, '41551_2020_640_MOESM3_ESM.xlsx'), 
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
            df = pd.read_excel(os.path.join(self.file_path, '41551_2020_640_MOESM3_ESM.xlsx'), 
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
            dtf.to_csv(os.path.join(self.profile_path, 'real_sick_user.csv'), index=False)
            del [tf,tf2,tf3,dtf]
            
        
        dtf = pd.read_csv(os.path.join(self.profile_path, 'real_sick_user.csv'))
        
        return dtf.User.unique()
    
    def get_sick_users_based_on_RHR(self):
        users = self.get_all_user()
        
        su = []
        for u in users:
            print("User: ",u)
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
        
        return df.loc[df.steps==0, ['datetime', 'heartrate']]
    
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

            df.to_csv(os.path.join(self.saved_file_path, final_file_name), index=False)

        return df
        
    # Process data within the interval and sliding window using data processing queue
    def read_file(self, user='', split_dt=False, read_from_zip_file=False, remove_unique_columns=False):
        if len(user)==0:
            return None
        
        final_file_name = f'{user}_{self.freqInMinute}_{self.slidingInMinute}.csv'
        
        df = self.getData(final_file_name)
                
        if df is None:
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

            df.to_csv(os.path.join(self.saved_file_path, final_file_name), index=False)

        return df
    
    # Create limit of each feature values
    def getLimitsByUserType(self, uType='Healthy'):
        
        pf = self.user_profile_file
        real_sick_user = self.get_sick_user()
        
        if not os.path.exists(pf):
        
            user_list = self.get_all_user()
            tmp = []

            for u in user_list:
                data = self.read_file(u)
                if data is None or data.shape[0]==0:
                    continue

                if data.avgRHR.max() < self.healthy_RHR+1:
                    t = [u, 'Healthy', 'Sick' if u in real_sick_user else 'Healthy']

                else:
                    t = [u, 'Sick', 'Sick' if u in real_sick_user else 'Healthy']

                t.extend([data.minRHR.min(), 
                          data.minRHR.max(), 
                          data.avgRHR.min(), 
                          data.avgRHR.max(), 
                          data.medRHR.min(), 
                          data.medRHR.max(), 
                          data.maxRHR.min(), 
                          data.maxRHR.max()]) 
                
                #, data.stdRHR.min(), data.stdRHR.max(), data.skewRHR.min(), data.skewRHR.max(), data.KurtosisRHR.min(), data.KurtosisRHR.max()])

                tmp.append(t)

                del [data]

            data = pd.DataFrame(tmp, columns=['User', 'Type', 'rType', 'lowMinRHR', 'highMinRHR', 'lowAvgRHR', 
                                              'highAvgRHR', 'lowMedRHR', 'highMedRHR', 'lowMaxRHR', 'highMaxRHR']) 
            
            #, 'lowStdRHR', 'highStdRHR', 'lowSkewRHR', 'highSkewRHR', 'lowKurtosisRHR', 'highKurtosisRHR'])
            
            data.to_csv(pf, index=False)

            del [tmp, data, user_list]
            
        data = pd.read_csv(pf)
        
        if len(uType)>0 and uType in ['Healthy', 'Sick']:
            data = data[data.Type==uType]

        minRHRRange = [data['lowMinRHR'].min(), data['highMinRHR'].max()]
        avgRHRRange = [data['lowAvgRHR'].min(), data['highAvgRHR'].max()]
        medRHRRange = [data['lowMedRHR'].min(), data['highMedRHR'].max()]
        maxRHRRange = [data['lowMaxRHR'].min(), data['highMaxRHR'].max()]
        
        #stdRHRRange = [data['lowStdRHR'].min(), data['highStdRHR'].max()]
        #skewRHRRange = [data['lowSkewRHR'].min(), data['highSkewRHR'].max()]
        #KurtosisRHRRange = [data['lowKurtosisRHR'].min(), data['highKurtosisRHR'].max()]

        return [minRHRRange, avgRHRRange, medRHRRange, maxRHRRange] #, stdRHRRange, skewRHRRange,KurtosisRHRRange]
    
    
# This class is used to convert data to any range and back to the original range
# Pass the original range as a parameter and save it as member object
# Works only for numeric value data
class DataConversion:
    def __init__(self, limit, low=0.0, high=1.0):
        self.limit = limit
        if low>high:
            low, high = high, low
        self.low = low
        self.high = high
        #print(low, high)
        
    # Convert data
    def __getNewValue__(self, x, newMin, newMax, oldMin, oldMax):
        D_old = oldMax-oldMin
        D_new = newMax-newMin
        return newMin+(((x-oldMin)*D_new)/D_old)
        
    # Convert data from original range to new range
    def convertToNewRange(self, X):
        X_new = X.copy()
        X_new = X_new.astype(np.float32)

        for d in range(np.size(X_new, 1)):
            minf, maxf = self.limit[d][0], self.limit[d][1]
            X_new[:, d] = [self.__getNewValue__(x, self.low, self.high, minf, maxf) for x in X_new[:, d]]

        return X_new 
    
    # Convert data from new range to original range
    def revertToOriginalValue(self, X):
        X_new = X.copy()
        X_new = X_new.astype(np.float32)

        for d in range(np.size(X_new, 1)):
            minf, maxf = self.limit[d][0], self.limit[d][1]
            X_new[:, d] = [self.__getNewValue__(x, minf, maxf, self.low, self.high) for x in X_new[:, d]]

        return X_new
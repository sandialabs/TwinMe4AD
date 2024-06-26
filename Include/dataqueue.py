"""
Developer: Methun K.
Date: Aug, 2023
Purpose: Process stream of time series health data using a Queue
"""

from collections import deque
import pandas as pd
import numpy as np

class DataProcessingQueue:
    def __init__(self, freqInMinute, slidingInMinute, datetime_features='datetime', health_vital_features=['heartrate']):
        self.freqInMinute = freqInMinute
        self.slidingInMinute = slidingInMinute
        self.queue = deque()
        self.data = []
        self.datetime_features = datetime_features
        self.health_vital_features = health_vital_features
        
    def __getStatistics__(self):
        tmp = pd.DataFrame(list(self.queue), columns=[self.datetime_features]+self.health_vital_features)
        fl = [self.queue[-1][0]]

        for f in self.health_vital_features:
            if len(self.health_vital_features)==1:
                fl.append(np.min(tmp[f]))
                fl.append(np.mean(tmp[f]))
                fl.append(np.median(tmp[f]))
                fl.append(np.max(tmp[f]))
            else:
                fl.append(np.mean(tmp[f]))
            
        return fl
        
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
        #print(record)
        self.queue.append(record)
        return self.__processRecord__()
        
    def getData(self):
        if len(self.data)==0:
            return None

        cols = [self.datetime_features]
        for f in self.health_vital_features:
            if len(self.health_vital_features)==1:
                cols.append(f"min_{f}")
                cols.append(f"avg_{f}")
                cols.append(f"med_{f}")
                cols.append(f"max_{f}")
            else:
                cols.append(f"avg_{f}")
                
        
        return pd.DataFrame(self.data, columns=cols)
        
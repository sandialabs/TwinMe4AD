"""
Developer: Methun Kamruzzaman
Date: Nov, 2023
purpose: This class is used to convert data to any range and back to the original range
            Pass the original range as a parameter and save it as member object
            Works only for numeric value data
"""

import numpy as np

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
"""
Developer: Methun K.
Date: Aug, 2023
"""

import numpy as np
import pandas as pd
from filehandler import FileReader
from converter import DataConversion
import tensorflow as tf
import scipy as sp

class UserData:
    def __init__(self, user_ID):
        self.__user_ID__ = user_ID
        self.limit = []
        self.dc = None
        self.datetimeColVal = []
        self.realization = None
        self.epsilon = None
        
    def get_user_id(self):
        return self.__user_ID__
        
    def get_data(self, data_dir, rhr):
        data = None
        fileReader = FileReader(filePath=data_dir, healthy_RHR_threshold=rhr)
        data= fileReader.get_raw_data(user=self.__user_ID__, zipped=False)
        self.limit = fileReader.get_limit(df.iloc[:, 1:])
        del[fileReader]
        return data
        
    def get_transformed_data(self, data_dir, rhr, transformation=(-1,1), as_tensor=False):
        data = None
        fileReader = FileReader(filePath=data_dir, healthy_RHR_threshold=rhr)
        df= fileReader.get_raw_data(user=self.__user_ID__, zipped=False)
        self.limit = fileReader.get_limit(df.iloc[:, 1:])
        self.dc = DataConversion(self.limit, low=transformation[0], high=transformation[1])
        data = self.dc.convertToNewRange(df.iloc[:, 1:].to_numpy())
        self.datetimeColVal = df.iloc[:, 0].to_numpy()
        del[fileReader]
        
        if as_tensor == True:
            return tf.convert_to_tensor(data)
        
        return data
        
    def get_reverse_transformed_data(self, transformed_data, is_numpy = False):
        
        if not isinstance(transformed_data, np.ndarray):
            transformed_data = transformed_data.numpy()
        
        tr_df = self.dc.revertToOriginalValue(transformed_data)
        tr_df = tr_df.astype(int)
        
        if is_numpy==True:
            return tr_df

        return pd.DataFrame({'datetime':self.datetimeColVal, 'steps':tr_df[:, 0], 'heartrate':tr_df[:, 1]})

    def get_perturbed_data(self, transformed_data, epsilon = 0.0):
        rows = transformed_data.shape[0]
        transformed_data_with_perturbation = np.zeros(transformed_data.shape)
        perturbation = np.zeros(transformed_data.shape)

        # Generate perturbation for HR and Steps independently
        lhs1 = sp.stats.qmc.LatinHypercube(d=1, seed=np.random.randint(100000))
        perturbation[:,0] = np.random.permutation(sp.stats.norm.ppf(lhs1.random(rows), 0, 1)).reshape((rows,))
        lhs1 = sp.stats.qmc.LatinHypercube(d=1, seed=np.random.randint(100000))
        perturbation[:,1] = np.random.permutation(sp.stats.norm.ppf(lhs1.random(rows), 0, 1)).reshape((rows,))

        # Scale perturbation to [-1,1]
        fr = FileReader()
        noise_limit = fr.get_limit(pd.DataFrame(perturbation))
        noise_dc = DataConversion(noise_limit, low=-1, high=1)
        transformed_perturbation = noise_dc.convertToNewRange(perturbation)
        del[fr]

        # Add perturbation to transformed data
        transformed_data_with_perturbation= transformed_data + epsilon * transformed_perturbation

        # Revert transformed_data+perturbation to original data range
        print("Original limits=",self.limit, flush=True)
        original_data_with_perturbation = self.dc.revertToOriginalValue(transformed_data_with_perturbation)
        original_data_with_perturbation = original_data_with_perturbation.astype(int)
        original_data_with_perturbation[original_data_with_perturbation<0] = 0
        del transformed_data_with_perturbation
        
        # Transform data+perturbation to new data range
        fr = FileReader()
        new_limit = fr.get_limit(pd.DataFrame(original_data_with_perturbation))
        new_dc = DataConversion(new_limit, low=-1, high=1)
        transformed_data_with_perturbation = new_dc.convertToNewRange(original_data_with_perturbation)
        del[fr]
        
        # Update UserData to reflect changes in data
        self.limit = new_limit
        self.dc = new_dc
        print("New limits=",self.limit, flush=True)
        
        return transformed_data_with_perturbation
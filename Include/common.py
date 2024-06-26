"""
Developer: Methun Kamruzzaman
Date: Nov, 2023
purpose: Common methods used in various files
"""

import os
import numpy as np
import pandas as pd
import multiprocessing as mp


# Create folder along the path if does not exists
def create_path(fpath):
    if os.path.exists(fpath):
        return fpath
    
    lock = mp.Lock()
    lock.acquire()
    try:
        os.makedirs(fpath, exist_ok=True)
    except Exception as e:
        print(f"[{os.getpid()}]Error to create file: {e}")
    
    lock.release()
    return fpath

# Generate latent data
# Fill each column with random value
def get_latent_data(nrows, latent_dim):
    X = np.empty([nrows, latent_dim])

    # For each column, generate random value with in a limit
    for j in range(latent_dim):
        X[:, j] = np.random.randn(nrows)

    # Convert the data to float32
    X = X.astype(np.float32)

    return X

# Get the difference between two time in minute scale
def getDurationInMin(t1, t2):
    t1 = pd.to_datetime(t1)
    t2 = pd.to_datetime(t2)
    
    if t1>t2:
        t1,t2=t2,t1
        
    t = t2-t1
    return round(t.total_seconds()/60, 2)


import tensorflow as tf
from WGAN import WGAN
# import matplotlib
# matplotlib.use('Agg')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)

# Folder containing real user data
data_dir = '../../Data/COVID-19-Wearables/'

# Folder where to save img, data, model and summary subfolders
outdir      = "./results/"

# Folder where to save results for ALL realizations, if ==None, files are saved in folder outdir/data
save_folder = "./results/"

# Users to clone
users = ["A45F9E6"]

# List of Perturbations per user
epsilon = [0.01]

# number of realizations per epsilon, per user
nrealizations = 2

# threshold values for Wasserstein distance
thresholds = {"RHR":0.015,"AHR":0.100,"OHR":0.007}

#%% Generate data
obj = WGAN(data_dir)
save_location = obj.generate_synthetic_data_for_user(users=users, 
                                                  latent_dim=100, 
                                                  nEpochs=1000, 
                                                  nEval=100,
                                                  verbose=1, 
                                                  thresholds=thresholds,
                                                  epsilon=epsilon,
                                                  nrealizations=nrealizations,
                                                  result_dir=outdir,
                                                  save_folder=save_folder,
                                                  save_original=True)

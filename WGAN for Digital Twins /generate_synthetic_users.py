import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from WGAN import WGAN
from Parser import Parser

# Check GPU availablility
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)

# Get parsed arguments
args = Parser()

# Generate Sythetic data
obj = WGAN(args.real_data_dir)
save_location = obj.generate_synthetic_data_for_user(users=args.users, 
                                                  nEpochs=args.nEpochs, 
                                                  nEval=args.nEval,
                                                  verbose=1, 
                                                  thresholds={"RHR":args.RHR,"AHR":args.AHR,"OHR":args.OHR},
                                                  epsilon=args.epsilon,
                                                  nrealizations=args.nrealizations,
                                                  result_dir=args.output_data_dir,
                                                  save_folder=args.synthetic_data_dir,
                                                  save_original=True)

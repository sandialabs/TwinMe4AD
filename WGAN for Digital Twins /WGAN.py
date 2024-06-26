import os
import sys
sys.path.insert(0, '../Include/')

from userdatahandler import UserData

from Model import GenerativeAI
# import matplotlib
# matplotlib.use('Agg')

# Ignore some warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

class WGAN:
    def __init__(self, data_dir, healthy_RHR_threshold=100):
        self.data_dir = data_dir
        self.healthy_RHR_threshold = healthy_RHR_threshold
            
    def generate_synthetic_data_for_user(self, users, 
                                         latent_dim=100, 
                                         nEpochs=50000, 
                                         nEval=1000,
                                         verbose=1, 
                                         thresholds = {"RHR":0.015,"AHR":0.100,"OHR":0.007},
                                         epsilon=[0.0],
                                         nrealizations = 1,
                                         keep_only_result_model=False,
                                         result_dir='./', 
                                         save_folder = None,
                                         save_original = False
                                        ):
        """
        result_dir: str, default ./ (current directory)
            Output directory where the image and model will save. In that directory, a folder named 'result' will create. Without having write permission on this directory will throw an error.
            
        """    
        nusers = len(users)
        for u in range(nusers):
            user_id = users[u]
            for e in epsilon:
                for r in range(0,nrealizations):
                    
                    user = UserData(user_id)
                    user.epsilon = e
                    
                    # Get perturbed data
                    transformed_data = user.get_transformed_data(self.data_dir, self.healthy_RHR_threshold)
                    transformed_data_with_perturbation = user.get_perturbed_data(transformed_data,user.epsilon)

                    # Instatiate GenAI
                    gan_model = GenerativeAI(user=user.__user_ID__, 
                                             latent_dim=latent_dim, 
                                             data_dim=transformed_data.shape[1], 
                                             output_dir=result_dir, 
                                             data_dir=save_folder, 
                                             epsilon=user.epsilon)

                    user.realization = gan_model.realization

                    print("\n\n", flush=True)
                    print("#######################", flush=True)
                    print("Running user",user.__user_ID__,", epsilon= ",str(user.epsilon),", Realization ",str(user.realization),"\n", flush=True)    

                    print(gan_model.summary(), flush=True)

                    # Save original (real) data with perturbations
                    if (save_original):
                        original_data_with_perturbation = user.get_reverse_transformed_data(transformed_data_with_perturbation)
                        original_data_with_perturbation[['datetime', 'steps']].to_csv(os.path.join(gan_model.data_dir,
                                                        f'{gan_model.generate_perturbed_user()}_epsilon_{user.epsilon}_r{user.realization}_steps.csv'))
                        original_data_with_perturbation[['datetime', 'heartrate']].to_csv(os.path.join(gan_model.data_dir, 
                                                        f'{gan_model.generate_perturbed_user()}_epsilon_{user.epsilon}_r{user.realization}_hr.csv'))
                    
                    # Train model
                    min_dist_rhr, min_dist_ahr, min_dist_ohr, min_epoch = gan_model.train(user, 
                                                                                          transformed_data,
                                                                                          transformed_data_with_perturbation, 
                                                                                          nEpochs=nEpochs, 
                                                                                          nEval=nEval, 
                                                                                          verbose=verbose,
                                                                                          thresholds=thresholds
                                                                                         )
        
        return save_folder

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import scipy as sp
from time import time
from datetime import timedelta
import matplotlib.pyplot as plt

# from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import sys

# Added to make code work on Singra
os.environ["TF_USE_LEGACY_KERAS"] = "1"    

# Compute probability distribution functions
class ProbabilityDistributionDistance:
    def Hellinger_distance(self, p, q):
        return sum([(np.sqrt(x)-np.sqrt(y))*(np.sqrt(x)-np.sqrt(y)) for x,y in zip(p,q)])/np.sqrt(2.)
    
    def KL_distance(self, p, q, epsilon = 0.00001):
        # You may want to instead make copies to avoid changing the np arrays.
        p = [i+epsilon for i in p]
        q = [i+epsilon for i in q]

        divergence = np.sum([x*np.log(x/y) for x,y in zip(p,q)])
        return divergence
    
    def Wasserstein_distance(self, p, q):
        return sp.stats.wasserstein_distance(p, q)

# According to the WGAN paper (https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf)
# the weight will be clipped after adjustment
class ClipConstraint(tf.keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clipValue, **kwargs):
        super(ClipConstraint, self).__init__()
        self.clip_value = np.absolute(clipValue)
        
    # clip model weights to hypercube
    def call(self, weight):
        # return backend.clip(weight, -self.clip_value, self.clip_value)
        return tf.keras.backend.clip(weight, -self.clip_value, self.clip_value)

    # get the clip value
    def get_Config(self):
        return {'clip_value':self.clip_value}

# The generator-discriminator model
class GenerativeAI:
    def __init__(self, user, latent_dim=100, data_dim=2, hidden_unit=256, weight_clip=0.1, init_stdv=0.2, learning_rate=0.0005, output_dir='./', data_dir=None, epsilon=0.0):
        self.user = user
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        #self.init = tf.keras.initializers.RandomNormal(stddev=init_stdv, seed=np.random.randint(100, 100000, size=1)[0])
        self.init = tf.keras.initializers.RandomNormal(stddev=init_stdv, seed=int(np.random.randint(100, 100000, size=1)[0]))
        self.const = ClipConstraint(weight_clip)
        self.generator = self.__generator__(hidden_unit)
        self.discriminator = self.__discriminator__(hidden_unit)
        self.generator_opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.discriminator_opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.prd = ProbabilityDistributionDistance()
        self.dloss=[]
        self.gloss=[]
        self.resting_distance = []
        self.non_resting_distance = []
        self.pr_distance = []
        self.time = []

        self.output_dir = os.path.join(output_dir, user, f"epsilon_{epsilon}")
        self.realization= self.__create_main_path__(self.output_dir)
        self.output_dir = os.path.join(output_dir, user, f"epsilon_{epsilon}",f"R{self.realization}")

        self.img_dir = os.path.join(self.output_dir, 'img')
        self.__create_path__(self.img_dir)
        self.model_dir = os.path.join(self.output_dir, 'model')
        self.__create_path__(self.model_dir)
        self.summary_dir = os.path.join(self.output_dir, 'summary')
        self.__create_path__(self.summary_dir)
        
        if data_dir is None:
            self.data_dir = os.path.join(self.output_dir, 'data')
        else:
            self.data_dir = os.path.join(output_dir, user, f"epsilon_{epsilon}", "Data")
            
        self.__create_path__(self.data_dir)
        
    # The generator model
    # The model outputs in the range (-1,1)
    def __generator__(self, hidden_unit=256):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(hidden_unit, kernel_initializer=self.init, kernel_constraint=self.const, input_dim=self.latent_dim, use_bias=False))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(self.data_dim, activation='tanh', use_bias=False))
        return model
    
    # The discriminator model
    # The output is linear
    def __discriminator__(self, hidden_unit=256):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(hidden_unit, kernel_initializer=self.init, kernel_constraint=self.const, input_dim=self.data_dim))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(1))
        return model
        
    # Loss = E_(x~Pr)[f(x)]-E_(x~Pt)[f(x)]
    # Maximize Loss <=> Minimize -Loss
    def __discriminator_loss__(self, real_output, fake_output):
        return -1*(tf.math.reduce_mean(real_output)-tf.math.reduce_mean(fake_output))
    
    # Minimize Loss <=> Maximize E_(x~Pt)[f(x)] <=> Minimize -E_(x~Pt)[f(x)]
    def __generator_loss__(self, output):
        return -1*tf.math.reduce_mean(output)
    
    # Generate synthetic user name
    def generate_user(self):
        return self.user+'_Syn'
    
    # Generate perturbed user name
    def generate_perturbed_user(self):
        return self.user+'_perturbed'
    
    # Create main folder along the path if does not exists
    def __create_main_path__(self, fpath):
        fflag = False
        r = 0
        while (not fflag):
            ffpath = os.path.join(fpath, f"R{r}")
            if os.path.exists(ffpath):
                print("*** Found previous realization at",ffpath," ***")
                r+=1
            else:
                print("*** Creating new realization at ",ffpath," ***")
                os.makedirs(ffpath)
                fflag = True
        return r
         
    # Create folder along the path if does not exists
    def __create_path__(self, fpath):
        if os.path.exists(fpath):
            return
        
        os.makedirs(fpath)

    # Generate latent data
    # Fill each column with random value
    def get_latent_data(self, nrows):
        X = np.empty([nrows, self.latent_dim])

        # For each column, generate random value with in a limit
        for j in range(self.latent_dim):
            X[:, j] = np.random.randn(nrows)

        # Convert the data to float32
        X = X.astype(np.float32)
        
        return X

    def __save_genertor_model__(self, epoch):
        self.generator.save(os.path.join(self.model_dir, f'{self.generate_user()}_{epoch}'))
        
    def __save_synthetic_data__(self, epoch, real_data_with_perturbation,userdata):
        latent_data = self.get_latent_data(real_data_with_perturbation.shape[0])
        opt_model = tf.keras.models.load_model(os.path.join(self.model_dir, f'{self.generate_user()}_{epoch}'))
        pv = opt_model(latent_data)
        synthetic_data = userdata.get_reverse_transformed_data(pv)
        synthetic_data[['datetime', 'steps']].to_csv(os.path.join(self.data_dir, f'r{userdata.realization}_epoch_{epoch}_steps.csv'))
        synthetic_data[['datetime', 'heartrate']].to_csv(os.path.join(self.data_dir, f'r{userdata.realization}_epoch_{epoch}_hr.csv'))
        
    def __get_PR_distance__(self, real_data, noise, method=''):
        generated_data = self.generator(noise, training=False)
        generated_data = generated_data.numpy()
        
        real_rhr = list(real_data[np.where(real_data[:,0]==-1), 1][0])
        fake_rhr = list(generated_data[np.where(generated_data[:,0]==-1), 1][0])
        
        real_ahr = list(real_data[np.where(real_data[:,0]>-1), 1][0])
        fake_ahr = list(generated_data[np.where(generated_data[:,0]>-1), 1][0])
        
        dist_rhr, dist_ahr, dist_hr = float('+inf'),float('+inf'),float('+inf')
        
        if len(real_rhr)>0 and len(fake_rhr)>0: 
            if method=='KL':
                dist_rhr = self.prd.KL_distance(real_rhr, fake_rhr)
            elif method=='EMD':
                dist_rhr = self.prd.Wasserstein_distance(real_rhr, fake_rhr)
            else:
                dist_rhr = self.prd.Hellinger_distance(real_rhr, fake_rhr)
                
        if len(real_ahr)>0 and len(fake_ahr)>0:
            if method=='KL':
                dist_ahr = self.prd.KL_distance(real_ahr, fake_ahr)
            elif method=='EMD':
                dist_ahr = self.prd.Wasserstein_distance(real_ahr, fake_ahr)
            else:
                dist_ahr = self.prd.Hellinger_distance(real_ahr, fake_ahr)
            
        if len(real_data)>0 and len(generated_data)>0:
            if method=='KL':
                dist_hr = self.prd.KL_distance(real_data[:, 1], generated_data[:, 1])
            elif method=='EMD':
                dist_hr = self.prd.Wasserstein_distance(real_data[:, 1], generated_data[:, 1])
            else:
                dist_hr = self.prd.Hellinger_distance(real_data[:, 1], generated_data[:, 1])

        return dist_rhr, dist_ahr, dist_hr
    
    def __summary__(self, 
                    epoch,
                    noise, 
                    userdata, 
                    real_data, 
                    real_data_with_perturbation, 
                    min_epoch, 
                    min_dist_rhr, 
                    min_dist_ahr, 
                    min_dist_ohr, 
                    window_ma, 
                    category, 
                    verbose):

        # Save model (for all checkpoints)
        generated_data = self.generator(noise, training=False)
        self.__save_genertor_model__(epoch)
        
        # Save synthetic data (only for Category 1)
        if (category == 1):
            self.__save_synthetic_data__(epoch, real_data_with_perturbation,userdata)
            
        # Write summary
        loss_file = os.path.join(self.summary_dir, f'{self.generate_user()}_Summary.csv')
        df_summary = self.getSummaryResults()
        df_summary.to_csv(loss_file, index=False)  
                     
        # Plot summary
        lw=1
        lw_ma=2
        ps = 30
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.rcParams['figure.dpi'] = 75
        plt.rcParams['font.style'] = 'normal'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 0.7*plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

        fig = plt.figure()

        gs = fig.add_gridspec(2,2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        # Plot Distance
        if (category >= 0):
            ax1.set_title(f"At Epoch={epoch}, RHR distance={self.resting_distance[-1]:.3f}, AHR distance={self.non_resting_distance[-1]:.3f}, OHR distance={self.pr_distance[-1]:.3f}, Category {category}")
        else:
            ax1.set_title(f"At Epoch={epoch}, RHR distance={self.resting_distance[-1]:.3f}, AHR distance={self.non_resting_distance[-1]:.3f}, OHR distance={self.pr_distance[-1]:.3f}")
            
        ax1.plot(self.resting_distance,label="RHR distance", color=((1,0,0,0.5)), lw=lw)
        ax1.plot(self.non_resting_distance,label="AHR distance", color=((0,0,1,0.5)), lw=lw)
        ax1.plot(self.pr_distance,label="OHR distance", color=((0.5,0.5,0.5,0.5)), lw=lw)
        
        # Here for now but might take it out after
        resting_distance = pd.DataFrame(self.resting_distance)
        resting_distance.replace([np.inf, -np.inf], 1000, inplace=True)
        RHR_distance_MA = resting_distance.rolling(window_ma).mean()
    
        non_resting_distance = pd.DataFrame(self.non_resting_distance)
        non_resting_distance.replace([np.inf, -np.inf], 1000, inplace=True)
        AHR_distance_MA = non_resting_distance.rolling(window_ma).mean()

        pr_distance = pd.DataFrame(self.pr_distance)
        pr_distance.replace([np.inf, -np.inf], 1000, inplace=True)
        OHR_distance_MA = pr_distance.rolling(window_ma).mean()

        ax1.plot(RHR_distance_MA.values,label="RHR distance MA, window="+str(window_ma), color=((1,0,0,1)), lw=lw_ma)
        ax1.plot(AHR_distance_MA.values,label="AHR distance MA, window="+str(window_ma), color=((0,0,1,1)), lw=lw_ma)
        ax1.plot(OHR_distance_MA.values,label="OHR distance MA, window="+str(window_ma), color=((0,0,0,1)), lw=lw_ma)
        
        # Plot
        label = f"Min Epoch={min_epoch}, min RHR distance={min_dist_rhr:.3f}, min AHR distance={min_dist_ahr:.3f}, min OHR distance={min_dist_ohr:.3f}"
        ax1.plot([min_epoch, min_epoch], [0,1],label=label, lw=lw,  color=((0,0,0,0.8)), linestyle='dashed')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Distance')
        ax1.legend(loc="upper left")
        ax1.set_ylim(0,1)
        ax1.set_xlim(left=window_ma)
        
        # Plot RHR data
        ax2.set_title("RHR Histogram")
        
        rhr = real_data[:, 1][real_data[:, 0] == -1]
        ax2.hist(rhr, bins=50, color=((0.5,0.5,0.5,0.1)), edgecolor=((0.5,0.5,0.5,0.2)), label='Real data')
        rhr = real_data_with_perturbation[:, 1][real_data_with_perturbation[:, 0] == -1]
        rhr_min = rhr.min()
        rhr_max = rhr.max()
        ax2.hist(rhr, bins=50, color=((1,0,0,0.3)), edgecolor=((1,0,0,0.5)), label='Real data with pert')
        rhr = generated_data[:, 1][generated_data[:, 0] == -1]
        ax2.hist(rhr, bins=50, range=(rhr_min,rhr_max), color=((0,0,1,0.3)), edgecolor=((0,0,1,0.5)), label='Generated data')
        
        ax2.legend()
        ax2.set_xlabel('RHR')
        ax2.set_yscale('log')

        # Plot all data
        ax3.scatter(real_data[:, 0], real_data[:, 1], label='Real data', s=ps, color=((0.5,0.5,0.5,0.2)))
        ax3.scatter(real_data_with_perturbation[:, 0], real_data_with_perturbation[:, 1], label='Real data with pert', s=ps, color=((1,0,0,0.2)))
        ax3.scatter(generated_data[:, 0], generated_data[:, 1], label='Generated data', s=ps, color=((0,0,1,0.2)))
        ax3.legend()
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Heart Rate')
        ax3.set_title("HR distribution")

        plt.tight_layout()
        
        if verbose>0:
            plt.savefig(os.path.join(self.img_dir, f'{self.generate_user()}_{epoch}.png'))
            plt.show()
        else:
            plt.savefig(os.path.join(self.img_dir, f'{self.generate_user()}_{epoch}.png'))

        plt.close()

    def summary(self):
        print('Generator model summary', flush=True)
        print(self.generator.summary(), flush=True)
        #print()
        print('Discriminator model summary', flush=True)
        print(self.discriminator.summary(), flush=True)
        
    def getSummaryResults(self):
        return pd.DataFrame({'epocs':[i for i in range(1, len(self.dloss)+1)], 
                             'disc_loss':self.dloss, 
                             'gen_loss':self.gloss, 
                             'RHR_distance':self.resting_distance,
                             'AHR_distance':self.non_resting_distance,
                             'PR_distance':self.pr_distance,
                             'Time':self.time
                            })
        
    def train(self, 
              userdata,
              real_data,  
              real_data_with_perturbation,
              nEpochs=50000, 
              nCritic=5, 
              nEval=1000, 
              verbose=1, 
              thresholds={"RHR":0.015,"AHR":0.100,"OHR":0.007}
              ):
        """
        userdata: Current instance of UserData class
        real_data: Pandas dataframe or numpy array. 
                   The dataset containing real valued information.
        real_data_with_perturbation: Pandas dataframe or numpy array. 
                                     The dataset containing real valued information 
                                     plus LHS perturbations
        nEpochs: int, default 50000. 
                 The number of training iterations
        nCritic: int, default 5. 
                 The number of times the descriminator training per epoch
        nEval: int, default 1000. 
               The interval to show results. Also used for moving average of 
               RHR_distance, etc. (for plot only, see summary)
        verbose: Boolean, default 1. 
                 Will plot summary on console when set it to 1, otherwise, 
                 it will just save figure.
        thresholds["RHR"]: float, default 0.015. 
                       The distance between the probability distributions 
                       (only for RHR) generated from real data (plus perturbation) 
                       and synthetic data
        thresholds["AHR"]: float, default 0.100. 
                       The distance between the probability distributions 
                       (only for AHR) generated from real data (plus perturbation) 
                       and synthetic data.
        thresholds["OHR"]: float, default 0.007. 
                       The distance between the probability distributions 
                       (for OHR = RHR + AHR) generated from real data (plus 
                       perturbation) and synthetic data.
        """
        start = time()
        if isinstance(real_data, pd.DataFrame):
            real_data = real_data.to_numpy()
            
        if isinstance(real_data_with_perturbation, pd.DataFrame):
            real_data_with_perturbation = real_data_with_perturbation.to_numpy()
                
        self.dloss=[]
        self.gloss=[]
        self.resting_distance = []
        self.non_resting_distance = []
        self.pr_distance = []
        self.time = []
        
        min_dist_rhr = float('+inf')
        min_dist_ahr = float('+inf')
        min_dist_ohr = float('+inf')
        min_epoch = -1
        min_epoch_old = -1
        category_one_count = 0
       
        for epoch in range(1, nEpochs+1):
            curTime = time()
            # if verbose>0:
            #     clear_output(wait=True)
                
            noise = self.get_latent_data(np.size(real_data_with_perturbation, 0))
            
            for critic in range(nCritic):
                with tf.GradientTape() as disc_tape:
                    fake_data = self.generator(noise, training=True)

                    real_output = self.discriminator(real_data_with_perturbation, training=True)
                    fake_output = self.discriminator(fake_data, training=True)

                    disc_loss = self.__discriminator_loss__(real_output, fake_output)
                
                disc_gradient = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
                self.discriminator_opt.apply_gradients(zip(disc_gradient, self.discriminator.trainable_variables))
                
            with tf.GradientTape() as gen_tape:
                fake_data = self.generator(noise, training=True)
                fake_output = self.discriminator(fake_data, training=True)
                gen_loss = self.__generator_loss__(fake_output)
                
            gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_opt.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
            
            # Save the results
            self.time.append(time()-curTime)
            self.dloss.append(-1*disc_loss.numpy())
            self.gloss.append(-1*gen_loss.numpy())
            d_rhr, d_ahr, d_ohr = self.__get_PR_distance__(real_data_with_perturbation, noise, method='EMD')
            self.resting_distance.append(d_rhr)
            self.non_resting_distance.append(d_ahr)
            self.pr_distance.append(d_ohr)
            
            # Compute min, max and ratio of RHR distance over retrospective window=nEval
            # This is used to break the epoch loop (see Category 3)
            window_ma = nEval
            resting_distance = np.array(self.resting_distance)
            RHR_distance_max = resting_distance[-window_ma:].max()
            RHR_distance_min = resting_distance[-window_ma:].min()
            RHR_distance_ratio = np.abs(RHR_distance_max/RHR_distance_min)

            # Print Results at nEval and finds min dist until this epoch
            if (epoch%nEval==0):
                print('\n', flush=True)
                print(f'---> Checkpoint at Epoch={epoch}: \n', flush=True)   
                print('Last overall minimum dist=',min_dist_ohr, flush=True)
                print('Current overall minimum dist=',d_ohr, flush=True)
                print('Last AHR minimum dist=',min_dist_ahr, flush=True)
                print('Current AHR minimum dist=',d_ahr, flush=True)
                print('Last RHR minimum dist=',min_dist_rhr, flush=True)
                print('Current RHR minimum dist=',d_rhr, flush=True)
                print('\n', flush=True)

                category = -1
                self.__summary__(epoch, 
                                 noise, 
                                 userdata,
                                 real_data, 
                                 real_data_with_perturbation, 
                                 min_epoch,
                                 min_dist_rhr, 
                                 min_dist_ahr, 
                                 min_dist_ohr, 
                                 window_ma, 
                                 category, 
                                 verbose)  

            # Print Results at nEval and finds min dist until this epoch
            if (epoch>nEval) and \
               (min_dist_ohr>d_ohr) and (min_dist_ahr>d_ahr) and (min_dist_rhr>d_rhr) and \
               (d_ohr > thresholds["OHR"]) and (d_ahr > thresholds["AHR"]) and (d_rhr > thresholds["RHR"]):
                print('\n', flush=True)
                print(f'---> Found minimum at Epoch={epoch} (Category 0): \n', flush=True)
                print('Last overall minimum dist=',min_dist_ohr, flush=True)
                print('Current overall minimum dist=',d_ohr, flush=True)
                print('Last AHR minimum dist=',min_dist_ahr, flush=True)
                print('Current AHR minimum dist=',d_ahr, flush=True)
                print('Last RHR minimum dist=',min_dist_rhr, flush=True)
                print('Current RHR minimum dist=',d_rhr, flush=True)
                print('\n', flush=True)

                if (min_dist_ohr>d_ohr) and (min_dist_ahr>d_ahr) and (min_dist_rhr>d_rhr):
                    min_dist_ohr = d_ohr
                    min_dist_ahr = d_ahr
                    min_dist_rhr = d_rhr
                    min_epoch = epoch
                
                category = 0
                self.__summary__(epoch, 
                                 noise, 
                                 userdata,
                                 real_data, 
                                 real_data_with_perturbation, 
                                 min_epoch, 
                                 min_dist_rhr, 
                                 min_dist_ahr, 
                                 min_dist_ohr, 
                                 window_ma, 
                                 category, 
                                 verbose)  
                
            # Category 1
            elif (epoch>nEval) and \
                 (d_ohr < thresholds["OHR"]) and (d_ahr < thresholds["AHR"]) and (d_rhr < thresholds["RHR"]):
                print('\n', flush=True)
                print(f'--->  Found minimum below threshold at Epoch={epoch} (Category 1): \n', flush=True)
                print('Last overall minimum dist=',min_dist_ohr, flush=True)
                print('Current overall minimum dist=',d_ohr, flush=True)
                print('Last AHR minimum dist=',min_dist_ahr, flush=True)
                print('Current AHR minimum dist=',d_ahr, flush=True)
                print('Last RHR minimum dist=',min_dist_rhr, flush=True)
                print('Current RHR minimum dist=',d_rhr, flush=True)
                print('\n', flush=True)

                min_dist_ohr = d_ohr
                min_dist_ahr = d_ahr
                min_dist_rhr = d_rhr
                min_epoch = epoch
                
                category_one_count = category_one_count + 1
                category = 1
                
                print("min_epoch_old=",min_epoch_old, flush=True)
                print("min_epoch=",min_epoch, flush=True)
                print("category_one_count=",category_one_count, flush=True)
                
                if (min_epoch-min_epoch_old >= nEval) or (category_one_count == 1):
                    min_epoch_old = min_epoch
                    self.__summary__(epoch, 
                                     noise,
                                     userdata,
                                     real_data, 
                                     real_data_with_perturbation, 
                                     min_epoch, 
                                     min_dist_rhr, 
                                     min_dist_ahr, 
                                     min_dist_ohr, 
                                     window_ma, 
                                     category, 
                                     verbose)  
            
            # Category 3
            elif (epoch >=10000) and (RHR_distance_max > 1.0 or RHR_distance_ratio > 100.0):
                print('\n', flush=True)
                print(f'--->  Distance diverged at Epoch={epoch} (Category 3): \n', flush=True)
                print('Last overall minimum dist=',min_dist_ohr, flush=True)
                print('Current overall minimum dist=',d_ohr, flush=True)
                print('Last AHR minimum dist=',min_dist_ahr, flush=True)
                print('Current AHR minimum dist=',d_ahr, flush=True)
                print('Last RHR minimum dist=',min_dist_rhr, flush=True)
                print('Current RHR minimum dist=',d_rhr, flush=True)
                print('\n', flush=True)
        
                # min_dist is already updated
                
                category = 3
                self.__summary__(epoch, 
                                 noise,
                                 userdata,
                                 real_data, 
                                 real_data_with_perturbation, 
                                 min_epoch, 
                                 min_dist_rhr, 
                                 min_dist_ahr, 
                                 min_dist_ohr, 
                                 window_ma, 
                                 category, 
                                 verbose)  
                break
            
            # Category 2 (might not be necessary, but it is good for clarity)
            elif (epoch == nEpochs):
                print('\n', flush=True)
                print(f'--->  Finished nEpochs={nEpochs} (Category 2): \n', flush=True)
                print('Last overall minimum dist=',min_dist_ohr, flush=True)
                print('Current overall minimum dist=',d_ohr, flush=True)
                print('Last AHR minimum dist=',min_dist_ahr, flush=True)
                print('Current AHR minimum dist=',d_ahr, flush=True)
                print('Last RHR minimum dist=',min_dist_rhr, flush=True)
                print('Current RHR minimum dist=',d_rhr, flush=True)
                print('\n', flush=True)
        
                # min_dist is already updated
                
                category = 2
                self.__summary__(epoch, 
                                 noise,
                                 userdata,
                                 real_data, 
                                 real_data_with_perturbation, 
                                 min_epoch, 
                                 min_dist_rhr, 
                                 min_dist_ahr, 
                                 min_dist_ohr, 
                                 window_ma, 
                                 category, 
                                 verbose)  
                
        print('\n', flush=True)
        print(f'Total time={str(timedelta(seconds=time()-start))}', flush=True)
        print('\n', flush=True)

        return min_dist_rhr, min_dist_ahr, min_dist_ohr, min_epoch

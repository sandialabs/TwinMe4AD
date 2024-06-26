import argparse
def Parser():
    # Setup of parser arguments
    parser = argparse.ArgumentParser(
             description="WGAN script for synthetic user generation")
    
    parser.add_argument("--users",
                        required=True,
                        nargs='+',
                        type=str,
                        help="Enter list of real users")

    parser.add_argument("--nEpochs",
                        #required=True,
                        type=int,
                        default=50000,
                        help="Enter maximum number of epochs per realization (default: 50000)")

    parser.add_argument("--nEval",
                        #required=True,
                        type=int,
                        default=1000,
                        help="Enter frequency (in epochs) to output results (default: 1000)")
    
    parser.add_argument("--epsilon",
                        #required=True,
                        nargs='+',
                        type=float,
                        default=0.0,
                        help="Enter list epsilons (default: [0.01])")
    
    parser.add_argument("--nrealizations",
                        #required=True,
                        type=int,
                        default=999,
                        help="Enter number of realizations to perform, per epsilon, per user (default: 999)")
    
    parser.add_argument("--RHR",
                        #required=True,
                        type=float,
                        default=0.015,
                        help="Enter Wasserstein distance threshold for RHR (default: 0.015)")
    
    parser.add_argument("--AHR",
                        #required=True,
                        type=float,
                        default=0.100,
                        help="Enter Wasserstein distance threshold for AHR (default: 0.100)")
    
    parser.add_argument("--OHR",
                        #required=True,
                        type=float,
                        default=0.007,
                        help="Enter Wasserstein distance threshold for OHR (default: 0.007)")
    
    parser.add_argument("--real_data_dir",
                        #required=True,
                        type=str,
                        default='../../Data/COVID-19-Wearables/',
                        help="Enter directory of real user data (default: ../../Data/COVID-19-Wearables/)")
    
    parser.add_argument("--output_data_dir",
                        #required=True,
                        type=str,
                        default='./results/',
                        help="Enter directory where to save results (i.e., img, data, model and summary subfolders; default: ./results/)")
    
    parser.add_argument("--synthetic_data_dir",
                        #required=True,
                        type=str,
                        default='./results/', help="Enter directory where to save synthetic data (default: ./results/)")
    
    args = parser.parse_args()
    
    print("**********************************************************")
    print("**********************************************************")
    print("                 WGAN for Digital Twins                   ")
    print("\n")
    print("  Input parameters:")
    print("    real_data_dir:",args.real_data_dir)
    print("    output_data_dir:",args.output_data_dir)
    print("    synthetic_data_dir:",args.synthetic_data_dir)
    print("    users:",args.users)
    print("    epsilon:",args.epsilon)
    print("    nrealizations:",args.nrealizations)
    print("    nEpochs:",args.nEpochs)
    print("    nEval:",args.nEval)
    print("    RHR:",args.RHR,
            ", AHR:",args.AHR,
            ", OHR:",args.OHR
         )
    print("\n")
    print("**********************************************************")
    print("**********************************************************")
    print("\n")
    
    return args

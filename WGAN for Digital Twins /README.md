# Synthetic User Generation

This script generates synthetic user data using a Wasserstein Generative Adversarial Network (WGAN) for Anomaly Detection. The main script to run the program is `generate_synthetic_users.py`.

## Table of Contents
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts Description](#details-of-scripts)

## Dependencies

The project requires the following dependencies:

- python 3.10.14 
- numpy 1.26.4
- pandas 2.2.2
- matplotlib 3.8.4
- tensorflow 2.16.1
- keras  3.3.3

## Installation

To install the required dependencies, follow these steps:

1. **Install Miniconda:**

   Download and install Miniconda from the official website: [Miniconda Installation](https://docs.conda.io/en/latest/miniconda.html).

2. **Create a new environment:**

    Open your terminal or command prompt and create a new environment with Python 3.10:

   ```
   conda create --name synthetic_user_gen python=3.10.14

3. **Activate the newly created environment:**
    
    In terminal, type the following command below to activate environment.
    
   ```
   conda activate synthetic_user_gen

4.  **Install the required dependencies:**

     In terminal, type the following command below to install dependencies.

    pip install numpy==1.26.4 pandas==2.2.2 matplotlib==3.8.4 tensorflow==2.16.1 keras==3.3.3

5.  **Download real user data:**
  
     Download real user data from the [COVID-19-Wearables](https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip) dataset.

## Usage 

1. **Clone the repository**

    In terminal, type the following command below to clone the repository.

    ```
    git clone <repository_url>
    cd <repository_folder>

2. **Specify real user data source**

    Change location of real user data source previously downloaded. This is specified by variable "data-dir" in script "generate_snythetic_users.py"

3. **Run the main script**

    In terminal, get into the folder that contains the "generate_snythetic_users.py" file and run it.

    ```
    python generate_snythetic_users.py

## Description Of Each File

`generate_synthetic_users.py`

This is the main script to run the synthetic user generation process


`Data.py`

This script contains class for data loading and preprocessing


`Model.py`

This script defines the GAN model architecture and builds, trains, and summarizes the models.

`WGAN.py`

This script implements a custom WGAN model and creates utility functions for generating synthetic data and saving the model. 


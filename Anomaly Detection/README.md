# GenAI based Digital Twins aided Data Augmentation Increases Accuracy in Real-Time Cokurtosis based Anomaly Detection of Wearable Data

Our paper focused on a novel runtime anomaly detection process using ``Cokurtosis and Hellinger distance`` based statistical modeling. We deployed our model on digital health which contains real world activity sensor data (steps, heart rate, and sleeps). To increase the data for our analysis, we created digital twin using generative AI (WGAN). We also created synthetic data of a user by adding controlled noise and digital twin of those synthetic users using our generative AI. From the spectrum of the actvity data, we choose resting heart rate (RHR: heart rate at inactive period) to analyze health anomaly. Besides individual level anomaly detection, we also explored uncertainty analysis by regulating the resting heart rate from *90* to *110* bpm. Along with RHR, the Hellinger distance is another indication of an anomaly event and we also emperically analyzed the uncertainty on Hellinger distance threshold.

# Real dataset
This publicly available data is collected from [here](https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip). Download the data and save to a folder. Lets give the folder name ``wearable``. Provide `write` permission to the ``wearable`` folder. Unzip the downloaded folder and place it as ``<PATH>/wearable/COVID-19-Wearables/``. This folder contains many csv files in following format:
```
<USER>_hr.csv       # Contains user, datetime, heartrate values
<USER>_steps.csv    # Contains user, datetime, steps values
<USER>_sleep.csv    # Contains user, datetime, duration, sleep type values
```

# Streaming data queue
To capture and process the realtime sensor data, we developed a time sensitive queue to store the data stream for $t_1$ minutes (we call it window size, $w$), generate features, and extract data for $t_2$ minutes (we call it sliding, $s$). In our experiment, we used two types of (window, sliding) combinations: (i) overlapping: when $s<w$, (ii) non-overlapping: when $s\ge w$. Our emperical analysis considering both cases concluded that there is no significant influence of these two parameter ($w$ and $s$) in our anomaly detection process. We used $w=60$ minutes and $s=30$ minutes for our uncertainty analysis.

# Uncertainty analysis
The following proprocessing steps are used for our uncertainty analysis:
1.  At first we create profile data considering all the users for a specific ($w$, $s$, RHR) selection. The following command is used to create all the files.
    ``` python
    >> python wearable_threshold_analysis.py --rangeRHR "[90, 110]" --window "[60, 120]" --fraction "[0.5, 1.0]" --no-multiprocess --event "RPD" --datapath '<PATH>/wearable/COVID-19-Wearables/'
    ```
    Instead of passing the sliding, we pass the ``--fraction`` parameter where sliding=window*fraction. 
    ``--no-multiprocess`` means code will run in serial order. No paralization will occur. To active paralization, pass ``--multiprocess``. It uses `90%` of the CPU threads/process to complete the tasks.

2. After completing the 1<sup>st</sup> step, the next command is used to compute `Hellinger distance` for each user. The following command is used for this:
    ``` python
    >> python wearable_threshold_analysis.py --rangeRHR "[90, 110]" --window "[60]" --fraction "[0.5]" --nCols "[2]" --resultpath "<PATH>/wearable/result/" --multiprocess --event "CHD" --datapath '<PATH>/wearable/COVID-19-Wearables/'
    ```
    The ``--nCols`` indicates the number of columns is used to compute Hellinger distance. The column count will start after the datetime column. This command will create folder under ``<PATH>/wearable/result/`` and store the results here.

3. After successfully completing last two steps, we compute ``confusion matrix`` for the range of RHR ($90\leq\alpha\leq 110$) and range of threshold ($0.005\leq\Delta\leq 0.02$). For each combination of ($\alpha,\Delta$), the following confusion matrix is used to compute the performance matrix (``F1 score`` and False negative rate (`FNR`)). 

    ||Healthy ($\mathcal{H}\leq\Delta$)|Sick ($\mathcal{H}>\Delta$)|
    | --- | --- | --- |
    |**Healthy** (RHR$\leq\alpha$)|*True positive* (TP)|*False negative* (FN)|
    |**Sick** (RHR$>\alpha$)|*False positive* (FP)|*True negative* (TN)|

    The command to execute this quere is as follow:
    ``` python
    >> python wearable_threshold_analysis.py --rangeRHR "[90, 110]" --window "[60]" --fraction "[0.5]" --nCols "[2]" --resultpath "<PATH>/wearable/" --multiprocess --event "CCM" --datapath '<PATH>/wearable/COVID-19-Wearables/'
    ```

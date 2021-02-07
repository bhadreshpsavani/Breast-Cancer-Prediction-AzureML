# Hyperparameter Tuning using HyperDrive

### 1. Import Dependencies:


```python
import logging
import os
import csv

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import pkg_resources

import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.core.compute import ComputeTarget, AmlCompute

from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn

from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive import TruncationSelectionPolicy
from azureml.train.hyperdrive import BayesianParameterSampling

from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
import os

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)
```

    SDK version: 1.20.0


### 2. Initialize Workspace


```python
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
```

    quick-starts-ws-137761
    aml-quickstarts-137761
    southcentralus
    81cefad3-d2c9-4f77-a466-99a7f541c7bb


### 3. Initialize Experiment


```python
ws = Workspace.from_config()
experiment_name = 'hyperdrive-experiment'
experiment=Experiment(ws, experiment_name)
experiment
```




<table style="width:100%"><tr><th>Name</th><th>Workspace</th><th>Report Page</th><th>Docs Page</th></tr><tr><td>hyperdrive-experiment</td><td>quick-starts-ws-137761</td><td><a href="https://ml.azure.com/experiments/hyperdrive-experiment?wsid=/subscriptions/81cefad3-d2c9-4f77-a466-99a7f541c7bb/resourcegroups/aml-quickstarts-137761/workspaces/quick-starts-ws-137761" target="_blank" rel="noopener">Link to Azure Machine Learning studio</a></td><td><a href="https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment.Experiment?view=azure-ml-py" target="_blank" rel="noopener">Link to Documentation</a></td></tr></table>



### 4. Create Compute Cluster


```python
cpu_cluster_name = "hyperdrive-compu"
vm_size='STANDARD_D2_V2'

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except:
    compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size, max_nodes=4)
    compute_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

# Can poll for a minimum number of nodes and for a specific timeout. 
# If no min node count is provided it uses the scale settings for the cluster.
compute_target.wait_for_completion(show_output=True)
```

    Found existing cluster, use it.
    Succeeded
    AmlCompute wait for completion finished
    
    Minimum number of nodes requested have been provisioned


### 5. Dataset




```python
data = datasets.load_breast_cancer()
print(data.data.shape)
print(data.feature_names)
print(data.DESCR)
```

    (569, 30)
    ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean smoothness' 'mean compactness' 'mean concavity'
     'mean concave points' 'mean symmetry' 'mean fractal dimension'
     'radius error' 'texture error' 'perimeter error' 'area error'
     'smoothness error' 'compactness error' 'concavity error'
     'concave points error' 'symmetry error' 'fractal dimension error'
     'worst radius' 'worst texture' 'worst perimeter' 'worst area'
     'worst smoothness' 'worst compactness' 'worst concavity'
     'worst concave points' 'worst symmetry' 'worst fractal dimension']
    .. _breast_cancer_dataset:
    
    Breast cancer wisconsin (diagnostic) dataset
    --------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry 
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    .. topic:: References
    
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.



```python
pd.Series(data.target).value_counts(normalize=True)
```




    1    0.627417
    0    0.372583
    dtype: float64




```python
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target']=data.target
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



## Hyperdrive Configuration

### Why Bayesian Sampling: 

Bayesian sampling is based on the Bayesian optimization algorithm. It picks samples based on how previous samples performed, so that new samples improve the primary metric.

Bayesian sampling is recommended if you have enough budget to explore the hyperparameter space. For best results, we recommend a maximum number of runs greater than or equal to 20 times the number of hyperparameters being tuned.

The number of concurrent runs has an impact on the effectiveness of the tuning process. A smaller number of concurrent runs may lead to better sampling convergence, since the smaller degree of parallelism increases the number of runs that benefit from previously completed runs.

### Early Stopping: 
Early stopping Policy is not implemented for Bayesian Sampling is not implemented for Hyperdrive


```python
# Specify parameter sampler
param_sampling = BayesianParameterSampling(
    parameter_space ={
        '--n_estimators' : choice(1,10,20,50,100,200,500),
        '--max_depth': choice(1, 5, 10, 20, 30, 50, 100),
        '--learning_rate': choice(1, 0.1, 0.01, 0.001)
        }
)

if "training" not in os.listdir():
    os.mkdir("./training")

# Create a SKLearn estimator for use with train.py
estimator = SKLearn(source_directory = "./",
            compute_target=compute_target,
            entry_script="train.py")

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_run_config = HyperDriveConfig(hyperparameter_sampling=param_sampling, 
                                     primary_metric_name='Accuracy',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     estimator=estimator,
                                     max_total_runs=80,
                                     max_concurrent_runs=12)
```

    WARNING:azureml.train.sklearn:'SKLearn' estimator is deprecated. Please use 'ScriptRunConfig' from 'azureml.core.script_run_config' with your own defined environment or the AzureML-Tutorial curated environment.


## Run Details

* We trained `GradientBoostingClassifier` model from sklearn with different values of parameters mentioned in above code. GradientBoosting based classifier has been shown significant results for many classfication problem. It is considered as powerful algorithm for classfication. It is build on top of DecisionTree Algorithm.

* Model is suppose to give different results for all the combination of parameters. We will select best performing model. 


```python
# Submit your hyperdrive run to the experiment and show run details with the widget.

# Start the HyperDrive run
hyperdrive_run = experiment.submit(hyperdrive_run_config)

# Monitor HyperDrive runs You can monitor the progress of the runs with the following Jupyter widget
RunDetails(hyperdrive_run).show()
```

    WARNING:root:If 'script' has been provided here and a script file name has been specified in 'run_config', 'script' provided in ScriptRunConfig initialization will take precedence.



    _HyperDriveWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO'…





```python
hyperdrive_run.wait_for_completion(show_output=True)
```

    RunId: HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792
    Web View: https://ml.azure.com/experiments/hyperdrive-experiment/runs/HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792?wsid=/subscriptions/81cefad3-d2c9-4f77-a466-99a7f541c7bb/resourcegroups/aml-quickstarts-137761/workspaces/quick-starts-ws-137761
    
    Streaming azureml-logs/hyperdrive.txt
    =====================================
    
    "<START>[2021-02-07T06:39:33.558671][API][INFO]Experiment created<END>\n""<START>[2021-02-07T06:39:34.678016][GENERATOR][INFO]Trying to sample '12' jobs from the hyperparameter space<END>\n""<START>[2021-02-07T06:39:35.061072][GENERATOR][INFO]Successfully sampled '12' jobs, they will soon be submitted to the execution target.<END>\n"<START>[2021-02-07T06:39:35.3275907Z][SCHEDULER][INFO]The execution environment is being prepared. Please be patient as it can take a few minutes.<END><START>[2021-02-07T06:40:06.6278469Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_1'<END><START>[2021-02-07T06:40:06.6266631Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_0'<END><START>[2021-02-07T06:40:06.6328824Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_5'<END><START>[2021-02-07T06:40:06.6337386Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_6'<END><START>[2021-02-07T06:40:06.6289389Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_2'<END><START>[2021-02-07T06:40:06.6348452Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_7'<END><START>[2021-02-07T06:40:06.6467018Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_11'<END><START>[2021-02-07T06:40:06.6300989Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_3'<END><START>[2021-02-07T06:40:06.6401772Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_8'<END><START>[2021-02-07T06:40:06.6407313Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_9'<END><START>[2021-02-07T06:40:06.6408933Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_10'<END><START>[2021-02-07T06:40:06.6260565Z][SCHEDULER][INFO]The execution environment was successfully prepared.<END><START>[2021-02-07T06:40:06.6312642Z][SCHEDULER][INFO]Scheduling job, id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_4'<END><START>[2021-02-07T06:40:07.4350858Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_0'<END><START>[2021-02-07T06:40:07.4710603Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_1'<END><START>[2021-02-07T06:40:07.4725194Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_6'<END><START>[2021-02-07T06:40:07.7260132Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_7'<END><START>[2021-02-07T06:40:07.8782992Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_11'<END><START>[2021-02-07T06:40:07.9406660Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_4'<END><START>[2021-02-07T06:40:08.0838761Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_3'<END><START>[2021-02-07T06:40:08.2177959Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_8'<END><START>[2021-02-07T06:40:08.2726709Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_10'<END><START>[2021-02-07T06:40:08.8707781Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_5'<END><START>[2021-02-07T06:40:09.0915028Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_2'<END><START>[2021-02-07T06:40:09.3010804Z][SCHEDULER][INFO]Successfully scheduled a job. Id='HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_9'<END>
    
    Execution Summary
    =================
    RunId: HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792
    Web View: https://ml.azure.com/experiments/hyperdrive-experiment/runs/HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792?wsid=/subscriptions/81cefad3-d2c9-4f77-a466-99a7f541c7bb/resourcegroups/aml-quickstarts-137761/workspaces/quick-starts-ws-137761
    





    {'runId': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792',
     'target': 'hyperdrive-compu',
     'status': 'Completed',
     'startTimeUtc': '2021-02-07T06:39:33.034036Z',
     'endTimeUtc': '2021-02-07T06:58:55.855673Z',
     'properties': {'primary_metric_config': '{"name": "Accuracy", "goal": "maximize"}',
      'resume_from': 'null',
      'runTemplate': 'HyperDrive',
      'azureml.runsource': 'hyperdrive',
      'platform': 'AML',
      'ContentSnapshotId': '8b4cf1cd-ef72-43aa-b581-a71b8388c782',
      'score': '0.9790209790209791',
      'best_child_run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_11',
      'best_metric_status': 'Succeeded'},
     'inputDatasets': [],
     'outputDatasets': [],
     'logFiles': {'azureml-logs/hyperdrive.txt': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792/azureml-logs/hyperdrive.txt?sv=2019-02-02&sr=b&sig=y%2BFOms0E8aU%2F8MCkFgijkKtevn1a1I4fF0eSYc1wW3Q%3D&st=2021-02-07T06%3A49%3A23Z&se=2021-02-07T14%3A59%3A23Z&sp=r'},
     'submittedBy': 'ODL_User 137761'}



## Best Model

Get the best model from the hyperdrive experiments and display all the properties of the model.


```python
hyperdrive_run.get_hyperparameters()
```




    {'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_0': '{"--n_estimators": 500, "--max_depth": 20, "--learning_rate": 0.1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_1': '{"--n_estimators": 20, "--max_depth": 20, "--learning_rate": 0.01}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_2': '{"--n_estimators": 1, "--max_depth": 5, "--learning_rate": 0.001}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_3': '{"--n_estimators": 1, "--max_depth": 50, "--learning_rate": 0.01}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_4': '{"--n_estimators": 1, "--max_depth": 30, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_5': '{"--n_estimators": 100, "--max_depth": 10, "--learning_rate": 0.1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_6': '{"--n_estimators": 20, "--max_depth": 10, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_7': '{"--n_estimators": 50, "--max_depth": 100, "--learning_rate": 0.001}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_8': '{"--n_estimators": 50, "--max_depth": 50, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_9': '{"--n_estimators": 1, "--max_depth": 10, "--learning_rate": 0.01}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_10': '{"--learning_rate": 0.01, "--max_depth": 20, "--n_estimators": 500}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_11': '{"--learning_rate": 0.1, "--max_depth": 1, "--n_estimators": 500}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_12': '{"--learning_rate": 1, "--max_depth": 5, "--n_estimators": 50}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_13': '{"--learning_rate": 0.001, "--max_depth": 100, "--n_estimators": 500}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_14': '{"--learning_rate": 1, "--max_depth": 20, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_15': '{"--learning_rate": 1, "--max_depth": 10, "--n_estimators": 10}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_16': '{"--learning_rate": 0.1, "--max_depth": 50, "--n_estimators": 10}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_17': '{"--learning_rate": 0.1, "--max_depth": 1, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_18': '{"--n_estimators": 500, "--max_depth": 20, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_19': '{"--n_estimators": 500, "--max_depth": 1, "--learning_rate": 0.001}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_20': '{"--n_estimators": 500, "--max_depth": 1, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_21': '{"--n_estimators": 500, "--max_depth": 5, "--learning_rate": 0.1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_22': '{"--learning_rate": 0.01, "--max_depth": 100, "--n_estimators": 500}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_23': '{"--learning_rate": 0.01, "--max_depth": 100, "--n_estimators": 50}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_24': '{"--n_estimators": 500, "--max_depth": 5, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_25': '{"--n_estimators": 20, "--max_depth": 5, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_26': '{"--learning_rate": 0.1, "--max_depth": 50, "--n_estimators": 200}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_27': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 10}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_28': '{"--learning_rate": 0.01, "--max_depth": 30, "--n_estimators": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_29': '{"--learning_rate": 1, "--max_depth": 1, "--n_estimators": 50}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_30': '{"--learning_rate": 0.01, "--max_depth": 100, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_31': '{"--learning_rate": 1, "--max_depth": 100, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_32': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_33': '{"--learning_rate": 1, "--max_depth": 1, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_34': '{"--n_estimators": 500, "--max_depth": 20, "--learning_rate": 0.001}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_35': '{"--learning_rate": 1, "--max_depth": 30, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_36': '{"--learning_rate": 1, "--max_depth": 10, "--n_estimators": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_37': '{"--learning_rate": 0.1, "--max_depth": 20, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_38': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 500}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_39': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 50}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_40': '{"--n_estimators": 1, "--max_depth": 100, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_41': '{"--learning_rate": 0.1, "--max_depth": 5, "--n_estimators": 200}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_42': '{"--learning_rate": 0.01, "--max_depth": 1, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_43': '{"--learning_rate": 0.001, "--max_depth": 30, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_44': '{"--n_estimators": 500, "--max_depth": 10, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_45': '{"--learning_rate": 0.01, "--max_depth": 30, "--n_estimators": 10}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_46': '{"--learning_rate": 0.01, "--max_depth": 5, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_47': '{"--n_estimators": 200, "--max_depth": 30, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_48': '{"--learning_rate": 0.01, "--max_depth": 10, "--n_estimators": 10}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_49': '{"--learning_rate": 0.1, "--max_depth": 100, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_50': '{"--n_estimators": 200, "--max_depth": 1, "--learning_rate": 0.001}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_51': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_52': '{"--learning_rate": 1, "--max_depth": 20, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_53': '{"--learning_rate": 0.01, "--max_depth": 50, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_54': '{"--learning_rate": 0.1, "--max_depth": 20, "--n_estimators": 200}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_55': '{"--learning_rate": 1, "--max_depth": 30, "--n_estimators": 10}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_56': '{"--learning_rate": 0.1, "--max_depth": 30, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_57': '{"--n_estimators": 1, "--max_depth": 1, "--learning_rate": 0.001}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_58': '{"--learning_rate": 0.01, "--max_depth": 50, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_59': '{"--learning_rate": 0.01, "--max_depth": 20, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_60': '{"--learning_rate": 0.001, "--max_depth": 30, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_61': '{"--learning_rate": 1, "--max_depth": 50, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_62': '{"--learning_rate": 0.1, "--max_depth": 100, "--n_estimators": 10}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_63': '{"--learning_rate": 1, "--max_depth": 10, "--n_estimators": 200}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_64': '{"--n_estimators": 1, "--max_depth": 1, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_65': '{"--learning_rate": 1, "--max_depth": 1, "--n_estimators": 10}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_66': '{"--learning_rate": 0.01, "--max_depth": 100, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_67': '{"--learning_rate": 0.1, "--max_depth": 50, "--n_estimators": 20}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_68': '{"--learning_rate": 0.1, "--max_depth": 10, "--n_estimators": 10}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_69': '{"--learning_rate": 0.001, "--max_depth": 10, "--n_estimators": 200}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_70': '{"--n_estimators": 500, "--max_depth": 100, "--learning_rate": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_71': '{"--learning_rate": 0.001, "--max_depth": 100, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_72': '{"--learning_rate": 0.01, "--max_depth": 1, "--n_estimators": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_73': '{"--learning_rate": 0.001, "--max_depth": 100, "--n_estimators": 200}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_74': '{"--learning_rate": 0.001, "--max_depth": 5, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_75': '{"--learning_rate": 0.001, "--max_depth": 10, "--n_estimators": 1}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_76': '{"--learning_rate": 1, "--max_depth": 20, "--n_estimators": 200}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_77': '{"--learning_rate": 0.001, "--max_depth": 5, "--n_estimators": 500}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_78': '{"--learning_rate": 0.01, "--max_depth": 5, "--n_estimators": 100}',
     'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_79': '{"--learning_rate": 0.1, "--max_depth": 20, "--n_estimators": 10}'}




```python
hyperdrive_run.get_children_sorted_by_primary_metric()
```




    [{'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_11',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 1, "--n_estimators": 500}',
      'best_primary_metric': 0.9790209790209791,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_33',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 1, "--n_estimators": 100}',
      'best_primary_metric': 0.965034965034965,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_20',
      'hyperparameters': '{"--n_estimators": 500, "--max_depth": 1, "--learning_rate": 1}',
      'best_primary_metric': 0.965034965034965,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_29',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 1, "--n_estimators": 50}',
      'best_primary_metric': 0.958041958041958,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_65',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 1, "--n_estimators": 10}',
      'best_primary_metric': 0.9440559440559441,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_24',
      'hyperparameters': '{"--n_estimators": 500, "--max_depth": 5, "--learning_rate": 1}',
      'best_primary_metric': 0.9300699300699301,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_25',
      'hyperparameters': '{"--n_estimators": 20, "--max_depth": 5, "--learning_rate": 1}',
      'best_primary_metric': 0.9300699300699301,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_12',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 5, "--n_estimators": 50}',
      'best_primary_metric': 0.9300699300699301,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_23',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 100, "--n_estimators": 50}',
      'best_primary_metric': 0.9230769230769231,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_79',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 20, "--n_estimators": 10}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_77',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 5, "--n_estimators": 500}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_68',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 10, "--n_estimators": 10}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_67',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 50, "--n_estimators": 20}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_62',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 100, "--n_estimators": 10}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_56',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 30, "--n_estimators": 100}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_54',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 20, "--n_estimators": 200}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_49',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 100, "--n_estimators": 100}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_38',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 500}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_37',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 20, "--n_estimators": 20}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_34',
      'hyperparameters': '{"--n_estimators": 500, "--max_depth": 20, "--learning_rate": 0.001}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_26',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 50, "--n_estimators": 200}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_16',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 50, "--n_estimators": 10}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_13',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 100, "--n_estimators": 500}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_5',
      'hyperparameters': '{"--n_estimators": 100, "--max_depth": 10, "--learning_rate": 0.1}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_0',
      'hyperparameters': '{"--n_estimators": 500, "--max_depth": 20, "--learning_rate": 0.1}',
      'best_primary_metric': 0.916083916083916,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_78',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 5, "--n_estimators": 100}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_76',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 20, "--n_estimators": 200}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_70',
      'hyperparameters': '{"--n_estimators": 500, "--max_depth": 100, "--learning_rate": 1}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_66',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 100, "--n_estimators": 100}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_61',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 50, "--n_estimators": 20}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_63',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 10, "--n_estimators": 200}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_59',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 20, "--n_estimators": 100}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_58',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 50, "--n_estimators": 100}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_52',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 20, "--n_estimators": 20}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_55',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 30, "--n_estimators": 10}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_47',
      'hyperparameters': '{"--n_estimators": 200, "--max_depth": 30, "--learning_rate": 1}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_44',
      'hyperparameters': '{"--n_estimators": 500, "--max_depth": 10, "--learning_rate": 1}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_40',
      'hyperparameters': '{"--n_estimators": 1, "--max_depth": 100, "--learning_rate": 1}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_36',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 10, "--n_estimators": 1}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_35',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 30, "--n_estimators": 20}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_31',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 100, "--n_estimators": 20}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_22',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 100, "--n_estimators": 500}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_18',
      'hyperparameters': '{"--n_estimators": 500, "--max_depth": 20, "--learning_rate": 1}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_15',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 10, "--n_estimators": 10}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_14',
      'hyperparameters': '{"--learning_rate": 1, "--max_depth": 20, "--n_estimators": 100}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_10',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 20, "--n_estimators": 500}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_8',
      'hyperparameters': '{"--n_estimators": 50, "--max_depth": 50, "--learning_rate": 1}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_4',
      'hyperparameters': '{"--n_estimators": 1, "--max_depth": 30, "--learning_rate": 1}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_6',
      'hyperparameters': '{"--n_estimators": 20, "--max_depth": 10, "--learning_rate": 1}',
      'best_primary_metric': 0.9090909090909091,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_41',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 5, "--n_estimators": 200}',
      'best_primary_metric': 0.9020979020979021,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_21',
      'hyperparameters': '{"--n_estimators": 500, "--max_depth": 5, "--learning_rate": 0.1}',
      'best_primary_metric': 0.9020979020979021,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_19',
      'hyperparameters': '{"--n_estimators": 500, "--max_depth": 1, "--learning_rate": 0.001}',
      'best_primary_metric': 0.8951048951048951,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_17',
      'hyperparameters': '{"--learning_rate": 0.1, "--max_depth": 1, "--n_estimators": 20}',
      'best_primary_metric': 0.8951048951048951,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_64',
      'hyperparameters': '{"--n_estimators": 1, "--max_depth": 1, "--learning_rate": 1}',
      'best_primary_metric': 0.8881118881118881,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_75',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 10, "--n_estimators": 1}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_73',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 100, "--n_estimators": 200}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_74',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 5, "--n_estimators": 100}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_72',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 1, "--n_estimators": 1}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_69',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 10, "--n_estimators": 200}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_71',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 100, "--n_estimators": 100}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_60',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 30, "--n_estimators": 20}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_57',
      'hyperparameters': '{"--n_estimators": 1, "--max_depth": 1, "--learning_rate": 0.001}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_53',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 50, "--n_estimators": 20}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_51',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 20}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_50',
      'hyperparameters': '{"--n_estimators": 200, "--max_depth": 1, "--learning_rate": 0.001}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_48',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 10, "--n_estimators": 10}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_45',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 30, "--n_estimators": 10}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_46',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 5, "--n_estimators": 20}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_43',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 30, "--n_estimators": 100}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_42',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 1, "--n_estimators": 20}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_39',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 50}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_32',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 100}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_30',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 100, "--n_estimators": 20}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_28',
      'hyperparameters': '{"--learning_rate": 0.01, "--max_depth": 30, "--n_estimators": 1}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_27',
      'hyperparameters': '{"--learning_rate": 0.001, "--max_depth": 50, "--n_estimators": 10}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_2',
      'hyperparameters': '{"--n_estimators": 1, "--max_depth": 5, "--learning_rate": 0.001}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_3',
      'hyperparameters': '{"--n_estimators": 1, "--max_depth": 50, "--learning_rate": 0.01}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_9',
      'hyperparameters': '{"--n_estimators": 1, "--max_depth": 10, "--learning_rate": 0.01}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_7',
      'hyperparameters': '{"--n_estimators": 50, "--max_depth": 100, "--learning_rate": 0.001}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_1',
      'hyperparameters': '{"--n_estimators": 20, "--max_depth": 20, "--learning_rate": 0.01}',
      'best_primary_metric': 0.6433566433566433,
      'status': 'Completed'},
     {'run_id': 'HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_preparation',
      'hyperparameters': None,
      'best_primary_metric': None,
      'status': 'Completed'}]




```python
from azureml.core.model import Model

### YOUR CODE HERE ###
best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()['Accuracy']
parameter_values = best_run.get_details()['runDefinition']['arguments']

print('Best Run Id: ', best_run.id)
print('Accuracy:', best_run_metrics)
print('learning_rate:',parameter_values[1])
print('max_depth:',parameter_values[3])
print('n_estimators:',parameter_values[5])
```

    Best Run Id:  HD_75a6838b-57a7-4a3e-9562-dd6c0a9c8792_11
    Accuracy: 0.9790209790209791
    learning_rate: 0.1
    max_depth: 1
    n_estimators: 500



```python
parameter_values
```




    ['--learning_rate', '0.1', '--max_depth', '1', '--n_estimators', '500']




```python
#TODO: Save the best model
best_run.download_file("/outputs/model.joblib", "Hyperdrive.joblib")
```

## Model Deployment

Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.

TODO: In the cell below, register the model, create an inference config and deploy the model as a web service.


```python

```

TODO: In the cell below, send a request to the web service you deployed to test it.


```python

```

TODO: In the cell below, print the logs of the web service and delete the service


```python

```

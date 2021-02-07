# Automated ML

## Import Dependencies


```python
import os
import csv
from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import pkg_resources

import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
from azureml.core.compute import ComputeTarget, AmlCompute

from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn

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
experiment_name = 'auto-experiment'
experiment=Experiment(ws, experiment_name)
experiment
```




<table style="width:100%"><tr><th>Name</th><th>Workspace</th><th>Report Page</th><th>Docs Page</th></tr><tr><td>auto-experiment</td><td>quick-starts-ws-137761</td><td><a href="https://ml.azure.com/experiments/auto-experiment?wsid=/subscriptions/81cefad3-d2c9-4f77-a466-99a7f541c7bb/resourcegroups/aml-quickstarts-137761/workspaces/quick-starts-ws-137761" target="_blank" rel="noopener">Link to Azure Machine Learning studio</a></td><td><a href="https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment.Experiment?view=azure-ml-py" target="_blank" rel="noopener">Link to Documentation</a></td></tr></table>



### 4. Create Compute Cluster


```python
cpu_cluster_name = "hyperdrive-compu"
vm_size='STANDARD_D14_V2'

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except:
    compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size, max_nodes=10)
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




```python
pd.Series(data.target).value_counts(normalize=True)
```




    1    0.627417
    0    0.372583
    dtype: float64



**Our dataset is slightly imbalanced**


```python
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.data.datapath import DataPath
# Create TabularDataset using TabularDatasetFactory
def_blob_store = ws.get_default_datastore()
print("Default datastore's name: {}".format(def_blob_store.name))
data_path = DataPath(datastore=def_blob_store, path_on_datastore='datapath')
ds = TabularDatasetFactory.register_pandas_dataframe(df, name='UCI_ML_Breast_Cancer', target=data_path)
```

    Method register_pandas_dataframe: This is an experimental method, and may change at any time.<br/>For more information, see https://aka.ms/azuremlexperimental.


    Default datastore's name: workspaceblobstore
    Validating arguments.
    Arguments validated.
    Successfully obtained datastore reference and path.
    Uploading file to datapath/5e6d4177-8965-4e99-83a2-a9190e1be837/
    Successfully uploaded file to datastore.
    Creating and registering a new dataset.
    Successfully created and registered a new dataset.


## AutoML Configuration:

Our task is `classification`


```python
automl_settings = {
    "experiment_timeout_minutes": 25,
    "max_concurrent_iterations": 10,
    "primary_metric" : 'accuracy'}

automl_config = AutoMLConfig(
    task='classification',
    training_data = ds,
    label_column_name = "target",
    compute_target=compute_target,
    **automl_settings)
```

## Run Details


```python
# TODO: Submit your experiment
remote_run = experiment.submit(automl_config)

RunDetails(remote_run).show()
```

    Running on remote.



    _AutoMLWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO', 's…





```python
remote_run.wait_for_completion(show_output=True)
```

    
    Current status: FeaturesGeneration. Generating features for the dataset.
    Current status: DatasetCrossValidationSplit. Generating individually featurized CV splits.
    Current status: ModelSelection. Beginning model selection.
    
    ****************************************************************************************************
    DATA GUARDRAILS: 
    
    TYPE:         Cross validation
    STATUS:       DONE
    DESCRIPTION:  Each iteration of the trained model was validated through cross-validation.
                  
    DETAILS:      
    +---------------------------------+
    |Number of folds                  |
    +=================================+
    |10                               |
    +---------------------------------+
    
    ****************************************************************************************************
    
    TYPE:         Class balancing detection
    STATUS:       PASSED
    DESCRIPTION:  Your inputs were analyzed, and all classes are balanced in your training data.
                  Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData
    
    ****************************************************************************************************
    
    TYPE:         Missing feature values imputation
    STATUS:       PASSED
    DESCRIPTION:  No feature missing values were detected in the training data.
                  Learn more about missing value imputation: https://aka.ms/AutomatedMLFeaturization
    
    ****************************************************************************************************
    
    TYPE:         High cardinality feature detection
    STATUS:       PASSED
    DESCRIPTION:  Your inputs were analyzed, and no high cardinality features were detected.
                  Learn more about high cardinality feature handling: https://aka.ms/AutomatedMLFeaturization
    
    ****************************************************************************************************
    
    ****************************************************************************************************
    ITERATION: The iteration being evaluated.
    PIPELINE: A summary description of the pipeline being evaluated.
    DURATION: Time taken for the current iteration.
    METRIC: The result of computing score on the fitted pipeline.
    BEST: The best observed score thus far.
    ****************************************************************************************************
    
     ITERATION   PIPELINE                                       DURATION      METRIC      BEST
             4   MinMaxScaler RandomForest                      0:00:50       0.9543    0.9543
             3   MinMaxScaler RandomForest                      0:00:59       0.9438    0.9543
            11   MinMaxScaler SVM                               0:00:52       0.9630    0.9630
            10   SparseNormalizer XGBoostClassifier             0:01:49       0.9666    0.9666
             7   SparseNormalizer LightGBM                      0:01:51       0.9508    0.9666
             8   StandardScalerWrapper XGBoostClassifier        0:01:51       0.9560    0.9666
             9   StandardScalerWrapper XGBoostClassifier        0:01:57       0.9666    0.9666
             0   MaxAbsScaler LightGBM                          0:01:58       0.9701    0.9701
             1   MaxAbsScaler XGBoostClassifier                 0:02:06       0.9683    0.9701
             2   MinMaxScaler RandomForest                      0:02:04       0.9525    0.9701
             5   SparseNormalizer XGBoostClassifier             0:01:58       0.9543    0.9701
            12   MaxAbsScaler LightGBM                          0:00:57       0.9560    0.9701
            13   SparseNormalizer XGBoostClassifier             0:00:51       0.9736    0.9736
            14   SparseNormalizer XGBoostClassifier             0:00:56       0.9666    0.9736
            16   SparseNormalizer XGBoostClassifier             0:00:48       0.9701    0.9736
            17   StandardScalerWrapper LightGBM                 0:00:47       0.9331    0.9736
             6   SparseNormalizer XGBoostClassifier             0:02:58       0.9701    0.9736
            15   SparseNormalizer XGBoostClassifier             0:00:57       0.9648    0.9736
            18   StandardScalerWrapper XGBoostClassifier        0:00:54       0.9385    0.9736
            19   MinMaxScaler LightGBM                          0:00:49       0.9525    0.9736
            20   RobustScaler LightGBM                          0:00:51       0.9578    0.9736
            21   RobustScaler RandomForest                      0:00:51       0.9490    0.9736
            23   MinMaxScaler LightGBM                          0:00:50       0.9543    0.9736
            22   RobustScaler KNN                               0:01:00       0.9367    0.9736
            24   MaxAbsScaler RandomForest                      0:00:57       0.9349    0.9736
            25   PCA XGBoostClassifier                          0:00:57       0.8822    0.9736
            26   StandardScalerWrapper LightGBM                 0:00:58       0.9525    0.9736
            27   SparseNormalizer LightGBM                      0:00:50       0.9648    0.9736
            28   MinMaxScaler LightGBM                          0:00:46       0.9349    0.9736
            29   SparseNormalizer XGBoostClassifier             0:01:02       0.6273    0.9736
            30   SparseNormalizer XGBoostClassifier             0:00:55       0.9684    0.9736
            31   MinMaxScaler LightGBM                          0:00:48       0.9683    0.9736
            32   StandardScalerWrapper ExtremeRandomTrees       0:00:57       0.9542    0.9736
            33   StandardScalerWrapper RandomForest             0:00:57       0.9508    0.9736
            34   StandardScalerWrapper XGBoostClassifier        0:00:53       0.6273    0.9736
            35   SparseNormalizer XGBoostClassifier             0:00:56       0.9684    0.9736
            36   SparseNormalizer XGBoostClassifier             0:00:52       0.9596    0.9736
            37   SparseNormalizer XGBoostClassifier             0:01:05       0.9331    0.9736
            38   SparseNormalizer XGBoostClassifier             0:00:53       0.9543    0.9736
            39   SparseNormalizer XGBoostClassifier             0:00:54       0.9578    0.9736
            40   StandardScalerWrapper XGBoostClassifier        0:00:55       0.9595    0.9736
            41   SparseNormalizer XGBoostClassifier             0:00:49       0.9631    0.9736
            42   MinMaxScaler LogisticRegression                0:00:51       0.9718    0.9736
            43   SparseNormalizer XGBoostClassifier             0:00:57       0.9578    0.9736
            44   SparseNormalizer RandomForest                  0:00:52       0.9419    0.9736
            45   StandardScalerWrapper XGBoostClassifier        0:00:51       0.9578    0.9736
            46   MinMaxScaler LightGBM                          0:00:49       0.9384    0.9736
            47   MinMaxScaler LightGBM                          0:00:53       0.9684    0.9736
            48   SparseNormalizer XGBoostClassifier             0:00:58       0.9525    0.9736
            49   StandardScalerWrapper LogisticRegression       0:00:58       0.9771    0.9771
            50   StandardScalerWrapper XGBoostClassifier        0:00:57       0.9560    0.9771
            51   MaxAbsScaler LogisticRegression                0:00:50       0.9647    0.9771
            52   MaxAbsScaler RandomForest                      0:00:59       0.9543    0.9771
            53   MinMaxScaler LogisticRegression                0:00:51       0.9736    0.9771
            54   StandardScalerWrapper XGBoostClassifier        0:00:55       0.9560    0.9771
            55   StandardScalerWrapper XGBoostClassifier        0:00:52       0.9507    0.9771
            56   MaxAbsScaler ExtremeRandomTrees                0:00:49       0.9419    0.9771
            57   StandardScalerWrapper XGBoostClassifier        0:00:53       0.9595    0.9771
            58   StandardScalerWrapper XGBoostClassifier        0:00:52       0.9560    0.9771
            59   SparseNormalizer XGBoostClassifier             0:00:47       0.9402    0.9771
            60   StandardScalerWrapper LogisticRegression       0:01:01       0.9735    0.9771
            61   StandardScalerWrapper ExtremeRandomTrees       0:00:52       0.9420    0.9771
            62   SparseNormalizer XGBoostClassifier             0:00:54       0.9507    0.9771
            63   RobustScaler ExtremeRandomTrees                0:00:50       0.9630    0.9771
            64   MinMaxScaler LogisticRegression                0:00:55       0.9806    0.9806
            65   SparseNormalizer XGBoostClassifier             0:00:49       0.9596    0.9806
            66   SparseNormalizer XGBoostClassifier             0:00:53       0.9613    0.9806
            67   MaxAbsScaler ExtremeRandomTrees                0:00:54       0.9648    0.9806
            68   SparseNormalizer XGBoostClassifier             0:00:51       0.9613    0.9806
            70   MaxAbsScaler LogisticRegression                0:00:56       0.9718    0.9806
            71   StandardScalerWrapper XGBoostClassifier        0:00:52       0.9613    0.9806
            69   StandardScalerWrapper SGD                      0:01:44       0.9824    0.9824
            72   SparseNormalizer XGBoostClassifier             0:00:52       0.9719    0.9824
            74   StandardScalerWrapper LogisticRegression       0:00:54       0.9841    0.9841
            75   StandardScalerWrapper LogisticRegression       0:00:48       0.9701    0.9841
            73   SparseNormalizer XGBoostClassifier             0:01:00       0.9648    0.9841
            76   StandardScalerWrapper LogisticRegression       0:00:51       0.9736    0.9841
            77   MaxAbsScaler LogisticRegression                0:00:49       0.9701    0.9841
            78   StandardScalerWrapper LogisticRegression       0:00:53       0.9736    0.9841
            79   MinMaxScaler LightGBM                          0:00:51       0.9648    0.9841
            80   MaxAbsScaler LogisticRegression                0:00:58       0.9736    0.9841
            81   SparseNormalizer LightGBM                      0:00:48       0.9578    0.9841
            82   RobustScaler LogisticRegression                0:00:52       0.9700    0.9841
            83   StandardScalerWrapper LogisticRegression       0:01:02       0.9789    0.9841
            84   MaxAbsScaler LightGBM                          0:00:52       0.9455    0.9841
            85   SparseNormalizer LogisticRegression            0:00:50       0.9155    0.9841
            86   StandardScalerWrapper LogisticRegression       0:00:56       0.9577    0.9841
            87   MaxAbsScaler LogisticRegression                0:00:56       0.9718    0.9841
            88   StandardScalerWrapper LogisticRegression       0:01:00       0.9806    0.9841
            89   SparseNormalizer XGBoostClassifier             0:01:05       0.9648    0.9841
            90   StandardScalerWrapper LogisticRegression       0:00:53       0.9701    0.9841
            91   MaxAbsScaler LogisticRegression                0:00:51       0.9718    0.9841
            92   SparseNormalizer LightGBM                      0:00:52       0.9631    0.9841
           101   MaxAbsScaler LogisticRegression                0:00:50       0.9701    0.9841
            93   StandardScalerWrapper SGD                      0:01:00       0.9771    0.9841
            94   RobustScaler SVM                               0:00:58       0.9665    0.9841
            95   StandardScalerWrapper LogisticRegression       0:01:17       0.9701    0.9841
            96   StandardScalerWrapper LogisticRegression       0:00:51       0.9701    0.9841
            97   SparseNormalizer XGBoostClassifier             0:00:57       0.9684    0.9841
            98   StandardScalerWrapper LogisticRegression       0:00:54       0.9841    0.9841
           100   MinMaxScaler LogisticRegression                0:00:56       0.9718    0.9841
           102   StandardScalerWrapper XGBoostClassifier        0:00:49       0.9648    0.9841
           103   StandardScalerWrapper LogisticRegression       0:00:55       0.9806    0.9841
           104   StandardScalerWrapper LogisticRegression       0:00:57       0.9701    0.9841
           105   StandardScalerWrapper SGD                      0:00:54       0.9771    0.9841
           106   MinMaxScaler LogisticRegression                0:00:55       0.9736    0.9841
            99   StandardScalerWrapper LightGBM                 0:00:59       0.9630    0.9841
           107   StandardScalerWrapper LogisticRegression       0:01:07       0.9736    0.9841
           108   StandardScalerWrapper LightGBM                 0:00:52       0.9701    0.9841
           109   StandardScalerWrapper LogisticRegression       0:00:59       0.9736    0.9841
           110   StandardScalerWrapper LogisticRegression       0:00:49       0.9753    0.9841
           111   StandardScalerWrapper LogisticRegression       0:00:46       0.9841    0.9841
           112   RobustScaler LogisticRegression                0:01:05       0.9841    0.9841
           113   StandardScalerWrapper LogisticRegression       0:00:57       0.9841    0.9841
           114   MaxAbsScaler LogisticRegression                0:00:54       0.9736    0.9841
           115   MinMaxScaler LogisticRegression                0:00:54       0.9718    0.9841
           116   MaxAbsScaler SVM                               0:00:48       0.9771    0.9841
           117   StandardScalerWrapper LogisticRegression       0:00:57       0.9753    0.9841
           118   StandardScalerWrapper LogisticRegression       0:01:15       0.9806    0.9841
           119   StandardScalerWrapper LogisticRegression       0:00:56       0.9841    0.9841
           122   StandardScalerWrapper LogisticRegression       0:00:53       0.9824    0.9841
           123   StandardScalerWrapper XGBoostClassifier        0:00:57       0.9665    0.9841
           124   MinMaxScaler LogisticRegression                0:00:53       0.9789    0.9841
           126   RobustScaler LogisticRegression                0:00:52       0.9471    0.9841
           120   StandardScalerWrapper LogisticRegression       0:01:34       0.9701    0.9841
           121   StandardScalerWrapper SGD                      0:01:25       0.9789    0.9841
           125   StandardScalerWrapper LogisticRegression       0:00:58       0.9226    0.9841
           127   StandardScalerWrapper LogisticRegression       0:00:48       0.9824    0.9841
           129   StandardScalerWrapper LogisticRegression       0:00:58       0.9806    0.9841
           130   MinMaxScaler LogisticRegression                0:00:49       0.9718    0.9841
           131   StandardScalerWrapper LogisticRegression       0:00:53       0.9824    0.9841
           128   MaxAbsScaler LogisticRegression                0:01:30       0.9736    0.9841
           133   MaxAbsScaler KNN                               0:00:55       0.9718    0.9841
           134   MinMaxScaler LogisticRegression                0:00:52       0.9736    0.9841
           137   StandardScalerWrapper XGBoostClassifier        0:00:54       0.9683    0.9841
           132   StandardScalerWrapper LogisticRegression       0:01:28       0.9718    0.9841
           136   StandardScalerWrapper XGBoostClassifier        0:01:13       0.9560    0.9841
           135   MinMaxScaler ExtremeRandomTrees                0:01:34       0.9613    0.9841
           138   StandardScalerWrapper XGBoostClassifier        0:01:01       0.9630    0.9841
           139   StandardScalerWrapper XGBoostClassifier        0:00:54       0.9596    0.9841
           140   MinMaxScaler LogisticRegression                0:00:55       0.9718    0.9841
           141   StandardScalerWrapper XGBoostClassifier        0:00:55       0.9683    0.9841
           142   MinMaxScaler LogisticRegression                0:00:52       0.9789    0.9841
           143   MinMaxScaler LogisticRegression                0:01:08       0.9771    0.9841
           144   StandardScalerWrapper LogisticRegression       0:00:54       0.9824    0.9841
           145   MaxAbsScaler LogisticRegression                0:00:56       0.9718    0.9841
           147   StandardScalerWrapper LogisticRegression       0:00:52       0.9824    0.9841
           148   RobustScaler LogisticRegression                0:00:55       0.9629    0.9841
           149   StandardScalerWrapper LogisticRegression       0:00:55       0.9718    0.9841
           150   StandardScalerWrapper LogisticRegression       0:00:54       0.9701    0.9841
           146   StandardScalerWrapper LogisticRegression       0:01:39       0.9824    0.9841
           151   StandardScalerWrapper LogisticRegression       0:00:55       0.9824    0.9841
           152   MinMaxScaler LogisticRegression                0:00:49       0.9753    0.9841
           153   MaxAbsScaler LogisticRegression                0:01:04       0.9718    0.9841
           154   MaxAbsScaler LogisticRegression                0:00:52       0.9718    0.9841
           155   MinMaxScaler LogisticRegression                0:00:51       0.9718    0.9841
           156   StandardScalerWrapper LogisticRegression       0:00:48       0.9718    0.9841
           157   RobustScaler RandomForest                      0:01:01       0.9525    0.9841
           158   StandardScalerWrapper LogisticRegression       0:00:58       0.9771    0.9841
           159   StandardScalerWrapper XGBoostClassifier        0:00:54       0.9665    0.9841
           160   MinMaxScaler LogisticRegression                0:00:55       0.9718    0.9841
           161   MinMaxScaler LogisticRegression                0:00:48       0.9736    0.9841
           162   StandardScalerWrapper LogisticRegression       0:00:51       0.9824    0.9841
           163   StandardScalerWrapper LogisticRegression       0:00:54       0.9489    0.9841
           164   MinMaxScaler LogisticRegression                0:00:50       0.9789    0.9841
           165   MaxAbsScaler LogisticRegression                0:00:52       0.9718    0.9841
           166   StandardScalerWrapper LogisticRegression       0:00:55       0.9841    0.9841
           167   MaxAbsScaler LogisticRegression                0:00:56       0.9718    0.9841
           168   MaxAbsScaler LogisticRegression                0:00:53       0.9701    0.9841
           169   StandardScalerWrapper LogisticRegression       0:00:58       0.9806    0.9841
           170   MaxAbsScaler LogisticRegression                0:00:52       0.9542    0.9841
           171   StandardScalerWrapper SGD                      0:00:46       0.9806    0.9841
           172   StandardScalerWrapper LogisticRegression       0:00:58       0.9736    0.9841
           173   MaxAbsScaler LogisticRegression                0:00:52       0.9718    0.9841
           174   StandardScalerWrapper LogisticRegression       0:00:54       0.9771    0.9841
           175   StandardScalerWrapper LogisticRegression       0:01:11       0.9718    0.9841
           176   StandardScalerWrapper XGBoostClassifier        0:00:54       0.9648    0.9841
           180   MaxAbsScaler LogisticRegression                0:01:07       0.9718    0.9841
           181   MaxAbsScaler LogisticRegression                0:00:54       0.9718    0.9841
           177   MaxAbsScaler LogisticRegression                0:01:24       0.9718    0.9841
    ERROR: {
        "additional_properties": {},
        "error": {
            "additional_properties": {
                "debugInfo": null
            },
            "code": "UserError",
            "severity": null,
            "message": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_format": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_parameters": {},
            "reference_code": null,
            "details_uri": null,
            "target": null,
            "details": [],
            "inner_error": {
                "additional_properties": {},
                "code": "ResourceExhausted",
                "inner_error": {
                    "additional_properties": {},
                    "code": "Timeout",
                    "inner_error": {
                        "additional_properties": {},
                        "code": "ExperimentTimeoutForIterations",
                        "inner_error": null
                    }
                }
            }
        },
        "correlation": null,
        "environment": null,
        "location": null,
        "time": {},
        "component_name": null
    }
           178   MaxAbsScaler LogisticRegression                0:01:21       0.9718    0.9841
    ERROR: {
        "additional_properties": {},
        "error": {
            "additional_properties": {
                "debugInfo": null
            },
            "code": "UserError",
            "severity": null,
            "message": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_format": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_parameters": {},
            "reference_code": null,
            "details_uri": null,
            "target": null,
            "details": [],
            "inner_error": {
                "additional_properties": {},
                "code": "ResourceExhausted",
                "inner_error": {
                    "additional_properties": {},
                    "code": "Timeout",
                    "inner_error": {
                        "additional_properties": {},
                        "code": "ExperimentTimeoutForIterations",
                        "inner_error": null
                    }
                }
            }
        },
        "correlation": null,
        "environment": null,
        "location": null,
        "time": {},
        "component_name": null
    }
           179   SparseNormalizer LightGBM                      0:01:25       0.9684    0.9841
    ERROR: {
        "additional_properties": {},
        "error": {
            "additional_properties": {
                "debugInfo": null
            },
            "code": "UserError",
            "severity": null,
            "message": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_format": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_parameters": {},
            "reference_code": null,
            "details_uri": null,
            "target": null,
            "details": [],
            "inner_error": {
                "additional_properties": {},
                "code": "ResourceExhausted",
                "inner_error": {
                    "additional_properties": {},
                    "code": "Timeout",
                    "inner_error": {
                        "additional_properties": {},
                        "code": "ExperimentTimeoutForIterations",
                        "inner_error": null
                    }
                }
            }
        },
        "correlation": null,
        "environment": null,
        "location": null,
        "time": {},
        "component_name": null
    }
           182   MinMaxScaler LogisticRegression                0:01:09       0.9771    0.9841
    ERROR: {
        "additional_properties": {},
        "error": {
            "additional_properties": {
                "debugInfo": null
            },
            "code": "UserError",
            "severity": null,
            "message": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_format": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_parameters": {},
            "reference_code": null,
            "details_uri": null,
            "target": null,
            "details": [],
            "inner_error": {
                "additional_properties": {},
                "code": "ResourceExhausted",
                "inner_error": {
                    "additional_properties": {},
                    "code": "Timeout",
                    "inner_error": {
                        "additional_properties": {},
                        "code": "ExperimentTimeoutForIterations",
                        "inner_error": null
                    }
                }
            }
        },
        "correlation": null,
        "environment": null,
        "location": null,
        "time": {},
        "component_name": null
    }
           183   StandardScalerWrapper KNN                      0:00:59       0.9647    0.9841
    ERROR: {
        "additional_properties": {},
        "error": {
            "additional_properties": {
                "debugInfo": null
            },
            "code": "UserError",
            "severity": null,
            "message": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_format": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_parameters": {},
            "reference_code": null,
            "details_uri": null,
            "target": null,
            "details": [],
            "inner_error": {
                "additional_properties": {},
                "code": "ResourceExhausted",
                "inner_error": {
                    "additional_properties": {},
                    "code": "Timeout",
                    "inner_error": {
                        "additional_properties": {},
                        "code": "ExperimentTimeoutForIterations",
                        "inner_error": null
                    }
                }
            }
        },
        "correlation": null,
        "environment": null,
        "location": null,
        "time": {},
        "component_name": null
    }
           184   MinMaxScaler KNN                               0:00:51       0.9718    0.9841
    ERROR: {
        "additional_properties": {},
        "error": {
            "additional_properties": {
                "debugInfo": null
            },
            "code": "UserError",
            "severity": null,
            "message": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_format": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_parameters": {},
            "reference_code": null,
            "details_uri": null,
            "target": null,
            "details": [],
            "inner_error": {
                "additional_properties": {},
                "code": "ResourceExhausted",
                "inner_error": {
                    "additional_properties": {},
                    "code": "Timeout",
                    "inner_error": {
                        "additional_properties": {},
                        "code": "ExperimentTimeoutForIterations",
                        "inner_error": null
                    }
                }
            }
        },
        "correlation": null,
        "environment": null,
        "location": null,
        "time": {},
        "component_name": null
    }
           185                                                  0:00:41          nan    0.9841
    ERROR: {
        "additional_properties": {},
        "error": {
            "additional_properties": {
                "debugInfo": null
            },
            "code": "UserError",
            "severity": null,
            "message": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_format": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_parameters": {},
            "reference_code": null,
            "details_uri": null,
            "target": null,
            "details": [],
            "inner_error": {
                "additional_properties": {},
                "code": "ResourceExhausted",
                "inner_error": {
                    "additional_properties": {},
                    "code": "Timeout",
                    "inner_error": {
                        "additional_properties": {},
                        "code": "ExperimentTimeoutForIterations",
                        "inner_error": null
                    }
                }
            }
        },
        "correlation": null,
        "environment": null,
        "location": null,
        "time": {},
        "component_name": null
    }
           186   MaxAbsScaler LogisticRegression                0:00:30       0.9718    0.9841
    ERROR: {
        "additional_properties": {},
        "error": {
            "additional_properties": {
                "debugInfo": null
            },
            "code": "UserError",
            "severity": null,
            "message": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_format": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_parameters": {},
            "reference_code": null,
            "details_uri": null,
            "target": null,
            "details": [],
            "inner_error": {
                "additional_properties": {},
                "code": "ResourceExhausted",
                "inner_error": {
                    "additional_properties": {},
                    "code": "Timeout",
                    "inner_error": {
                        "additional_properties": {},
                        "code": "ExperimentTimeoutForIterations",
                        "inner_error": null
                    }
                }
            }
        },
        "correlation": null,
        "environment": null,
        "location": null,
        "time": {},
        "component_name": null
    }
           187                                                  0:00:10          nan    0.9841
    ERROR: {
        "additional_properties": {},
        "error": {
            "additional_properties": {
                "debugInfo": null
            },
            "code": "UserError",
            "severity": null,
            "message": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_format": "Experiment timeout reached, please consider increasing your experiment timeout.",
            "message_parameters": {},
            "reference_code": null,
            "details_uri": null,
            "target": null,
            "details": [],
            "inner_error": {
                "additional_properties": {},
                "code": "ResourceExhausted",
                "inner_error": {
                    "additional_properties": {},
                    "code": "Timeout",
                    "inner_error": {
                        "additional_properties": {},
                        "code": "ExperimentTimeoutForIterations",
                        "inner_error": null
                    }
                }
            }
        },
        "correlation": null,
        "environment": null,
        "location": null,
        "time": {},
        "component_name": null
    }
           189    StackEnsemble                                 0:02:59       0.9841    0.9841
           188    VotingEnsemble                                0:03:16       0.9876    0.9876





    {'runId': 'AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40',
     'target': 'hyperdrive-compu',
     'status': 'Completed',
     'startTimeUtc': '2021-02-07T07:37:04.267154Z',
     'endTimeUtc': '2021-02-07T08:09:01.40534Z',
     'properties': {'num_iterations': '1000',
      'training_type': 'TrainFull',
      'acquisition_function': 'EI',
      'primary_metric': 'accuracy',
      'train_split': '0',
      'acquisition_parameter': '0',
      'num_cross_validation': None,
      'target': 'hyperdrive-compu',
      'AMLSettingsJsonString': '{"path":null,"name":"auto-experiment","subscription_id":"81cefad3-d2c9-4f77-a466-99a7f541c7bb","resource_group":"aml-quickstarts-137761","workspace_name":"quick-starts-ws-137761","region":"southcentralus","compute_target":"hyperdrive-compu","spark_service":null,"azure_service":"remote","many_models":false,"pipeline_fetch_max_batch_size":1,"iterations":1000,"primary_metric":"accuracy","task_type":"classification","data_script":null,"validation_size":0.0,"n_cross_validations":null,"y_min":null,"y_max":null,"num_classes":null,"featurization":"auto","_ignore_package_version_incompatibilities":false,"is_timeseries":false,"max_cores_per_iteration":1,"max_concurrent_iterations":10,"iteration_timeout_minutes":null,"mem_in_mb":null,"enforce_time_on_windows":false,"experiment_timeout_minutes":25,"experiment_exit_score":null,"whitelist_models":null,"blacklist_algos":["TensorFlowLinearClassifier","TensorFlowDNN"],"supported_models":["LightGBM","GradientBoosting","AveragedPerceptronClassifier","MultinomialNaiveBayes","TensorFlowDNN","TensorFlowLinearClassifier","SGD","ExtremeRandomTrees","LogisticRegression","XGBoostClassifier","BernoulliNaiveBayes","KNN","RandomForest","LinearSVM","DecisionTree","SVM"],"auto_blacklist":true,"blacklist_samples_reached":false,"exclude_nan_labels":true,"verbosity":20,"_debug_log":"azureml_automl.log","show_warnings":false,"model_explainability":true,"service_url":null,"sdk_url":null,"sdk_packages":null,"enable_onnx_compatible_models":false,"enable_split_onnx_featurizer_estimator_models":false,"vm_type":"STANDARD_D15_V2","telemetry_verbosity":20,"send_telemetry":true,"enable_dnn":false,"scenario":"SDK-1.13.0","environment_label":null,"force_text_dnn":false,"enable_feature_sweeping":true,"enable_early_stopping":false,"early_stopping_n_iters":10,"metrics":null,"enable_ensembling":true,"enable_stack_ensembling":true,"ensemble_iterations":15,"enable_tf":false,"enable_subsampling":null,"subsample_seed":null,"enable_nimbusml":false,"enable_streaming":false,"force_streaming":false,"track_child_runs":true,"allowed_private_models":[],"label_column_name":"target","weight_column_name":null,"cv_split_column_names":null,"enable_local_managed":false,"_local_managed_run_id":null,"cost_mode":1,"lag_length":0,"metric_operation":"maximize","preprocess":true}',
      'DataPrepJsonString': '{\\"training_data\\": \\"{\\\\\\"blocks\\\\\\": [{\\\\\\"id\\\\\\": \\\\\\"99baf4dd-e47f-485a-b409-20ff01bfc546\\\\\\", \\\\\\"type\\\\\\": \\\\\\"Microsoft.DPrep.GetDatastoreFilesBlock\\\\\\", \\\\\\"arguments\\\\\\": {\\\\\\"datastores\\\\\\": [{\\\\\\"datastoreName\\\\\\": \\\\\\"workspaceblobstore\\\\\\", \\\\\\"path\\\\\\": \\\\\\"datapath/5e6d4177-8965-4e99-83a2-a9190e1be837/\\\\\\", \\\\\\"resourceGroup\\\\\\": \\\\\\"aml-quickstarts-137761\\\\\\", \\\\\\"subscription\\\\\\": \\\\\\"81cefad3-d2c9-4f77-a466-99a7f541c7bb\\\\\\", \\\\\\"workspaceName\\\\\\": \\\\\\"quick-starts-ws-137761\\\\\\"}]}, \\\\\\"localData\\\\\\": {}, \\\\\\"isEnabled\\\\\\": true, \\\\\\"name\\\\\\": null, \\\\\\"annotation\\\\\\": null}, {\\\\\\"id\\\\\\": \\\\\\"68146a44-a3b7-4aa2-874e-c94ce800a878\\\\\\", \\\\\\"type\\\\\\": \\\\\\"Microsoft.DPrep.ReadParquetFileBlock\\\\\\", \\\\\\"arguments\\\\\\": {\\\\\\"preview\\\\\\": false}, \\\\\\"localData\\\\\\": {}, \\\\\\"isEnabled\\\\\\": true, \\\\\\"name\\\\\\": null, \\\\\\"annotation\\\\\\": null}, {\\\\\\"id\\\\\\": \\\\\\"84b373e6-595b-46dd-9b96-d854d0449c83\\\\\\", \\\\\\"type\\\\\\": \\\\\\"Microsoft.DPrep.DropColumnsBlock\\\\\\", \\\\\\"arguments\\\\\\": {\\\\\\"columns\\\\\\": {\\\\\\"type\\\\\\": 0, \\\\\\"details\\\\\\": {\\\\\\"selectedColumns\\\\\\": [\\\\\\"Path\\\\\\"]}}}, \\\\\\"localData\\\\\\": {}, \\\\\\"isEnabled\\\\\\": true, \\\\\\"name\\\\\\": null, \\\\\\"annotation\\\\\\": null}], \\\\\\"inspectors\\\\\\": [], \\\\\\"meta\\\\\\": {\\\\\\"savedDatasetId\\\\\\": \\\\\\"0cf8d310-4b7c-40a5-abb4-6d77dbb32341\\\\\\", \\\\\\"datasetType\\\\\\": \\\\\\"tabular\\\\\\", \\\\\\"subscriptionId\\\\\\": \\\\\\"81cefad3-d2c9-4f77-a466-99a7f541c7bb\\\\\\", \\\\\\"workspaceId\\\\\\": \\\\\\"11ba80c0-0e32-44fa-afda-8080081712d4\\\\\\", \\\\\\"workspaceLocation\\\\\\": \\\\\\"southcentralus\\\\\\"}}\\", \\"activities\\": 0}',
      'EnableSubsampling': None,
      'runTemplate': 'AutoML',
      'azureml.runsource': 'automl',
      'display_task_type': 'classification',
      'dependencies_versions': '{"azureml-widgets": "1.20.0", "azureml-train": "1.20.0", "azureml-train-restclients-hyperdrive": "1.20.0", "azureml-train-core": "1.20.0", "azureml-train-automl": "1.20.0", "azureml-train-automl-runtime": "1.20.0", "azureml-train-automl-client": "1.20.0", "azureml-tensorboard": "1.20.0", "azureml-telemetry": "1.20.0", "azureml-sdk": "1.20.0", "azureml-samples": "0+unknown", "azureml-pipeline": "1.20.0", "azureml-pipeline-steps": "1.20.0", "azureml-pipeline-core": "1.20.0", "azureml-opendatasets": "1.20.0", "azureml-model-management-sdk": "1.0.1b6.post1", "azureml-mlflow": "1.20.0.post1", "azureml-interpret": "1.20.0", "azureml-explain-model": "1.20.0", "azureml-defaults": "1.20.0", "azureml-dataset-runtime": "1.20.0", "azureml-dataprep": "2.7.3", "azureml-dataprep-rslex": "1.5.0", "azureml-dataprep-native": "27.0.0", "azureml-datadrift": "1.20.0", "azureml-core": "1.20.0", "azureml-contrib-services": "1.20.0", "azureml-contrib-server": "1.20.0", "azureml-contrib-reinforcementlearning": "1.20.0", "azureml-contrib-pipeline-steps": "1.20.0", "azureml-contrib-notebook": "1.20.0", "azureml-contrib-interpret": "1.20.0", "azureml-contrib-gbdt": "1.20.0", "azureml-contrib-fairness": "1.20.0", "azureml-contrib-dataset": "1.20.0", "azureml-cli-common": "1.20.0", "azureml-automl-runtime": "1.20.0", "azureml-automl-core": "1.20.0", "azureml-accel-models": "1.20.0"}',
      '_aml_system_scenario_identification': 'Remote.Parent',
      'ClientType': 'SDK',
      'environment_cpu_name': 'AzureML-AutoML',
      'environment_cpu_label': 'prod',
      'environment_gpu_name': 'AzureML-AutoML-GPU',
      'environment_gpu_label': 'prod',
      'root_attribution': 'automl',
      'attribution': 'AutoML',
      'Orchestrator': 'AutoML',
      'CancelUri': 'https://southcentralus.experiments.azureml.net/jasmine/v1.0/subscriptions/81cefad3-d2c9-4f77-a466-99a7f541c7bb/resourceGroups/aml-quickstarts-137761/providers/Microsoft.MachineLearningServices/workspaces/quick-starts-ws-137761/experimentids/71f00d76-b22a-442f-8ac1-abb0115484be/cancel/AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40',
      'azureml.git.repository_uri': 'https://github.com/bhadreshpsavani/Breast-Cancer-Prediction-AzureML.git',
      'mlflow.source.git.repoURL': 'https://github.com/bhadreshpsavani/Breast-Cancer-Prediction-AzureML.git',
      'azureml.git.branch': 'main',
      'mlflow.source.git.branch': 'main',
      'azureml.git.commit': '65810cf926536ed5caecaa6b7515b0242ec21d38',
      'mlflow.source.git.commit': '65810cf926536ed5caecaa6b7515b0242ec21d38',
      'azureml.git.dirty': 'True',
      'ClientSdkVersion': '1.21.0',
      'snapshotId': '00000000-0000-0000-0000-000000000000',
      'SetupRunId': 'AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_setup',
      'SetupRunContainerId': 'dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_setup',
      'FeaturizationRunJsonPath': 'featurizer_container.json',
      'FeaturizationRunId': 'AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_featurize',
      'ProblemInfoJsonString': '{"dataset_num_categorical": 0, "is_sparse": false, "subsampling": false, "dataset_classes": 2, "dataset_features": 30, "dataset_samples": 569, "single_frequency_class_detected": false}',
      'ModelExplainRunId': 'AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_ModelExplain'},
     'inputDatasets': [{'dataset': {'id': '0cf8d310-4b7c-40a5-abb4-6d77dbb32341'}, 'consumptionDetails': {'type': 'RunInput', 'inputName': 'training_data', 'mechanism': 'Direct'}}],
     'outputDatasets': [],
     'logFiles': {},
     'submittedBy': 'ODL_User 137761'}



## Best Model

TODO: In the cell below, get the best model from the automl experiments and display all the properties of the model.




```python
# Retrieve and save your best automl model.
best_run, fitted_model = remote_run.get_output()
# get_metrics()
# Returns the metrics
print("Best run metrics :",best_run.get_metrics())
# get_details()
# Returns a dictionary with the details for the run
print("Best run details :",best_run.get_details())
```

    WARNING:root:The version of the SDK does not match the version the model was trained on.
    WARNING:root:The consistency in the result may not be guaranteed.
    WARNING:root:Package:azureml-automl-core, training version:1.21.0, current version:1.20.0
    Package:azureml-automl-runtime, training version:1.21.0, current version:1.20.0
    Package:azureml-core, training version:1.21.0.post1, current version:1.20.0
    Package:azureml-dataprep, training version:2.8.2, current version:2.7.3
    Package:azureml-dataprep-native, training version:28.0.0, current version:27.0.0
    Package:azureml-dataprep-rslex, training version:1.6.0, current version:1.5.0
    Package:azureml-dataset-runtime, training version:1.21.0, current version:1.20.0
    Package:azureml-defaults, training version:1.21.0, current version:1.20.0
    Package:azureml-interpret, training version:1.21.0, current version:1.20.0
    Package:azureml-pipeline-core, training version:1.21.0, current version:1.20.0
    Package:azureml-telemetry, training version:1.21.0, current version:1.20.0
    Package:azureml-train-automl-client, training version:1.21.0, current version:1.20.0
    Package:azureml-train-automl-runtime, training version:1.21.0, current version:1.20.0
    WARNING:root:Please ensure the version of your local conda dependencies match the version on which your model was trained in order to properly retrieve your model.


    Best run metrics : {'balanced_accuracy': 0.9854676440849343, 'f1_score_micro': 0.987625313283208, 'precision_score_micro': 0.987625313283208, 'AUC_macro': 0.9963579512143124, 'precision_score_weighted': 0.9883351195972386, 'AUC_weighted': 0.9963579512143124, 'recall_score_weighted': 0.987625313283208, 'average_precision_score_weighted': 0.9967590444840608, 'precision_score_macro': 0.988327067669173, 'average_precision_score_micro': 0.9964311117596619, 'weighted_accuracy': 0.9897722338333474, 'recall_score_micro': 0.987625313283208, 'f1_score_macro': 0.9864051261050873, 'log_loss': 0.06840767880916371, 'recall_score_macro': 0.9854676440849343, 'f1_score_weighted': 0.9875504672580133, 'norm_macro_recall': 0.9709352881698686, 'matthews_correlation': 0.9737259340055987, 'accuracy': 0.987625313283208, 'AUC_micro': 0.9963578569701195, 'average_precision_score_macro': 0.9965934318774204, 'accuracy_table': 'aml://artifactId/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/accuracy_table', 'confusion_matrix': 'aml://artifactId/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/confusion_matrix'}
    Best run details : {'runId': 'AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188', 'target': 'hyperdrive-compu', 'status': 'Completed', 'startTimeUtc': '2021-02-07T08:05:14.774296Z', 'endTimeUtc': '2021-02-07T08:08:30.482382Z', 'properties': {'runTemplate': 'automl_child', 'pipeline_id': '__AutoML_Ensemble__', 'pipeline_spec': '{"pipeline_id":"__AutoML_Ensemble__","objects":[{"module":"azureml.train.automl.ensemble","class_name":"Ensemble","spec_class":"sklearn","param_args":[],"param_kwargs":{"automl_settings":"{\'task_type\':\'classification\',\'primary_metric\':\'accuracy\',\'verbosity\':20,\'ensemble_iterations\':15,\'is_timeseries\':False,\'name\':\'auto-experiment\',\'compute_target\':\'hyperdrive-compu\',\'subscription_id\':\'81cefad3-d2c9-4f77-a466-99a7f541c7bb\',\'region\':\'southcentralus\',\'spark_service\':None}","ensemble_run_id":"AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188","experiment_name":"auto-experiment","workspace_name":"quick-starts-ws-137761","subscription_id":"81cefad3-d2c9-4f77-a466-99a7f541c7bb","resource_group_name":"aml-quickstarts-137761"}}]}', 'training_percent': '100', 'predicted_cost': None, 'iteration': '188', '_aml_system_scenario_identification': 'Remote.Child', '_azureml.ComputeTargetType': 'amlcompute', 'ContentSnapshotId': '0503d0e1-7b66-4016-b0c6-3f22fc60b3f1', 'ProcessInfoFile': 'azureml-logs/process_info.json', 'ProcessStatusFile': 'azureml-logs/process_status.json', 'run_template': 'automl_child', 'run_preprocessor': '', 'run_algorithm': 'VotingEnsemble', 'conda_env_data_location': 'aml://artifact/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/outputs/conda_env_v_1_0_0.yml', 'model_data_location': 'aml://artifact/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/outputs/model.pkl', 'model_size_on_disk': '96110', 'scoring_data_location': 'aml://artifact/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/outputs/scoring_file_v_1_0_0.py', 'model_exp_support': 'True', 'pipeline_graph_version': '1.0.0', 'model_name': 'AutoML180ef0089188', 'staticProperties': '{}', 'score': '0.987625313283208', 'run_properties': "classification_labels=None,\n                              estimators=[('166',\n                                           Pipeline(memory=None,\n                                                    steps=[('standardscalerwrapper',\n                                                            <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7fb4ec79eda0>", 'pipeline_script': '{"pipeline_id":"__AutoML_Ensemble__","objects":[{"module":"azureml.train.automl.ensemble","class_name":"Ensemble","spec_class":"sklearn","param_args":[],"param_kwargs":{"automl_settings":"{\'task_type\':\'classification\',\'primary_metric\':\'accuracy\',\'verbosity\':20,\'ensemble_iterations\':15,\'is_timeseries\':False,\'name\':\'auto-experiment\',\'compute_target\':\'hyperdrive-compu\',\'subscription_id\':\'81cefad3-d2c9-4f77-a466-99a7f541c7bb\',\'region\':\'southcentralus\',\'spark_service\':None}","ensemble_run_id":"AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188","experiment_name":"auto-experiment","workspace_name":"quick-starts-ws-137761","subscription_id":"81cefad3-d2c9-4f77-a466-99a7f541c7bb","resource_group_name":"aml-quickstarts-137761"}}]}', 'training_type': 'MeanCrossValidation', 'num_classes': '', 'framework': 'sklearn', 'fit_time': '104', 'goal': 'accuracy_max', 'class_labels': '', 'primary_metric': 'accuracy', 'errors': '{}', 'onnx_model_resource': '{}', 'error_code': '', 'failure_reason': '', 'feature_skus': 'automatedml_sdk_guardrails', 'dependencies_versions': '{"azureml-train-automl-runtime": "1.21.0", "azureml-train-automl-client": "1.21.0", "azureml-telemetry": "1.21.0", "azureml-pipeline-core": "1.21.0", "azureml-model-management-sdk": "1.0.1b6.post1", "azureml-interpret": "1.21.0", "azureml-defaults": "1.21.0", "azureml-dataset-runtime": "1.21.0", "azureml-dataprep": "2.8.2", "azureml-dataprep-rslex": "1.6.0", "azureml-dataprep-native": "28.0.0", "azureml-core": "1.21.0.post1", "azureml-automl-runtime": "1.21.0", "azureml-automl-core": "1.21.0"}', 'num_cores': '20', 'num_logical_cores': '20', 'peak_memory_usage': '567252', 'vm_configuration': 'Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz', 'core_hours': '0.025508733333333332'}, 'inputDatasets': [], 'outputDatasets': [], 'runDefinition': {'script': 'automl_driver.py', 'command': '', 'useAbsolutePath': False, 'arguments': [], 'sourceDirectoryDataStore': None, 'framework': 'Python', 'communicator': 'None', 'target': 'hyperdrive-compu', 'dataReferences': {}, 'data': {}, 'outputData': {}, 'jobName': None, 'maxRunDurationSeconds': None, 'nodeCount': 1, 'priority': None, 'credentialPassthrough': False, 'environment': {'name': 'AutoML-AzureML-AutoML', 'version': 'Autosave_2021-02-07T06:34:02Z_2a314762', 'python': {'interpreterPath': 'python', 'userManagedDependencies': False, 'condaDependencies': {'channels': ['anaconda', 'conda-forge', 'pytorch'], 'dependencies': ['python=3.6.2', 'pip=20.2.4', {'pip': ['azureml-core==1.21.0.post1', 'azureml-pipeline-core==1.21.0', 'azureml-telemetry==1.21.0', 'azureml-defaults==1.21.0', 'azureml-interpret==1.21.0', 'azureml-automl-core==1.21.0', 'azureml-automl-runtime==1.21.0', 'azureml-train-automl-client==1.21.0', 'azureml-train-automl-runtime==1.21.0', 'azureml-dataset-runtime==1.21.0', 'inference-schema', 'py-cpuinfo==5.0.0', 'boto3==1.15.18', 'botocore==1.18.18']}, 'numpy~=1.18.0', 'scikit-learn==0.22.1', 'pandas~=0.25.0', 'py-xgboost<=0.90', 'fbprophet==0.5', 'holidays==0.9.11', 'setuptools-git', 'psutil>5.0.0,<6.0.0'], 'name': 'azureml_20a8278aa8b20dd48cc50f56a6d2586c'}, 'baseCondaEnvironment': None}, 'environmentVariables': {'EXAMPLE_ENV_VAR': 'EXAMPLE_VALUE'}, 'docker': {'baseImage': 'mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20210104.v1', 'platform': {'os': 'Linux', 'architecture': 'amd64'}, 'baseDockerfile': None, 'baseImageRegistry': {'address': None, 'username': None, 'password': None}, 'enabled': True, 'arguments': []}, 'spark': {'repositories': [], 'packages': [], 'precachePackages': True}, 'inferencingStackVersion': None}, 'history': {'outputCollection': True, 'directoriesToWatch': ['logs'], 'enableMLflowTracking': True}, 'spark': {'configuration': {'spark.app.name': 'Azure ML Experiment', 'spark.yarn.maxAppAttempts': '1'}}, 'parallelTask': {'maxRetriesPerWorker': 0, 'workerCountPerNode': 1, 'terminalExitCodes': None, 'configuration': {}}, 'amlCompute': {'name': None, 'vmSize': None, 'retainCluster': False, 'clusterMaxNodeCount': None}, 'aiSuperComputer': {'instanceType': None, 'frameworkImage': None, 'imageVersion': None, 'location': None, 'aiSuperComputerStorageData': None, 'interactive': False, 'scalePolicy': None}, 'tensorflow': {'workerCount': 1, 'parameterServerCount': 1}, 'mpi': {'processCountPerNode': 1}, 'pyTorch': {'communicationBackend': None, 'processCount': None}, 'hdi': {'yarnDeployMode': 'Cluster'}, 'containerInstance': {'region': None, 'cpuCores': 2.0, 'memoryGb': 3.5}, 'exposedPorts': None, 'docker': {'useDocker': True, 'sharedVolumes': True, 'shmSize': '2g', 'arguments': []}, 'cmk8sCompute': {'configuration': {}}, 'commandReturnCodeConfig': {'returnCode': 'Zero', 'successfulReturnCodes': []}}, 'logFiles': {'azureml-logs/55_azureml-execution-tvmps_43ed9e00393b7a9ac95897cdc3dfae4e60b9ca7641a82fc0928ee2e89f041aa4_d.txt': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/azureml-logs/55_azureml-execution-tvmps_43ed9e00393b7a9ac95897cdc3dfae4e60b9ca7641a82fc0928ee2e89f041aa4_d.txt?sv=2019-02-02&sr=b&sig=HGnDCoLg4xbt9rq6b%2B9CZzXXaSUnw44XnwwIQK6xa%2B0%3D&st=2021-02-07T07%3A59%3A31Z&se=2021-02-07T16%3A09%3A31Z&sp=r', 'azureml-logs/65_job_prep-tvmps_43ed9e00393b7a9ac95897cdc3dfae4e60b9ca7641a82fc0928ee2e89f041aa4_d.txt': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/azureml-logs/65_job_prep-tvmps_43ed9e00393b7a9ac95897cdc3dfae4e60b9ca7641a82fc0928ee2e89f041aa4_d.txt?sv=2019-02-02&sr=b&sig=4fGU6Jw6SDmtWEv6I0%2FVZ%2Bmd6lmA4ijauwoey7lQ5v0%3D&st=2021-02-07T07%3A59%3A31Z&se=2021-02-07T16%3A09%3A31Z&sp=r', 'azureml-logs/70_driver_log.txt': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/azureml-logs/70_driver_log.txt?sv=2019-02-02&sr=b&sig=vYPeIoQC11Ky83V6YS%2BCYULMBORbWPpUd5vonfF%2FiNc%3D&st=2021-02-07T07%3A59%3A31Z&se=2021-02-07T16%3A09%3A31Z&sp=r', 'azureml-logs/75_job_post-tvmps_43ed9e00393b7a9ac95897cdc3dfae4e60b9ca7641a82fc0928ee2e89f041aa4_d.txt': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/azureml-logs/75_job_post-tvmps_43ed9e00393b7a9ac95897cdc3dfae4e60b9ca7641a82fc0928ee2e89f041aa4_d.txt?sv=2019-02-02&sr=b&sig=pu02Lv15N7shoeE1oSlL36Z1jTfXxsSKi74kYgztGMA%3D&st=2021-02-07T07%3A59%3A31Z&se=2021-02-07T16%3A09%3A31Z&sp=r', 'azureml-logs/process_info.json': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/azureml-logs/process_info.json?sv=2019-02-02&sr=b&sig=bJE7%2BaIxwWqiOfi440pjFUgcOv2VX8qfwe7gAJK%2Bj%2BQ%3D&st=2021-02-07T07%3A59%3A31Z&se=2021-02-07T16%3A09%3A31Z&sp=r', 'azureml-logs/process_status.json': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/azureml-logs/process_status.json?sv=2019-02-02&sr=b&sig=fcyPc5A7U%2BoOsucxsIIxUt%2BedhycEUCUBZgRrtAk4Og%3D&st=2021-02-07T07%3A59%3A31Z&se=2021-02-07T16%3A09%3A31Z&sp=r', 'logs/azureml/109_azureml.log': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/logs/azureml/109_azureml.log?sv=2019-02-02&sr=b&sig=9eBMsetLD8Xay9Q18qKCgcJR%2FYt5sBQA7yvdiQT0mM4%3D&st=2021-02-07T07%3A59%3A30Z&se=2021-02-07T16%3A09%3A30Z&sp=r', 'logs/azureml/azureml_automl.log': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/logs/azureml/azureml_automl.log?sv=2019-02-02&sr=b&sig=vL57rGthm%2BajaLeogNw5z2%2FoumjVxKdCDc9QhKszPQA%3D&st=2021-02-07T07%3A59%3A30Z&se=2021-02-07T16%3A09%3A30Z&sp=r', 'logs/azureml/job_prep_azureml.log': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/logs/azureml/job_prep_azureml.log?sv=2019-02-02&sr=b&sig=vJw7V2AFJUPvhj4Hl16wMZwA0yI%2BEvQ9FKhAFCOpz2w%3D&st=2021-02-07T07%3A59%3A30Z&se=2021-02-07T16%3A09%3A30Z&sp=r', 'logs/azureml/job_release_azureml.log': 'https://mlstrg137761.blob.core.windows.net/azureml/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/logs/azureml/job_release_azureml.log?sv=2019-02-02&sr=b&sig=6ymTk4gAAn5fmQ09hiDhohDnJu12S2zXOm%2F%2FHQzRKLU%3D&st=2021-02-07T07%3A59%3A30Z&se=2021-02-07T16%3A09%3A30Z&sp=r'}, 'submittedBy': 'ODL_User 137761'}



```python
fitted_model._final_estimator
```




    PreFittedSoftVotingClassifier(classification_labels=None,
                                  estimators=[('166',
                                               Pipeline(memory=None,
                                                        steps=[('standardscalerwrapper',
                                                                <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7fb08c774d68>),
                                                               ('logisticregression',
                                                                LogisticRegression(C=6866.488450042998,
                                                                                   class_weight=None,
                                                                                   dual=False,
                                                                                   fit_intercept=True,
                                                                                   intercept_scaling=1,...
                                                                           degree=3,
                                                                           gamma='scale',
                                                                           kernel='rbf',
                                                                           max_iter=-1,
                                                                           probability=True,
                                                                           random_state=None,
                                                                           shrinking=True,
                                                                           tol=0.001,
                                                                           verbose=False))],
                                                        verbose=False))],
                                  flatten_transform=None,
                                  weights=[0.18181818181818182, 0.09090909090909091,
                                           0.09090909090909091, 0.09090909090909091,
                                           0.09090909090909091, 0.18181818181818182,
                                           0.09090909090909091, 0.09090909090909091,
                                           0.09090909090909091])




```python
print(fitted_model)
```

    Pipeline(memory=None,
             steps=[('datatransformer',
                     DataTransformer(enable_dnn=None, enable_feature_sweeping=None,
                                     feature_sweeping_config=None,
                                     feature_sweeping_timeout=None,
                                     featurization_config=None, force_text_dnn=None,
                                     is_cross_validation=None,
                                     is_onnx_compatible=None, logger=None,
                                     observer=None, task=None, working_dir=None)),
                    ('prefittedsoftvotingclassifier',...
                                                                                            gamma='scale',
                                                                                            kernel='rbf',
                                                                                            max_iter=-1,
                                                                                            probability=True,
                                                                                            random_state=None,
                                                                                            shrinking=True,
                                                                                            tol=0.001,
                                                                                            verbose=False))],
                                                                         verbose=False))],
                                                   flatten_transform=None,
                                                   weights=[0.18181818181818182,
                                                            0.09090909090909091,
                                                            0.09090909090909091,
                                                            0.09090909090909091,
                                                            0.09090909090909091,
                                                            0.18181818181818182,
                                                            0.09090909090909091,
                                                            0.09090909090909091,
                                                            0.09090909090909091]))],
             verbose=False)



```python
# Get all metrics of the best run
best_run_metrics = best_run.get_metrics()
for metric_name in best_run_metrics:
    metric = best_run_metrics[metric_name]
    print(metric_name, metric)
```

    balanced_accuracy 0.9854676440849343
    f1_score_micro 0.987625313283208
    precision_score_micro 0.987625313283208
    AUC_macro 0.9963579512143124
    precision_score_weighted 0.9883351195972386
    AUC_weighted 0.9963579512143124
    recall_score_weighted 0.987625313283208
    average_precision_score_weighted 0.9967590444840608
    precision_score_macro 0.988327067669173
    average_precision_score_micro 0.9964311117596619
    weighted_accuracy 0.9897722338333474
    recall_score_micro 0.987625313283208
    f1_score_macro 0.9864051261050873
    log_loss 0.06840767880916371
    recall_score_macro 0.9854676440849343
    f1_score_weighted 0.9875504672580133
    norm_macro_recall 0.9709352881698686
    matthews_correlation 0.9737259340055987
    accuracy 0.987625313283208
    AUC_micro 0.9963578569701195
    average_precision_score_macro 0.9965934318774204
    accuracy_table aml://artifactId/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/accuracy_table
    confusion_matrix aml://artifactId/ExperimentRun/dcid.AutoML_180ef008-9bb3-46e4-8e6e-64a66d835c40_188/confusion_matrix



```python
# Print detailed parameters of the fitted model
from pprint import pprint
def print_model(model, prefix=""):
    for step in model.steps:
        print(prefix + step[0])
        if hasattr(step[1], 'estimators') and hasattr(step[1], 'weights'):
            pprint({'estimators': list(
                e[0] for e in step[1].estimators), 'weights': step[1].weights})
            print()
            for estimator in step[1].estimators:
                print_model(estimator[1], estimator[0] + ' - ')
        else:
            pprint(step[1].get_params())
            print()

print_model(fitted_model)
```

    datatransformer
    {'enable_dnn': None,
     'enable_feature_sweeping': None,
     'feature_sweeping_config': None,
     'feature_sweeping_timeout': None,
     'featurization_config': None,
     'force_text_dnn': None,
     'is_cross_validation': None,
     'is_onnx_compatible': None,
     'logger': None,
     'observer': None,
     'task': None,
     'working_dir': None}
    
    prefittedsoftvotingclassifier
    {'estimators': ['166', '119', '113', '111', '112', '13', '88', '83', '116'],
     'weights': [0.18181818181818182,
                 0.09090909090909091,
                 0.09090909090909091,
                 0.09090909090909091,
                 0.09090909090909091,
                 0.18181818181818182,
                 0.09090909090909091,
                 0.09090909090909091,
                 0.09090909090909091]}
    
    166 - standardscalerwrapper
    {'class_name': 'StandardScaler',
     'copy': True,
     'module_name': 'sklearn.preprocessing._data',
     'with_mean': True,
     'with_std': True}
    
    166 - logisticregression
    {'C': 6866.488450042998,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'l1_ratio': None,
     'max_iter': 100,
     'multi_class': 'ovr',
     'n_jobs': 1,
     'penalty': 'l2',
     'random_state': None,
     'solver': 'saga',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}
    
    119 - standardscalerwrapper
    {'class_name': 'StandardScaler',
     'copy': True,
     'module_name': 'sklearn.preprocessing._data',
     'with_mean': True,
     'with_std': True}
    
    119 - logisticregression
    {'C': 232.99518105153672,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'l1_ratio': None,
     'max_iter': 100,
     'multi_class': 'ovr',
     'n_jobs': 1,
     'penalty': 'l2',
     'random_state': None,
     'solver': 'saga',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}
    
    113 - standardscalerwrapper
    {'class_name': 'StandardScaler',
     'copy': True,
     'module_name': 'sklearn.preprocessing._data',
     'with_mean': True,
     'with_std': True}
    
    113 - logisticregression
    {'C': 1048.1131341546852,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'l1_ratio': None,
     'max_iter': 100,
     'multi_class': 'ovr',
     'n_jobs': 1,
     'penalty': 'l2',
     'random_state': None,
     'solver': 'saga',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}
    
    111 - standardscalerwrapper
    {'class_name': 'StandardScaler',
     'copy': True,
     'module_name': 'sklearn.preprocessing._data',
     'with_mean': True,
     'with_std': True}
    
    111 - logisticregression
    {'C': 2222.996482526191,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'l1_ratio': None,
     'max_iter': 100,
     'multi_class': 'ovr',
     'n_jobs': 1,
     'penalty': 'l2',
     'random_state': None,
     'solver': 'saga',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}
    
    112 - robustscaler
    {'copy': True,
     'quantile_range': [10, 90],
     'with_centering': True,
     'with_scaling': True}
    
    112 - logisticregression
    {'C': 16.768329368110066,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'l1_ratio': None,
     'max_iter': 100,
     'multi_class': 'ovr',
     'n_jobs': 1,
     'penalty': 'l1',
     'random_state': None,
     'solver': 'saga',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}
    
    13 - sparsenormalizer
    {'copy': True, 'norm': 'max'}
    
    13 - xgboostclassifier
    {'base_score': 0.5,
     'booster': 'gbtree',
     'colsample_bylevel': 1,
     'colsample_bynode': 1,
     'colsample_bytree': 0.5,
     'eta': 0.5,
     'gamma': 0.01,
     'learning_rate': 0.1,
     'max_delta_step': 0,
     'max_depth': 10,
     'max_leaves': 7,
     'min_child_weight': 1,
     'missing': nan,
     'n_estimators': 50,
     'n_jobs': 1,
     'nthread': None,
     'objective': 'reg:logistic',
     'random_state': 0,
     'reg_alpha': 0,
     'reg_lambda': 0.8333333333333334,
     'scale_pos_weight': 1,
     'seed': None,
     'silent': None,
     'subsample': 1,
     'tree_method': 'auto',
     'verbose': -10,
     'verbosity': 0}
    
    88 - standardscalerwrapper
    {'class_name': 'StandardScaler',
     'copy': True,
     'module_name': 'sklearn.preprocessing._data',
     'with_mean': True,
     'with_std': True}
    
    88 - logisticregression
    {'C': 0.3906939937054613,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'l1_ratio': None,
     'max_iter': 100,
     'multi_class': 'ovr',
     'n_jobs': 1,
     'penalty': 'l2',
     'random_state': None,
     'solver': 'saga',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}
    
    83 - standardscalerwrapper
    {'class_name': 'StandardScaler',
     'copy': True,
     'module_name': 'sklearn.preprocessing._data',
     'with_mean': True,
     'with_std': True}
    
    83 - logisticregression
    {'C': 0.12648552168552957,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'l1_ratio': None,
     'max_iter': 100,
     'multi_class': 'multinomial',
     'n_jobs': 1,
     'penalty': 'l2',
     'random_state': None,
     'solver': 'saga',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}
    
    116 - maxabsscaler
    {'copy': True}
    
    116 - svcwrapper
    {'C': 35.564803062231285,
     'break_ties': False,
     'cache_size': 200,
     'class_weight': 'balanced',
     'coef0': 0.0,
     'decision_function_shape': 'ovr',
     'degree': 3,
     'gamma': 'scale',
     'kernel': 'rbf',
     'max_iter': -1,
     'probability': True,
     'random_state': None,
     'shrinking': True,
     'tol': 0.001,
     'verbose': False}
    


## Model Deployment

Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.

In the cell below, register the model, create an inference config and deploy the model as a web service.


```python
#Save the model
model = best_run.register_model(model_path='outputs/model.pkl', model_name='automl_breast_cancer_predictor',
                                tags={'Training context':'Auto ML'},
                                properties={'Accuracy': best_run_metrics['accuracy']})

print(model)
```

    Model(workspace=Workspace.create(name='quick-starts-ws-137761', subscription_id='81cefad3-d2c9-4f77-a466-99a7f541c7bb', resource_group='aml-quickstarts-137761'), name=automl_breast_cancer_predictor, id=automl_breast_cancer_predictor:1, version=1, tags={'Training context': 'Auto ML'}, properties={'Accuracy': '0.987625313283208'})



```python
# Download scoring file 
best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'score.py')

# Download environment file
best_run.download_file('outputs/conda_env_v_1_0_0.yml', 'envFile.yml')
```


```python
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(entry_script='score.py',
                                    environment=best_run.get_environment())
```


```python
from azureml.core.webservice import AciWebservice
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
```


```python
from azureml.core import Model
service_name = 'breast-cancer-endpoint'
service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)
service.wait_for_deployment(show_output=True)
print(service.state)
print(service.scoring_uri)
print(service.swagger_uri)
```

    Tips: You can try get_logs(): https://aka.ms/debugimage#dockerlog or local deployment: https://aka.ms/debugimage#debug-locally to debug if deployment takes longer than 10 minutes.
    Running............................................
    Succeeded
    ACI service creation operation finished, operation "Succeeded"
    Healthy
    http://2099fe8f-747d-425d-b77b-751b8871d5a9.southcentralus.azurecontainer.io/score
    http://2099fe8f-747d-425d-b77b-751b8871d5a9.southcentralus.azurecontainer.io/swagger.json



```python
service
```




    AciWebservice(workspace=Workspace.create(name='quick-starts-ws-137761', subscription_id='81cefad3-d2c9-4f77-a466-99a7f541c7bb', resource_group='aml-quickstarts-137761'), name=breast-cancer-endpoint, image_id=None, compute_type=None, state=ACI, scoring_uri=Healthy, tags=http://2099fe8f-747d-425d-b77b-751b8871d5a9.southcentralus.azurecontainer.io/score, properties={}, created_by={'azureml.git.repository_uri': 'https://github.com/bhadreshpsavani/Breast-Cancer-Prediction-AzureML.git', 'mlflow.source.git.repoURL': 'https://github.com/bhadreshpsavani/Breast-Cancer-Prediction-AzureML.git', 'azureml.git.branch': 'main', 'mlflow.source.git.branch': 'main', 'azureml.git.commit': '65810cf926536ed5caecaa6b7515b0242ec21d38', 'mlflow.source.git.commit': '65810cf926536ed5caecaa6b7515b0242ec21d38', 'azureml.git.dirty': 'True', 'hasInferenceSchema': 'True', 'hasHttps': 'False'})



In the cell below, send a request to the web service you deployed to test it.


```python
import json

test_df = df.sample(2)
label_df = test_df.pop('target')
test_sample = json.dumps({'data': test_df.to_dict(orient='records')})
print(test_sample)
```

    {"data": [{"mean radius": 14.74, "mean texture": 25.42, "mean perimeter": 94.7, "mean area": 668.6, "mean smoothness": 0.08275, "mean compactness": 0.07214, "mean concavity": 0.04105, "mean concave points": 0.03027, "mean symmetry": 0.184, "mean fractal dimension": 0.0568, "radius error": 0.3031, "texture error": 1.385, "perimeter error": 2.177, "area error": 27.41, "smoothness error": 0.004775, "compactness error": 0.01172, "concavity error": 0.01947, "concave points error": 0.01269, "symmetry error": 0.0187, "fractal dimension error": 0.002626, "worst radius": 16.51, "worst texture": 32.29, "worst perimeter": 107.4, "worst area": 826.4, "worst smoothness": 0.106, "worst compactness": 0.1376, "worst concavity": 0.1611, "worst concave points": 0.1095, "worst symmetry": 0.2722, "worst fractal dimension": 0.06956}, {"mean radius": 11.29, "mean texture": 13.04, "mean perimeter": 72.23, "mean area": 388.0, "mean smoothness": 0.09834, "mean compactness": 0.07608, "mean concavity": 0.03265, "mean concave points": 0.02755, "mean symmetry": 0.1769, "mean fractal dimension": 0.0627, "radius error": 0.1904, "texture error": 0.5293, "perimeter error": 1.164, "area error": 13.17, "smoothness error": 0.006472, "compactness error": 0.01122, "concavity error": 0.01282, "concave points error": 0.008849, "symmetry error": 0.01692, "fractal dimension error": 0.002817, "worst radius": 12.32, "worst texture": 16.18, "worst perimeter": 78.27, "worst area": 457.5, "worst smoothness": 0.1358, "worst compactness": 0.1507, "worst concavity": 0.1275, "worst concave points": 0.0875, "worst symmetry": 0.2733, "worst fractal dimension": 0.08022}]}



```python
%%time
import requests 

# Set the content type
headers = {'Content-type': 'application/json'}

response = requests.post(service.scoring_uri, test_sample, headers=headers)
print("response")
print(response.text)
```

    response
    "{\"result\": [1, 1]}"
    CPU times: user 1.39 ms, sys: 4.13 ms, total: 5.53 ms
    Wall time: 125 ms


In the cell below, print the logs of the web service and delete the service


```python
print(service.get_logs())
```

    2021-02-07T08:20:26,084274449+00:00 - gunicorn/run 
    /usr/sbin/nginx: /azureml-envs/azureml_20a8278aa8b20dd48cc50f56a6d2586c/lib/libcrypto.so.1.0.0: no version information available (required by /usr/sbin/nginx)
    /usr/sbin/nginx: /azureml-envs/azureml_20a8278aa8b20dd48cc50f56a6d2586c/lib/libcrypto.so.1.0.0: no version information available (required by /usr/sbin/nginx)
    /usr/sbin/nginx: /azureml-envs/azureml_20a8278aa8b20dd48cc50f56a6d2586c/lib/libssl.so.1.0.0: no version information available (required by /usr/sbin/nginx)
    /usr/sbin/nginx: /azureml-envs/azureml_20a8278aa8b20dd48cc50f56a6d2586c/lib/libssl.so.1.0.0: no version information available (required by /usr/sbin/nginx)
    /usr/sbin/nginx: /azureml-envs/azureml_20a8278aa8b20dd48cc50f56a6d2586c/lib/libssl.so.1.0.0: no version information available (required by /usr/sbin/nginx)
    2021-02-07T08:20:26,085830974+00:00 - rsyslog/run 
    2021-02-07T08:20:26,086069478+00:00 - iot-server/run 
    2021-02-07T08:20:26,091234863+00:00 - nginx/run 
    rsyslogd: /azureml-envs/azureml_20a8278aa8b20dd48cc50f56a6d2586c/lib/libuuid.so.1: no version information available (required by rsyslogd)
    EdgeHubConnectionString and IOTEDGE_IOTHUBHOSTNAME are not set. Exiting...
    2021-02-07T08:20:26,290812648+00:00 - iot-server/finish 1 0
    2021-02-07T08:20:26,292153370+00:00 - Exit code 1 is normal. Not restarting iot-server.
    Starting gunicorn 19.9.0
    Listening at: http://127.0.0.1:31311 (12)
    Using worker: sync
    worker timeout is set to 300
    Booting worker with pid: 42
    SPARK_HOME not set. Skipping PySpark Initialization.
    Generating new fontManager, this may take some time...
    Initializing logger
    2021-02-07 08:20:27,550 | root | INFO | Starting up app insights client
    2021-02-07 08:20:27,550 | root | INFO | Starting up request id generator
    2021-02-07 08:20:27,550 | root | INFO | Starting up app insight hooks
    2021-02-07 08:20:27,550 | root | INFO | Invoking user's init function
    2021-02-07 08:20:30,153 | root | INFO | Users's init has completed successfully
    2021-02-07 08:20:30,156 | root | INFO | Skipping middleware: dbg_model_info as it's not enabled.
    2021-02-07 08:20:30,156 | root | INFO | Skipping middleware: dbg_resource_usage as it's not enabled.
    2021-02-07 08:20:30,157 | root | INFO | Scoring timeout is found from os.environ: 60000 ms
    2021-02-07 08:20:36,074 | root | INFO | 200
    127.0.0.1 - - [07/Feb/2021:08:20:36 +0000] "GET /swagger.json HTTP/1.0" 200 4480 "-" "Go-http-client/1.1"
    2021-02-07 08:20:39,947 | root | INFO | 200
    127.0.0.1 - - [07/Feb/2021:08:20:39 +0000] "GET /swagger.json HTTP/1.0" 200 4480 "-" "Go-http-client/1.1"
    2021-02-07 08:21:31,073 | root | INFO | Validation Request Content-Type
    2021-02-07 08:21:31,074 | root | INFO | Scoring Timer is set to 60.0 seconds
    2021-02-07 08:21:31,169 | root | INFO | 200
    127.0.0.1 - - [07/Feb/2021:08:21:31 +0000] "POST /score HTTP/1.0" 200 22 "-" "python-requests/2.25.1"
    



```python
service.delete()
```


```python
model.delete()
```


```python

```

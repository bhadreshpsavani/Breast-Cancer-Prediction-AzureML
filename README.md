# Beast Cancer Prediction with Azure ML

We will do Binary Classification on breast cancer dataset. Our Goal is to compare Hyperdrive and AutoML trained modela and deploy the best performing model.

First we will use HyperDrive to do Hyperparameter Tuning and get best performing model. We will use Automl and get Best performing model. We will compare best model from both the approach and deploy it. We will do inference on deployed model.

## Project Set Up and Installation

Step1. Open Azure Machine Learning Studio By login into https://portal.azure.com/

Step2. Create a Cluster Instance with Default parameters. Detailed steps are provide in this [link](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance)

Step3. Open Terminal from created Cluster Instance.

Step4. Clone this repository by using below command in terminal.
```
git clone https://github.com/bhadreshpsavani/Breast-Cancer-Prediction-AzureML.git
```

## Dataset

### Overview
The breast cancer dataset is a classic and very easy **binary classification** dataset.

| Attribute | Value |
| --- | --- |
| Classes | 2 |
| Samples per class | 212(M),357(B) |
| Samples total | 569 |
| Dimensionality | 30 |
| Features | real, positive |

We used sklearn version of dataset

The target is distributed like this,
| Target/Label | Percentage |
| --- | --- |
| 1 | 62.7417 |
| 0 | 37.2583|

### Task
Our task is to do Binary Classification on Breast Cancer Dataset, we will use all 30 features for training and testing.

### Access
Since we are using dataset from sklearn we can directly access dataset in Azure Notebook by importing sklearn datasets module,

We can do it like this, 
```
from sklearn import datasets
data = datasets.load_breast_cancer(as_frame=True)

```
We should need to register dataset in Azure Studio while using it in AutoML we can do it like this,
```
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.data.datapath import DataPath

# Create TabularDataset using TabularDatasetFactory
def_blob_store = ws.get_default_datastore()
print("Default datastore's name: {}".format(def_blob_store.name))
data_path = DataPath(datastore=def_blob_store, path_on_datastore='datapath')
ds = TabularDatasetFactory.register_pandas_dataframe(df, name='UCI_ML_Breast_Cancer', target=data_path)
```

## Automated ML
We choose  `experiment_timeout_minutes=50mins` to give enough time to try all experiments,  `max_concurrent_iterations=5` according to max_nodes given while creating `training_instance`, We choose Accuracy as Primary Metrics since data is not highly imbalanced it seems good fit for binary classification problem.

### Results
We got two models having highest same `98.94%` score:
1. Voting Ensemble
2. Stack Ensemble

It can be seen in below image
![AutoML_Run](Resources/Images/AutoML_Run.PNG)
![Best AutoML Model](/Resources/Images/AutoML_Best_Model.PNG)

Both the best performing models are made up of multiple models i have shown models available in our best performing `stackensembleclassifier` in the [AutoML Notebook](automl.ipynb). 

## Hyperparameter Tuning
If we look at AutoML Run LightGBM and SGD which Gradient Boosted Models seems to be performing well. But they are Not available in SKlearn Library. SKlearn provides `GradientBoostingClassifier` which is based on similiar approach of Gradient Boosting. For simplicity i used sklearn based `GradientBoostingClassifier`.

The code for finetuning the model is available in `train.py` file. We simply use this code in every run and pass different set of Hyperparameter in the Python file.

Amoung various parameters of `GradientBoostingClassifier` below three parameters seems to be effective performance based on sklearn documentation
* **n_estimators** : The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance. The selected range is `(1,10,20,50,100,200,500)`
* **max_depth** : The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance. The selected range is `(1, 5, 10, 20, 30, 50, 100)`
* **learning_rate** : Learning rate shrinks the contribution of each tree. The selected range of parameters is `(1, 0.1, 0.01, 0.001)`

### Results
Below images shows image of run widget and best performing model run. 

Which is shown in below image after deployement.
![HyperDrive_run](Resources/Images/HyperDrive_Run.PNG)
![HyperDrive_Best_Model](Resources/Images/HyperDrive_Best_Model.PNG)

Out best model is having below accuracy and parameters:
```
Accuracy: 0.9790209790209791
learning_rate: 0.1
max_depth: 1
n_estimators: 500
```

## Model Deployment
Best performing model is from AutoML run. I deployed **stackensembleclassifier** from automl run. 
![Active_endpoint](Resources/Images/Endpoint_Active.PNG)
![Active_Status](Resources/Images/Active_Endpoint.PNG)
![State_Endpoint](Resources/Images/State_Endpoint.PNG)

After Deployment we can perform inference using RestEndpoint URL/ Scoring URI by simple post request. I have provided Sample Code for doing inference in [AutoML Notebook](automl.ipynb).

## Screen Recording
[Youtube](https://youtu.be/DfyGiSjVQm4)

## Future Improvement
* Perform Inference on IOT device using Deployed Model
* Try LightGBM/SGD algorithm in Hyperdrive since it seems to give better comparative results
* Convert Model to ONNX format and save it
* Enable Application Insight/Login in the app

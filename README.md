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
```python
from sklearn import datasets
data = datasets.load_breast_cancer(as_frame=True)

```
We should need to register dataset in Azure Studio while using it in AutoML we can do it like this,
```python
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.data.datapath import DataPath

# Create TabularDataset using TabularDatasetFactory
def_blob_store = ws.get_default_datastore()
print("Default datastore's name: {}".format(def_blob_store.name))
data_path = DataPath(datastore=def_blob_store, path_on_datastore='datapath')
ds = TabularDatasetFactory.register_pandas_dataframe(df, name='UCI_ML_Breast_Cancer', target=data_path)
```

## Automated ML:
AutoML does everything like model selection and Hyperparameter Tuning internally. We just need to provide details related to Task, Input and Compute. Here is the configuration for my automl experiment run.
```python
automl_settings = {
    "experiment_timeout_minutes": 50,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'accuracy'}

automl_config = AutoMLConfig(
    task='classification',
    training_data = ds,
    label_column_name = "target",
    compute_target=compute_target,
    **automl_settings)
```
* `task`: we need to specify task type like `classification` / `regression` / `time-series`. Our's is classification,
* `training_data` : we need to provide entire data including labels/target. Note the dataset object should be `TabularDatasetFactory` type,
* `label_column_name` : The column name of Label Column / Target Column
* `compute_target` : is the object of training cluster on which automl job will be running
* `experiment_timeout_minutes`: We choose  `experiment_timeout_minutes=50mins` to give enough time to try all experiments,  
* `max_concurrent_iterations`: `5` according to max_nodes given while creating `training_instance`, 
* `primary_metric` : We choose Accuracy as Primary Metrics since data is not highly imbalanced it seems good fit for binary classification problem.

### Results
We got two models having highest same `98.94%` score:
1. Voting Ensemble
2. Stack Ensemble

It can be seen in below image
![AutoML_Run](Resources/Images/AutoML_Run.PNG)
![Best AutoML Model](/Resources/Images/AutoML_Best_Model.PNG)

Both the best performing models are made up of multiple models i have shown models available in our best performing `stackensembleclassifier` in the [AutoML Notebook](automl.ipynb).

Stack Ensemble Model is made up of below parameters and Models,   
```
stackensembleclassifier
{'16': Pipeline(memory=None,
         steps=[('sparsenormalizer',
                 <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f554fe1bba8>),
                ('xgboostclassifier',
                 XGBoostClassifier(base_score=0.5, booster='gbtree',
                                   colsample_bylevel=1, colsample_bynode=1,
                                   colsample_bytree=0.7, eta=0.2, gamma=0,
                                   learning_rate=0.1, max_delta_step=0,
                                   max_depth=8, max_leaves=255,
                                   min_child_weight=1, missing=nan,
                                   n_estimators=100, n_jobs=1, nthread=None,
                                   objective='reg:logistic', random_state=0,
                                   reg_alpha=1.5625,
                                   reg_lambda=0.8333333333333334,
                                   scale_pos_weight=1, seed=None, silent=None,
                                   subsample=0.7, tree_method='auto',
                                   verbose=-10, verbosity=0))],
         verbose=False),
 '16__memory': None,
 '16__sparsenormalizer': <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f554fe1bba8>,
 '16__sparsenormalizer__copy': True,
 '16__sparsenormalizer__norm': 'max',
 '16__steps': [('sparsenormalizer',
                <azureml.automl.runtime.shared.model_wrappers.SparseNormalizer object at 0x7f554fe1bba8>),
               ('xgboostclassifier',
                XGBoostClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.7, eta=0.2, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=8,
                  max_leaves=255, min_child_weight=1, missing=nan,
                  n_estimators=100, n_jobs=1, nthread=None,
                  objective='reg:logistic', random_state=0, reg_alpha=1.5625,
                  reg_lambda=0.8333333333333334, scale_pos_weight=1, seed=None,
                  silent=None, subsample=0.7, tree_method='auto', verbose=-10,
                  verbosity=0))],
 '16__verbose': False,
 '16__xgboostclassifier': XGBoostClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.7, eta=0.2, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=8,
                  max_leaves=255, min_child_weight=1, missing=nan,
                  n_estimators=100, n_jobs=1, nthread=None,
                  objective='reg:logistic', random_state=0, reg_alpha=1.5625,
                  reg_lambda=0.8333333333333334, scale_pos_weight=1, seed=None,
                  silent=None, subsample=0.7, tree_method='auto', verbose=-10,
                  verbosity=0),
 '16__xgboostclassifier__base_score': 0.5,
 '16__xgboostclassifier__booster': 'gbtree',
 '16__xgboostclassifier__colsample_bylevel': 1,
 '16__xgboostclassifier__colsample_bynode': 1,
 '16__xgboostclassifier__colsample_bytree': 0.7,
 '16__xgboostclassifier__eta': 0.2,
 '16__xgboostclassifier__gamma': 0,
 '16__xgboostclassifier__learning_rate': 0.1,
 '16__xgboostclassifier__max_delta_step': 0,
 '16__xgboostclassifier__max_depth': 8,
 '16__xgboostclassifier__max_leaves': 255,
 '16__xgboostclassifier__min_child_weight': 1,
 '16__xgboostclassifier__missing': nan,
 '16__xgboostclassifier__n_estimators': 100,
 '16__xgboostclassifier__n_jobs': 1,
 '16__xgboostclassifier__nthread': None,
 '16__xgboostclassifier__objective': 'reg:logistic',
 '16__xgboostclassifier__random_state': 0,
 '16__xgboostclassifier__reg_alpha': 1.5625,
 '16__xgboostclassifier__reg_lambda': 0.8333333333333334,
 '16__xgboostclassifier__scale_pos_weight': 1,
 '16__xgboostclassifier__seed': None,
 '16__xgboostclassifier__silent': None,
 '16__xgboostclassifier__subsample': 0.7,
 '16__xgboostclassifier__tree_method': 'auto',
 '16__xgboostclassifier__verbose': -10,
 '16__xgboostclassifier__verbosity': 0,
 '34': Pipeline(memory=None,
         steps=[('standardscalerwrapper',
                 <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f554fe06898>),
                ('sgdclassifierwrapper',
                 SGDClassifierWrapper(alpha=1.4286571428571428,
                                      class_weight=None, eta0=0.01,
                                      fit_intercept=True,
                                      l1_ratio=0.7551020408163265,
                                      learning_rate='constant', loss='log',
                                      max_iter=1000, n_jobs=1, penalty='none',
                                      power_t=0.4444444444444444,
                                      random_state=None, tol=0.001))],
         verbose=False),
 '34__memory': None,
 '34__sgdclassifierwrapper': SGDClassifierWrapper(alpha=1.4286571428571428, class_weight=None, eta0=0.01,
                     fit_intercept=True, l1_ratio=0.7551020408163265,
                     learning_rate='constant', loss='log', max_iter=1000,
                     n_jobs=1, penalty='none', power_t=0.4444444444444444,
                     random_state=None, tol=0.001),
 '34__sgdclassifierwrapper__alpha': 1.4286571428571428,
 '34__sgdclassifierwrapper__class_weight': None,
 '34__sgdclassifierwrapper__eta0': 0.01,
 '34__sgdclassifierwrapper__fit_intercept': True,
 '34__sgdclassifierwrapper__l1_ratio': 0.7551020408163265,
 '34__sgdclassifierwrapper__learning_rate': 'constant',
 '34__sgdclassifierwrapper__loss': 'log',
 '34__sgdclassifierwrapper__max_iter': 1000,
 '34__sgdclassifierwrapper__n_jobs': 1,
 '34__sgdclassifierwrapper__penalty': 'none',
 '34__sgdclassifierwrapper__power_t': 0.4444444444444444,
 '34__sgdclassifierwrapper__random_state': None,
 '34__sgdclassifierwrapper__tol': 0.001,
 '34__standardscalerwrapper': <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f554fe06898>,
 '34__standardscalerwrapper__class_name': 'StandardScaler',
 '34__standardscalerwrapper__copy': True,
 '34__standardscalerwrapper__module_name': 'sklearn.preprocessing._data',
 '34__standardscalerwrapper__with_mean': True,
 '34__standardscalerwrapper__with_std': True,
 '34__steps': [('standardscalerwrapper',
                <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f554fe06898>),
               ('sgdclassifierwrapper',
                SGDClassifierWrapper(alpha=1.4286571428571428, class_weight=None, eta0=0.01,
                     fit_intercept=True, l1_ratio=0.7551020408163265,
                     learning_rate='constant', loss='log', max_iter=1000,
                     n_jobs=1, penalty='none', power_t=0.4444444444444444,
                     random_state=None, tol=0.001))],
 '34__verbose': False,
 '36': Pipeline(memory=None,
         steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                ('logisticregression',
                 LogisticRegression(C=4714.8663634573895, class_weight=None,
                                    dual=False, fit_intercept=True,
                                    intercept_scaling=1, l1_ratio=None,
                                    max_iter=100, multi_class='ovr', n_jobs=1,
                                    penalty='l1', random_state=None,
                                    solver='saga', tol=0.0001, verbose=0,
                                    warm_start=False))],
         verbose=False),
 '36__logisticregression': LogisticRegression(C=4714.8663634573895, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='ovr', n_jobs=1, penalty='l1',
                   random_state=None, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False),
 '36__logisticregression__C': 4714.8663634573895,
 '36__logisticregression__class_weight': None,
 '36__logisticregression__dual': False,
 '36__logisticregression__fit_intercept': True,
 '36__logisticregression__intercept_scaling': 1,
 '36__logisticregression__l1_ratio': None,
 '36__logisticregression__max_iter': 100,
 '36__logisticregression__multi_class': 'ovr',
 '36__logisticregression__n_jobs': 1,
 '36__logisticregression__penalty': 'l1',
 '36__logisticregression__random_state': None,
 '36__logisticregression__solver': 'saga',
 '36__logisticregression__tol': 0.0001,
 '36__logisticregression__verbose': 0,
 '36__logisticregression__warm_start': False,
 '36__memory': None,
 '36__minmaxscaler': MinMaxScaler(copy=True, feature_range=(0, 1)),
 '36__minmaxscaler__copy': True,
 '36__minmaxscaler__feature_range': (0, 1),
 '36__steps': [('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
               ('logisticregression',
                LogisticRegression(C=4714.8663634573895, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='ovr', n_jobs=1, penalty='l1',
                   random_state=None, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False))],
 '36__verbose': False,
 '41': Pipeline(memory=None,
         steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
                ('svcwrapper',
                 SVCWrapper(C=35.564803062231285, break_ties=False,
                            cache_size=200, class_weight='balanced', coef0=0.0,
                            decision_function_shape='ovr', degree=3,
                            gamma='scale', kernel='rbf', max_iter=-1,
                            probability=True, random_state=None, shrinking=True,
                            tol=0.001, verbose=False))],
         verbose=False),
 '41__maxabsscaler': MaxAbsScaler(copy=True),
 '41__maxabsscaler__copy': True,
 '41__memory': None,
 '41__steps': [('maxabsscaler', MaxAbsScaler(copy=True)),
               ('svcwrapper',
                SVCWrapper(C=35.564803062231285, break_ties=False, cache_size=200,
           class_weight='balanced', coef0=0.0, decision_function_shape='ovr',
           degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=True,
           random_state=None, shrinking=True, tol=0.001, verbose=False))],
 '41__svcwrapper': SVCWrapper(C=35.564803062231285, break_ties=False, cache_size=200,
           class_weight='balanced', coef0=0.0, decision_function_shape='ovr',
           degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=True,
           random_state=None, shrinking=True, tol=0.001, verbose=False),
 '41__svcwrapper__C': 35.564803062231285,
 '41__svcwrapper__break_ties': False,
 '41__svcwrapper__cache_size': 200,
 '41__svcwrapper__class_weight': 'balanced',
 '41__svcwrapper__coef0': 0.0,
 '41__svcwrapper__decision_function_shape': 'ovr',
 '41__svcwrapper__degree': 3,
 '41__svcwrapper__gamma': 'scale',
 '41__svcwrapper__kernel': 'rbf',
 '41__svcwrapper__max_iter': -1,
 '41__svcwrapper__probability': True,
 '41__svcwrapper__random_state': None,
 '41__svcwrapper__shrinking': True,
 '41__svcwrapper__tol': 0.001,
 '41__svcwrapper__verbose': False,
 '41__verbose': False,
 '42': Pipeline(memory=None,
         steps=[('standardscalerwrapper',
                 <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f55596b7c88>),
                ('logisticregression',
                 LogisticRegression(C=2222.996482526191, class_weight=None,
                                    dual=False, fit_intercept=True,
                                    intercept_scaling=1, l1_ratio=None,
                                    max_iter=100, multi_class='ovr', n_jobs=1,
                                    penalty='l2', random_state=None,
                                    solver='saga', tol=0.0001, verbose=0,
                                    warm_start=False))],
         verbose=False),
 '42__logisticregression': LogisticRegression(C=2222.996482526191, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2',
                   random_state=None, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False),
 '42__logisticregression__C': 2222.996482526191,
 '42__logisticregression__class_weight': None,
 '42__logisticregression__dual': False,
 '42__logisticregression__fit_intercept': True,
 '42__logisticregression__intercept_scaling': 1,
 '42__logisticregression__l1_ratio': None,
 '42__logisticregression__max_iter': 100,
 '42__logisticregression__multi_class': 'ovr',
 '42__logisticregression__n_jobs': 1,
 '42__logisticregression__penalty': 'l2',
 '42__logisticregression__random_state': None,
 '42__logisticregression__solver': 'saga',
 '42__logisticregression__tol': 0.0001,
 '42__logisticregression__verbose': 0,
 '42__logisticregression__warm_start': False,
 '42__memory': None,
 '42__standardscalerwrapper': <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f55596b7c88>,
 '42__standardscalerwrapper__class_name': 'StandardScaler',
 '42__standardscalerwrapper__copy': True,
 '42__standardscalerwrapper__module_name': 'sklearn.preprocessing._data',
 '42__standardscalerwrapper__with_mean': True,
 '42__standardscalerwrapper__with_std': True,
 '42__steps': [('standardscalerwrapper',
                <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f55596b7c88>),
               ('logisticregression',
                LogisticRegression(C=2222.996482526191, class_weight=None, dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2',
                   random_state=None, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False))],
 '42__verbose': False,
 '43': Pipeline(memory=None,
         steps=[('standardscalerwrapper',
                 <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f554fe15898>),
                ('sgdclassifierwrapper',
                 SGDClassifierWrapper(alpha=2.040895918367347,
                                      class_weight='balanced', eta0=0.0001,
                                      fit_intercept=True,
                                      l1_ratio=0.6938775510204082,
                                      learning_rate='constant',
                                      loss='modified_huber', max_iter=1000,
                                      n_jobs=1, penalty='none',
                                      power_t=0.8888888888888888,
                                      random_state=None, tol=0.0001))],
         verbose=False),
 '43__memory': None,
 '43__sgdclassifierwrapper': SGDClassifierWrapper(alpha=2.040895918367347, class_weight='balanced',
                     eta0=0.0001, fit_intercept=True,
                     l1_ratio=0.6938775510204082, learning_rate='constant',
                     loss='modified_huber', max_iter=1000, n_jobs=1,
                     penalty='none', power_t=0.8888888888888888,
                     random_state=None, tol=0.0001),
 '43__sgdclassifierwrapper__alpha': 2.040895918367347,
 '43__sgdclassifierwrapper__class_weight': 'balanced',
 '43__sgdclassifierwrapper__eta0': 0.0001,
 '43__sgdclassifierwrapper__fit_intercept': True,
 '43__sgdclassifierwrapper__l1_ratio': 0.6938775510204082,
 '43__sgdclassifierwrapper__learning_rate': 'constant',
 '43__sgdclassifierwrapper__loss': 'modified_huber',
 '43__sgdclassifierwrapper__max_iter': 1000,
 '43__sgdclassifierwrapper__n_jobs': 1,
 '43__sgdclassifierwrapper__penalty': 'none',
 '43__sgdclassifierwrapper__power_t': 0.8888888888888888,
 '43__sgdclassifierwrapper__random_state': None,
 '43__sgdclassifierwrapper__tol': 0.0001,
 '43__standardscalerwrapper': <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f554fe15898>,
 '43__standardscalerwrapper__class_name': 'StandardScaler',
 '43__standardscalerwrapper__copy': True,
 '43__standardscalerwrapper__module_name': 'sklearn.preprocessing._data',
 '43__standardscalerwrapper__with_mean': True,
 '43__standardscalerwrapper__with_std': True,
 '43__steps': [('standardscalerwrapper',
                <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f554fe15898>),
               ('sgdclassifierwrapper',
                SGDClassifierWrapper(alpha=2.040895918367347, class_weight='balanced',
                     eta0=0.0001, fit_intercept=True,
                     l1_ratio=0.6938775510204082, learning_rate='constant',
                     loss='modified_huber', max_iter=1000, n_jobs=1,
                     penalty='none', power_t=0.8888888888888888,
                     random_state=None, tol=0.0001))],
 '43__verbose': False,
 'base_learners': None,
 'meta_learner': None,
 'metalearner__Cs': 10,
 'metalearner__class_weight': None,
 'metalearner__cv': None,
 'metalearner__dual': False,
 'metalearner__fit_intercept': True,
 'metalearner__intercept_scaling': 1.0,
 'metalearner__l1_ratios': None,
 'metalearner__max_iter': 100,
 'metalearner__multi_class': 'auto',
 'metalearner__n_jobs': None,
 'metalearner__penalty': 'l2',
 'metalearner__random_state': None,
 'metalearner__refit': True,
 'metalearner__scoring': <azureml.automl.runtime.stack_ensemble_base.Scorer object at 0x7f554fe20e80>,
 'metalearner__solver': 'lbfgs',
 'metalearner__tol': 0.0001,
 'metalearner__verbose': 0,
 'training_cv_folds': None}
```
It is having `XGBoostClassifier`, `SGDClassifierWrapper`, `LogisticRegression`, `SVCWrapper` etc with different hyperparameters mentioned above,

While `Voting Ensemble`  is also made of many models details are mentioned below,

```

```

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
Model Name: GradientBoostingClassifier
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

Sample Input: 
```python
sample = {"data": [
{"mean radius": 16.65, "mean texture": 21.38, "mean perimeter": 110.0, "mean area": 904.6, "mean smoothness": 0.1121, "mean compactness": 0.1457, "mean concavity": 0.1525, "mean concave points": 0.0917, "mean symmetry": 0.1995, "mean fractal dimension": 0.0633, "radius error": 0.8068, "texture error": 0.9017, "perimeter error": 5.455, "area error": 102.6, "smoothness error": 0.006048, "compactness error": 0.01882, "concavity error": 0.02741, "concave points error": 0.0113, "symmetry error": 0.01468, "fractal dimension error": 0.002801, "worst radius": 26.46, "worst texture": 31.56, "worst perimeter": 177.0, "worst area": 2215.0, "worst smoothness": 0.1805, "worst compactness": 0.3578, "worst concavity": 0.4695, "worst concave points": 0.2095, "worst symmetry": 0.3613, "worst fractal dimension": 0.09564}, 
{"mean radius": 17.27, "mean texture": 25.42, "mean perimeter": 112.4, "mean area": 928.8, "mean smoothness": 0.08331, "mean compactness": 0.1109, "mean concavity": 0.1204, "mean concave points": 0.05736, "mean symmetry": 0.1467, "mean fractal dimension": 0.05407, "radius error": 0.51, "texture error": 1.679, "perimeter error": 3.283, "area error": 58.38, "smoothness error": 0.008109, "compactness error": 0.04308, "concavity error": 0.04942, "concave points error": 0.01742, "symmetry error": 0.01594, "fractal dimension error": 0.003739, "worst radius": 20.38, "worst texture": 35.46, "worst perimeter": 132.8, "worst area": 1284.0, "worst smoothness": 0.1436, "worst compactness": 0.4122, "worst concavity": 0.5036, "worst concave points": 0.1739, "worst symmetry": 0.25, "worst fractal dimension": 0.07944}
]}
```

How to do inference using python?
```python
import requests 
# Set the content type
headers = {'Content-type': 'application/json'}
response = requests.post(service.scoring_uri, test_sample, headers=headers)
print("response")
print(response.text)
```

## Screen Recording
[Youtube](https://youtu.be/DfyGiSjVQm4)

## Future Improvement
- [ ] Perform Inference on IOT device using Deployed Model
- [ ] Try LightGBM/SGD algorithm in Hyperdrive since it seems to give better comparative results
- [ ] Convert Model to ONNX format and save it
- [ ] Enable Application Insight/Login in the app

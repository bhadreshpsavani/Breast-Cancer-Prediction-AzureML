from sklearn.ensemble import GradientBoostingClassifier
import argparse
import os
import numpy as np
# from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from sklearn import datasets

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=10, help="The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performancThe maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree.")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate shrinks the contribution of each tree by learning_rateThe number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performancThe maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree.")
    parser.add_argument('--max_depth', type=int, default=1, help="The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree.")
    
    args = parser.parse_args()

    run.log("Number of estimators:", np.int(args.n_estimators))
    run.log("Learning Rate:", np.float(args.learning_rate))
    run.log("Maximum Depth of Tree:", np.int(args.max_depth))

    data = datasets.load_breast_cancer()

    x, y = data.data, data.target

    x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=12345)

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators, 
        learning_rate=args.learning_rate, 
        max_depth=args.max_depth, 
        random_state=12345).fit(x_train, y_train)

    # Calculate accuracy
    accuracy = model.score(x_test, y_test)
    # y_prob = model.predict_proba(x_test)[:, 1]
    # AUC = roc_auc_score(y_test, y_prob, average='weighted')
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
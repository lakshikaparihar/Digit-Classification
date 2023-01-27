import os
import warnings
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the MNIST data from sklearn    
    mnist = fetch_openml('mnist_784',version=1)

    # Split the data and target columns.
    x, y = mnist['data'],mnist['target']

    # Split the train and test data into 60000:10000 as its default in MNIST dataset.
    x_train , x_test , y_train , y_test = x.iloc[:60000], x.iloc[60000:], y.iloc[:60000], y.iloc[60000:]

    #  Convert the target or y into a integer as we are trying to identify digits
    y_train, y_test = y_train.astype(np.uint8), y_test.astype(np.uint8)
    
    with mlflow.start_run():
        #Training a Logistic Regression Model
        model = LogisticRegression()
        model.fit(x_train, y_train)

        #Evaluating and Logging metrics
        predicted_digits = model.predict(x_test)
        (rmse, mae, r2) = eval_metrics(y_test, predicted_digits)
        print("Logistic Regression metric Evaluation")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        model_data = mlflow.sklearn.log_model(
            model,
            "digit-classification-model", 
            registered_model_name="digit-classification-model" )
        
        print("Model Logged into Mlflow Registry")

        # Transition into production
        client = MlflowClient()
        client.transition_model_version_stage(
            name="digit-classification-model",
            version=1,
            stage="Production")


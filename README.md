# Digit-Classification
An End-to-End Usecase of digit classification from training till inferencing

### Dataset we are using

Link : http://yann.lecun.com/exdb/mnist/


* MNIST is a set of 70000 small images of 28x28 pixels.
* Value of one pixel's intensity is between 0(white) and 255(black).
* Each image has 784 features (28x28)

### Algorithm we are using

Logistic Regression (As accuracy is not the criteria we will keep it simple)

### MLOps tool we are using 

MLflow would be used for model repo , artifact storage, versioning and other things in the pipeline

### Inferencing tool we are using 

We will use mlflow dockerfile to inference our model but I have optimized the file to lower the image size. We can use the same file for other models as well. Only need to change a single line thats the reference of the model which we have logged in


### Prerequisites

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"
pip3 install requirements.txt
```

### To Run the UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Things you can do in MLflow UI :
* Register model into production
* check all the models with versions and other data regarding the model
* Easily switch between model versions in production

[](https://github.com/lakshikaparihar/Digit-Classification/blob/main/images/mlflow-ui.png)


## To run the project locally

1. clone the repo
2. run the train and log file

```bash
python3 train_log.py
```

   It will train you model and  create a pickle file as well also create local sqlite database (mlruns.db) and a artifact folder (mlruns).

3. test the model locally before serving

```bash
python3 test.py runs:/<run-id>/digit-classification-model
```

3. Now you can serve the model through two ways 

    1. using mlflow serve command

   ```bash
   mlflow models serve -m models:/digit-classification-model/production -p 2000 --env-manager local
   ```
   **Note :** If you are facing error in serving then try set the MLFLOW_TRACKING_URI on your terminal

   (For linux) 
   ```bash
   export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"
   ```
   
    2. using mlflow optimized docker image
    
   ```bash
   docker pull lakshika1064/ml_models:digitClassificationSklearn
   ```
   **Note :**  If you are facing error in pulling the docker image, than you can just build it locally

   ```bash
   docker build -t lakshika1064/ml_models:digitClassificationSklearn .
   ```
   ```bash
   docker run -p 2000:8080 lakshika1064/ml_models:digitClassificationSklearn
   ```
   
4. Finally you can hit the curl command either from terminal or postman and get the prediction

   ```
   curl http://127.0.0.1:2000/invocations -H "Content-Type: application/json" -d '{"instances":[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,94.0,163.0,99.0,228.0,255.0,202.0,49.0,58.0,47.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,171.0,245.0,253.0,253.0,253.0,254.0,221.0,236.0,174.0,173.0,72.0,136.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,254.0,253.0,253.0,253.0,253.0,208.0,128.0,197.0,250.0,243.0,142.0,123.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,241.0,253.0,253.0,199.0,80.0,35.0,23.0,47.0,87.0,87.0,97.0,110.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,137.0,253.0,253.0,54.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,92.0,255.0,254.0,254.0,119.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.0,158.0,254.0,253.0,199.0,4.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,77.0,253.0,254.0,180.0,31.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.0,203.0,253.0,254.0,108.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,37.0,253.0,253.0,254.0,43.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,14.0,219.0,254.0,255.0,18.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,199.0,253.0,228.0,62.0,55.0,55.0,55.0,88.0,35.0,55.0,12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,109.0,253.0,254.0,253.0,253.0,253.0,253.0,254.0,240.0,253.0,186.0,95.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,11.0,215.0,254.0,253.0,253.0,253.0,253.0,254.0,253.0,253.0,253.0,253.0,84.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,26.0,189.0,253.0,253.0,253.0,253.0,228.0,162.0,207.0,253.0,253.0,254.0,18.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,14.0,85.0,0.0,0.0,0.0,0.0,40.0,207.0,255.0,109.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,89.0,248.0,254.0,56.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,21.0,159.0,245.0,253.0,165.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,79.0,200.0,230.0,253.0,245.0,137.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,117.0,163.0,194.0,194.0,61.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]}'
   ```

#### Cleanup script

```bash
rm -rf mlruns*
```

**Note :** 

*  We have put the model into production through code but we can do it manually through mlflow ui as well
*  To create docker image we can also use mlflow docker command :
```bash
mlflow models build-docker -m "models:/digit-classification-model/production" -n "digit-classification-model"
```
* you can get the run-id from the mlrun folder

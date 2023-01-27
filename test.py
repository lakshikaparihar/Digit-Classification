import mlflow
import sys
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")
logged_model_uri = sys.argv[1]

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model_uri)

# Predict on a Pandas DataFrame.
data = pd.read_csv("test_data.csv",header=None)

start = time.time()
print("Output : " , loaded_model.predict(data))
end = time.time()
print("latency : ",end - start)
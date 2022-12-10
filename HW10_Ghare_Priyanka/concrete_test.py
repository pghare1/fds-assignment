import pandas as pd
from sklearn.metrics import r2_score
import pycaret.regression as carreg
from time import perf_counter
import glob
import os

# Read in testing data (none of the colums are an index)
test_df = pd.read_csv("test_concrete.csv", index_col=False)

## Your code here


# Break into two dataframes

X_test = test_df.drop(columns=["csMPa"])
## Your code here
y_test = test_df["csMPa"]
## Your code here

# Get a list of all the pkl files in the current directory
model_files = glob.glob("*.pkl")

## Your code here

# Hack off the .pkl extension
model_names = [os.path.splitext(p)[0] for p in model_files]

# For each model
for model_name in model_names:

    # Load the model
    model = carreg.load_model(model_name)

    ## Your code here

    # Do the inference
    t1 = perf_counter()
    y_pred = model.predict(X_test)

    ## Your code here
    t2 = perf_counter()

    # Get the R2 score
    r2 = r2_score(y_test, y_pred)

    ## Your code here

    # Print the results
    print(f"{model_name}:")
    print(f"\tInference: {t2 - t1:.4f} seconds")
    print(f"\tR2 on test data = {r2:.4f}")

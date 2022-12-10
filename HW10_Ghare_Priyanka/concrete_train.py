import pycaret.regression as carreg
from time import perf_counter
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# How many models we will tune
n_select = 6

# Read in the training data
train_df = pd.read_csv("train_concrete.csv")
## Your code here

# Set up the regression experiment session
print("* Setting up session***")
t1 = perf_counter()
exp1 = carreg.setup(train_df, target="csMPa", session_id=8371)

## Your code here
t2 = perf_counter()
print(f"* Set up: {t2 - t1:.2f} seconds")

# Do a basic comparison of models (no turbo)
best = carreg.compare_models(sort="R2", n_select=n_select, turbo=False)
## Your code here
t3 = perf_counter()
print(f"* compare_models: {t3 - t2:.2f} seconds")

# List the best models
print(f"* Best:")
for b in best:
    print(f"\t{b.__class__.__name__}")

# Go through the list of models
for i, model in enumerate(best):

    # Tune the model (try 24 parameter combinations)
    print(f"* Tuning {model.__class__.__name__}")
    tuned = carreg.tune_model(model, n_iter=24, optimize="R2")

    ## Your code here

    # Finalize the model
    print(f"* Finalizing {model.__class__.__name__}")
    final = carreg.finalize_model(tuned)

    ## Your code here

    # Save the model
    print(f"* Saving {model.__class__.__name__}")
    carreg.save_model(final, f"{best[i].__class__.__name__}")

    ## Your code here

t4 = perf_counter()
print(f"* Tuning and finalizing: {t4 - t3:.2f} seconds")
print(f"* Total time: {t4 - t1:.2f} seconds")
select = 6

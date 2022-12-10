import pickle as pkl
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from time import perf_counter
import numpy as np
from keras.datasets import mnist
import pandas as pd


def show_stats(cv_results):
    ## Your code here
    df = pd.DataFrame(cv_results)
    df = df.sort_values(by="rank_test_score")
    df.reset_index(inplace=True)

    for i in range(len(df)):
        print(
            f'metric={df.loc[i,"params"]["metric"]} n_neighbors={df.loc[i,"params"]["metric"]} weights={df.loc[i,"params"]["weights"]} : Mean accuracy={round(100.0*df.loc[i,"mean_test_score"],2)}%'
        )


# Read in the MNIST Training DAta
d = 28 * 28
(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape((-1, d))
n = X_train.shape[0]

# Make a dictionary with all the parameter values you want to test
## Your code here
parameters = {
    "metric": ["euclidean", "manhattan"],
    "n_neighbors": [1, 3, 5, 7],
    "weights": ["uniform", "distance"],
}

# Create a KNN classifier
## Your code here
knn_model = KNeighborsClassifier()

# Create a GridSearchCV to determine the best values for the parameters
grid_searcher = GridSearchCV(knn_model, parameters, verbose=3, cv=4)  ## Your code here

# Run it
start = perf_counter()
## Your code here
grid_searcher.fit(X_train, y_train)
print(f"Took {perf_counter() - start:.1f} seconds")

# List out the combinations and their scores
show_stats(grid_searcher.cv_results_)

# Get the best values
best_combo_index = grid_searcher.best_index_  ## Your code here
best_params = grid_searcher.best_params_  ## Your code here

print(f"Best parameters: {best_params}")

# Create a new classifier with the best hyperparameters
## Your code here
classifier = KNeighborsClassifier(metric="euclidean", n_neighbors=3, weights="distance")
# get all the parameters from classifier
print(f"Params: {classifier.get_params()}")

# Fit to training data
## Your code here
classifier.fit(X_train, y_train)

# Do predictions for the training data, and print the accuracy
## Your code here
y_pred = classifier.predict(X_train)

acc = accuracy_score(y_train, y_pred)
print(f"Accuracy on training data = { 100.0*acc} %")

# Write the classifier out to a pickle file
with open("knn_model.pkl", "wb") as f:
    pkl.dump(classifier, f)
print("Wrote knn_model.pkl")

import pandas as pd
import numpy as np
import pickle as pkl
import util

TRAIN_PATH = "train_gn.csv"
PARAMETERS_PATH = "parameters_gn.pkl"


def show_array(category_label, array, labels):
    print(f"\t{category_label} -> ", end="")
    for i in range(len(array)):
        print(f"{labels[i]}:{array[i]: >7.4f}     ", end="")
    print()


train_df = pd.read_csv(TRAIN_PATH)
X_df = train_df.iloc[:, :-1]
Y_df = train_df.iloc[:, -1]
n = len(X_df)
d = len(X_df.columns)
print(f"Read {n} samples with {d} attributes from {TRAIN_PATH}")

class_labels = np.unique(Y_df)
class_means = np.zeros((len(class_labels), d))
class_stds = np.zeros((len(class_labels), d))
for i in range(len(class_labels)):
    class_label = class_labels[i]
    class_df = X_df[Y_df == class_label]
    class_means[i] = np.mean(class_df, axis=0)
    class_stds[i] = np.std(class_df, axis=0)

# compute the prior probabilities of each class
class_priors = np.zeros(len(class_labels))
for i in range(len(class_labels)):
    class_label = class_labels[i]
    class_priors[i] = np.sum(Y_df == class_label) / n

# Print calculated prior percentange
print("Priors:")
for i in range(len(class_labels)):
    print(f"\t{class_labels[i]}: {class_priors[i]*100:.1f}%")

# Print calculated mean and standard deviation
for i in range(len(class_labels)):
    class_label = class_labels[i]
    print(f"{class_label}:")
    show_array("Means", class_means[i], X_df.columns)
    show_array("Stdvs", class_stds[i], X_df.columns)

# Save the parameters
parameters = {
    "class_labels": class_labels,
    "class_means": class_means,
    "class_stds": class_stds,
    "class_priors": class_priors,
}
with open(PARAMETERS_PATH, "wb") as f:
    pkl.dump(parameters, f)
print(f"Wrote parameters to {PARAMETERS_PATH}")

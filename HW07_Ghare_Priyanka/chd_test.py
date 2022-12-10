import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import logging
import warnings

warnings.filterwarnings("ignore")
# Configure logger
## Your code here
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Read in test data
## Your code here
test = pd.read_csv("test.csv", index_col=False)

# Read in model
## Your code here
with open("classifier.pkl", "rb") as f:
    scaler, model = pickle.load(f)


# Scale X

## Your code here
X_test = test.drop(columns=["TenYearCHD"])


# Check accuracy on test data
## Your code here
X_test = scaler.transform(X_test)
y_test = test["TenYearCHD"]
test_accuracy = model.score(X_test, y_test)


print(f"Test Accuracy = {test_accuracy}")

# Show confusion matrix
## Your code here
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix = \n{cm}")


# Try a bunch of thresholds

threshold = 0.0
best_f1 = -1.0
thresholds = []
recall_scores = []
precision_scores = []
f1_scores = []
while threshold <= 0.80:
    thresholds.append(threshold)

    ## Your code here
    y_pred = model.predict_proba(X_test)[:, 1] > threshold
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = model.score(X_test, y_pred > threshold)
    f1 = 2 * (precision * recall) / (precision + recall)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

    logger = logging.getLogger()

    recall_scores.append(recall)
    precision_scores.append(precision)
    f1_scores.append(f1)
    logger.info(
        f"Threshold={threshold:.3f} Accuracy={accuracy:.3f} Recall={recall:.2f} Precision={precision:.2f} F1 = {f1:.3f}"
    )
    threshold += 0.02


# Make a confusion matrix for the best threshold
## Your code here
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold = {best_threshold}")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred > best_threshold)
print(f"Confusion matrix = \n{cm}")

# Plot recall, precision, and F1 vs Threshold
fig, ax = plt.subplots()
ax.plot(thresholds, recall_scores, "b", label="Recall")
ax.plot(thresholds, precision_scores, "g", label="Precision", color="g")
ax.plot(thresholds, f1_scores, "r", label="F1", color="r")
ax.vlines(best_threshold, 0, 1, "r", linewidth=0.5, linestyle="dashed")
ax.set_xlabel("Threshold")
ax.legend()
fig.savefig("threshold.png")

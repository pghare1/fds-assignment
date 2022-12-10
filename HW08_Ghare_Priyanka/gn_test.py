import pandas as pd
import numpy as np
import pickle as pk
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import csv
import util

TESTCSV = "test_gn.csv"
CONFUSION_GN = "confusion_gn.png"
PLOT_GN = "confidence_gn.png"
PARAMETERS_GN = "parameters_gn.pkl"

print(f"Read parameters from {PARAMETERS_GN}")
with open(PARAMETERS_GN, "rb") as f:
    parameters = pk.load(f)

test_data = pd.read_csv(TESTCSV)
X = test_data.iloc[:, :-1].to_numpy(dtype=np.float64)
n = X.shape[0]
Y_df = test_data.iloc[:, -1]
print(f"Read {len(test_data)} rows from {TESTCSV}")

pred_values = []
pred_confidences = []
constant = np.log(2 * np.pi) / 2

# initialize this to skip column name row
skip = 0
# Step through the test.csv file
with open("test_gn.csv", "r") as f:
    reader = csv.reader(f)

    skip_tweet_cnt = 0

    prediction = []
    pred_prob = []
    gt = []
    log_of_posteriors = []
    for row in reader:
        # print(row)
        if skip == 0:
            skip += 1
            continue

        # Check to see if the row has two entries
        if len(row) != 5:
            continue

        confidence = []

        for index_labels in range(len(parameters["class_labels"])):
            likelihood = 0
            for i in range(4):
                like_at = (
                    -np.log(parameters["class_stds"][index_labels][i])
                    - constant
                    - (
                        (
                            (float(row[i]) - parameters["class_means"][index_labels][i])
                            / parameters["class_stds"][index_labels][i]
                        )
                        ** 2
                        / 2
                    )
                )

                likelihood += like_at

            log_post = likelihood + np.log(
                parameters["class_priors"][index_labels]
            )
            log_of_posteriors.append(log_post)
            confidence.append(log_post)

        prediction.append(np.argmax(confidence))
        pred_prob.append(max(confidence))
        gt.append(list(parameters["class_labels"]).index(row[-1]))

n = len(log_of_posteriors) // 5
l = len(parameters["class_labels"])

log_of_posteriors = np.reshape(log_of_posteriors, (n, l))
pred_prob = np.array(pred_prob)
prediction_probs = (
    np.exp(log_of_posteriors - pred_prob[:, np.newaxis])
    / np.sum(np.exp(log_of_posteriors - pred_prob[:, np.newaxis]), axis=1)[
        :, np.newaxis
    ]
)


print("Here are 10 rows of results:")
for i, value in enumerate(gt):
    if i == 10:
        break
    print(f"\tGT={ parameters['class_labels'][value] }->", end="")
    for j in range(len(parameters["class_labels"])):
        print(
            f"\t{parameters['class_labels'][j]}: {prediction_probs[i, j]*100.0:.1f}%", end=""
        )
    print()


print("\n*** Analysis ***")
diff = np.array(gt) - np.array(prediction)
correct = np.sum(diff == 0)
accuracy = correct / n
print(f"{n} data points analyzed, {correct} correct ({accuracy * 100.0:.1f}% accuracy)")

# Confusion matrix
cm = confusion_matrix(np.array(gt), np.array(prediction))
print("Confusion:\n", cm)

labels = parameters["class_labels"]
priors = parameters["class_priors"]
best_label_idx = np.argmax(priors)

# Save out a confusion matrix plot
fig, ax = plt.subplots()
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
fig.savefig("confusion_gn.png")
print("Wrote confusion matrix matrix plot to confusion_gn.png")

print("\n*** Making a plot ****")

confidence = np.max(prediction_probs, axis=1)
steps = 32
thresholds = np.linspace(0.2, 1.0, steps)
correct_ratio = np.zeros(steps)
confident_ratio = np.zeros(steps)

for i in range(steps):
    threshold = thresholds[i]
    if np.sum(confidence > threshold) != 0:
        correct_ratio[i] = np.sum(
            (confidence >= threshold) & (np.array(gt) == prediction)
        ) / np.sum(confidence >= threshold)
    else:
        correct_ratio[i] = 1
    confident_ratio[i] = np.sum(confidence >= threshold) / n

fig, ax = plt.subplots()
ax.set_title("Confidence and Accuracy Are Correlated")
ax.set_xlabel("Confidence Threshold")
ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100.0:.0f}%")
ax.plot(
    thresholds, correct_ratio, "blue", linewidth=0.8, label="Accuracy Above Threshod"
)
ax.plot(
    thresholds,
    confident_ratio,
    "r",
    linestyle="dashed",
    linewidth=0.8,
    label="Test data scoring above threshold",
)
ax.hlines(
    priors[best_label_idx],
    0.2,
    1,
    "blue",
    linestyle="dashed",
    linewidth=0.8,
    label=f"Accuracy Guessing {labels[best_label_idx]}",
)
ax.legend()
fig.savefig("confidence_gn.png")

print(f'Saved to "{PLOT_GN}".')
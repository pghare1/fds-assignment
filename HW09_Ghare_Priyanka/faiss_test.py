import pickle as pkl
import faiss
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from time import perf_counter
from keras.datasets import mnist
import numpy as np
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} (exact | approximate)")
    exit(-1)

K = 3

# Read in the MNIST test data

d = 28 * 28
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape((-1, d))


# Read the values for the index from values.pkl

with open(f"exact.pkl", "rb") as f:
    y_train = pkl.load(f)

# Read the index from index.faiss



if sys.argv[1] == "exact":
    index_path = "exact.faiss"
    plot_path = "exact_confusion.png"
else:
    index_path = "approx.faiss"
    plot_path = "approx_confusion.png"
# Read in the index itself
index = faiss.read_index(index_path)
print(f"Read {index_path}: Index has {index.ntotal} data points.")

start = perf_counter()

# Do prediction (This is the hardest part)

X_test = X_test.astype(np.float32)
D, I = index.search(X_test, K)


print(f"Inference: elapsed time = {perf_counter() - start:.2f} seconds")

# Print the accuracy

y_pred = y_train[I]
y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, y_pred)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

print(f"Confusion: \n{cm}")

# Save it to a display
fig, ax = plt.subplots()
disp.plot(ax=ax)
fig.savefig(plot_path)
ax.set_title(f"MNIST with Faiss (k={K})")
print(f"Wrote {plot_path}")

import pickle as pkl
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from time import perf_counter
from keras.datasets import mnist

# Read in the MNIST data
d = 28 * 28
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape((-1, d))

# Read in the classifier
with open("knn_model.pkl", "rb") as f:
    classifier = pkl.load(f)

start = perf_counter()
y_pred = classifier.predict(X_test)
print(f"Testing: elapsed time = {perf_counter() - start:.2f} seconds")

# Make predictions for the test data
## Your code here
print(f"Inference: elapsed time = {perf_counter() - start:.2f} seconds")


# Show accuracy
print(f"Accuracy: {100*accuracy_score(y_test, y_pred):.1f}%")
## Your code here


# Create a =onfusion matrix
cm = confusion_matrix(y_test, y_pred)
## Your code here
print(cm)

# Make a display
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


fig, ax = plt.subplots()
ax.set_title("MNIST Confusion Matrix")
disp.plot(ax=ax)

## Your code here

fig.savefig("sk_confusion.png")
print("Wrote sk_confusion.png")

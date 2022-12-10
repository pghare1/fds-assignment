import pickle as pkl
import faiss
import numpy as np
from keras.datasets import mnist
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} (exact | approximate)")
    exit(-1)

# Read in the MNIST training data

d = 28 * 28
(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape((-1, d))

n = X_train.shape[0]


if sys.argv[1] == "exact":
    index = faiss.IndexFlatL2(d)
    print("Using IndexFlatL2 for true KNN")
    path_prefix = "exact"
else:
   
    quantizer = faiss.IndexHNSWFlat(d, 32)
    index = faiss.IndexIVFPQ(quantizer, d, 1024, 16, 8)
    print("Using HNSW/IVFPQ for approximate KNN")
    path_prefix = "approx"

# Train and add with the training data



X_train = X_train.astype(np.float32)
index.train(X_train)
index.add(X_train)


print(f"Index has {index.ntotal} data points.")

path = f"{path_prefix}.faiss"

faiss.write_index(index, path)
with open(f"{path_prefix}.pkl", "wb") as f:
    pkl.dump(y_train, f)


print(f"Wrote index to values.pkl and {path}")

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Read the training data
print("Reading input...")
## Your code here
df = pd.read_csv("train.csv")
print(df)
print("Scaling...")
## Your code here
# X = df.to_numpy();
X_withLast = df
print(f"Using {X_withLast.shape[1]} column")
X_Final = X_withLast.drop(["TenYearCHD"], axis=1)
X = X_Final.to_numpy()
# print(X)
print(f"Using {X.shape[1]} column")
Y = df["TenYearCHD"].to_numpy()
# print(Y)

print("Fitting...")
## Your code herer
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)
classifier = LogisticRegression(random_state=0)
classifier.fit(X, Y)
pred = classifier.predict(X)
print(pred)

# Get the accuracy on the training data
## Your code here
train_accuracy = accuracy_score(Y, pred)
print(f"Training accuracy = {train_accuracy}")
# print ("Accuracy : ", accuracy_score(Y, pred));
# Write out the scaler and logisticregression objects into a pickle file
pickle_path = "classifier.pkl"
print(f"Writing scaling and logistic regression model to {pickle_path}...")
## Your code here

with open(pickle_path, "wb") as f:
    pickle.dump((scaler, classifier), f)

# obj_dict = {1:scaler, 2:classifier};
# pickle_output = open(pickle_path, "wb");
# pickle.dump(obj_dict, pickle_output);
# pickle_output.close();

# pickle_in = open("classifier.pkl", "rb")
# print_dict = pickle.load(pickle_in);
# print(print_dict);
# print(print_dict[1]);
# print(print_dict[2]);

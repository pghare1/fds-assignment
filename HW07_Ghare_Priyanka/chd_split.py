import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("framingham.csv", index_col=False)
print(f"Read {df.shape[0]} rows")

# Drop rows with missing values
## Your code here
df = df.dropna()
print(f"Using {df.shape[0]} rows")

# Split into training and testing dataframes
## Your code here
train, test = train_test_split(df, test_size=0.2)
print(train)
print(test)
# Write out each as a CSV
## Your code here
train.to_csv("train.csv")
test.to_csv("test.csv")

import pandas as pd
from pandas_profiling import ProfileReport

# Read in the data
df = pd.read_csv("framingham.csv", index_col=None)
print(df.head())

# Create a report and save it to a file
## Your code here

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("data_report.html")

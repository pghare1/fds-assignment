import pandas as pd
from datetime import date
import sys

from sklearn.preprocessing import OrdinalEncoder

def series_report(
    series, is_ordinal=False, is_continuous=False, is_categorical=False
):
    print(f"{series.name}: {series.dtype}")
    if is_ordinal and is_continuous and not is_categorical:
        missingKey = series.isnull().sum()
        if missingKey > 0:
            print(f"   Missing in {missingKey} rows ({missingKey / 100}%)")
        print(f"   Range: {series.min()} - {series.max()}")
        print(f"   Mean: {series.mean():.2f}")
        if is_ordinal and is_continuous:
            print(f"   Standard Deviation: {series.std():.2f}")
        print(f"   Median: {series.median():.2f}") 
    if is_categorical:
        missingKey = series.isnull().sum()
        print(f"   Missing in {missingKey} rows ({missingKey / 100}%)")
        print(f"      {series.value_counts().to_string()}\t\t")
    if is_ordinal and series.dtype == object:
        print(f"   Range: {series.min()} - {series.max()}")
    elif is_ordinal and not is_continuous and not is_categorical :
        print(f"   Range: {series.min()} - {series.max()}")
    
# Check command line arguments
if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <input_file>")
    exit(1)

# Read in the data
df = pd.read_csv(
    sys.argv[1], index_col="employee_id"
)

# Convert strings to dates for dob and death
df['dob'] = df['dob'].apply(lambda x: date.fromisoformat(x))
df['death'] = df['death'].apply(lambda x: date.fromisoformat(x))

# Show the shape of the dataframe
(row_count, col_count) = df.shape
print(f"*** Basics ***")
print(f"Rows: {row_count:,}")
print(f"Columns: {col_count}")

# Do a report for each column
print(f"\n*** Columns ***")
series_report(df.index, is_ordinal=True)
series_report(df["gender"], is_categorical=True)
series_report(df["height"], is_ordinal=True, is_continuous=True)
series_report(df["waist"], is_ordinal=True, is_continuous=True)
series_report(df["salary"], is_ordinal=True, is_continuous=True)
series_report(df["dob"], is_ordinal=True)
series_report(df["death"], is_ordinal=True)


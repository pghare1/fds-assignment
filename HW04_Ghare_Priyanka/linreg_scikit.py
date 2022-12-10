import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import kstest, norm
import matplotlib.pyplot as plt
import sys
import util
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter


# Check command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# Read in the argument
infilename = sys.argv[1]

# Read the spreadsheet
X, Y, labels = util.read_excel_data(infilename)

n, d = X.shape
print(f"Read {n} rows, {d-1} features from '{infilename}'.")

# Don't need the intercept added -- X has column of 1s
lin_reg = LinearRegression(fit_intercept=False)

# Fit the model
lin_reg.fit(X, Y)

# Pretty print coefficients
print(util.format_prediction(lin_reg.coef_, labels))

# Get residual
## Your code here
residual = Y - lin_reg.predict(X)

# Make a histogram of the residual and save as "res_hist.png"
## Your code here

fig, hist = plt.subplots()
hist.hist(residual, bins=18)

hist.set_ylabel("Density")
hist.set_xlabel("Residual")
hist.set_title("Residual Histogram")


def yhistisFormat(x, pos):
    return "$%1.0fK" % (x * 1e-3)


formatYhistis = FuncFormatter(yhistisFormat)
hist.xaxis.set_major_formatter(formatYhistis)
plt.show()
fig.savefig("res_hist.png")


# hists[0,0].xhistis.set_major_formatter("${x:1.0f}")

# Do a Kolmogorov-Smirnov to see if the residual is normally
# distributed
## Your code here
ks_stat, p_value = kstest(residual, "norm")
print(f"KS-statistic: {ks_stat:.3f}, p-value: {p_value}")


# Calculate the standard deviation
## Your code here
standard_deviation = np.std(
    residual,
    ddof=d,
)

print(
    f"68% of predictions with this formula will be within ${standard_deviation:,.02f} of the actual price."
)
print(
    f"95% of predictions with this formula will be within ${2.0 * standard_deviation:,.02f} of the actual price."
)

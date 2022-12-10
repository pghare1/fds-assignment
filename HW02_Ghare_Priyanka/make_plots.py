from cgitb import text
from gettext import npgettext
from itertools import count
from math import fabs
from msilib import sequence
from tkinter import Canvas, Grid
from turtle import color, width, xcor
from typing import Sequence
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import seaborn as sb
from sklearn.linear_model import LinearRegression

# Read in bikes.csv into a pandas dataframe
### Your code here
bikeFile = pd.read_csv("bikes.csv")

# Read in DOX.csv into a pandas dataframe
# Be sure to parse the 'Date' column as a datetime
### Your code here
doxFile = pd.read_csv("DOX.csv")

# Divide the figure into six subplots
# Divide the figure into subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
fig.canvas.draw()
# Make a pie chart
### Your code here

df_pie = (
    bikeFile["status"]
    .value_counts()
    .rename_axis("Status_name")
    .reset_index(name="counts")
)
# print(df_pie.to_string())

my_labels = df_pie.Status_name
my_values = df_pie.counts

pie_chart = axs[0, 0]
pie_chart.set_title("Current Status")

patch, text, pct = pie_chart.pie(my_values, labels=my_labels, autopct="%1.2f%%")
plt.setp(pct, color="white")
for i, pt in enumerate(patch):
    text[i].set_color(pt.get_facecolor())
    text[i].set_size(12)

# Make a histogram with quartile lines
# There should be 20 bins

### Your code here
histogram_bikes_area = axs[0, 1]
histogram_bikes_area.hist(
    bikeFile["purchase_price"],
    bins=20,
    histtype="stepfilled",
    edgecolor="lightblue",
    alpha=1,
    color="#fff",
)

min = bikeFile["purchase_price"].min()
print(min)

histogram_bikes_area.axvline(min, color="black", linestyle="dashed", linewidth=1)
histogram_bikes_area.text(min + 8, 15, "min: $" + str(round(min)), rotation=90)


max = bikeFile["purchase_price"].max()
print(max)
histogram_bikes_area.axvline(max, color="black", linestyle="dashed", linewidth=1)
histogram_bikes_area.text(max + 8, 15, "max: $ {:,.0f}".format(round(max)), rotation=90)

sum = bikeFile["purchase_price"].count()
quartiles = bikeFile["purchase_price"].quantile([0.25, 0.5, 0.75])

percent25 = np.percentile(bikeFile["purchase_price"], 25)
percent50 = np.percentile(bikeFile["purchase_price"], 50)
percent75 = np.percentile(bikeFile["purchase_price"], 75)
histogram_bikes_area.text(
    percent25 + 8, 15, "25%: $" + str(round(percent25)), rotation=90
)
histogram_bikes_area.axvline(percent25, color="black", linestyle="dashed", linewidth=1)
histogram_bikes_area.text(
    percent50 + 8, 15, "50%: $" + str(round(percent50)), rotation=90
)
histogram_bikes_area.axvline(percent50, color="black", linestyle="dashed", linewidth=1)
histogram_bikes_area.text(
    percent75 + 8, 15, "75%: $" + str(round(percent75)), rotation=90
)
histogram_bikes_area.axvline(percent75, color="black", linestyle="dashed", linewidth=1)


histogram_bikes_area.set_title("Price Histogram (1000 bikes)")
histogram_bikes_area.set_xlabel("US Dollars")
histogram_bikes_area.set_ylabel("Number of Bikes")

histogram_bikes_area.xaxis.set_major_formatter("${x:.0f}")


# Make a scatter plot with a trend line
### Your code here
scatter_dotted_area = axs[1, 0]
price = bikeFile["purchase_price"].tolist()
# print(price)
weight = bikeFile["weight"].tolist()
# print(weight)
scatter_dotted_area.scatter(price, weight, s=1)
scatter_dotted_area.set_xlabel("Price")
scatter_dotted_area.set_ylabel("Weight")
scatter_dotted_area.set_title("Price vs Weight")
scatter_dotted_area.xaxis.set_major_formatter("${x:.0f}")
scatter_dotted_area.yaxis.set_major_formatter("{x:.0f}Kg")
x = np.array(price)
y = np.array(weight)
m, b = np.polyfit(x, y, 1)
scatter_dotted_area.plot(x, m * x + b, color="red")

# time series
scatter_timeline_area = axs[1, 1]
scatter_timeline_area.set_title("DOX")
dates = pd.to_datetime(doxFile["Date"])

scatter_timeline_area.plot(dates, doxFile["Adj Close"])
# dates = [dates.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]

scatter_timeline_area.yaxis.set_major_formatter("${x}")
myFmt = mdates.DateFormatter("%m\%d")
axs[1, 1].xaxis.grid(True)
axs[1, 1].yaxis.grid(True)
scatter_timeline_area.xaxis.set_major_formatter(myFmt)

# Make a boxplot sorted so mean values are increasing
# Hide outliers

box_brand_price_compare = axs[2, 0]
sortedBoxData = bikeFile.sort_values(by=["purchase_price", "brand"])[
    ["purchase_price", "brand"]
]

data_to_plot = [sortedBoxData]

# sb.boxplot(  y="purchase_price", x= "brand", data=sortedBoxData, orient='v')
sb.boxplot(
    data=sortedBoxData,
    x="brand",
    y="purchase_price",
    order=["Giant", "GT", "Canyon", "Trek", "BMC", "Cdale"],
    palette="Blues",
    color="white",
    fliersize=5,
    linewidth=1,
    ax=axs[2, 0],
)
axs[2, 0].yaxis.grid(True)
axs[2, 0].xaxis.grid(True)
box_brand_price_compare.set_title("Brand vs. Price")
box_brand_price_compare.set_xlabel("")
box_brand_price_compare.set_ylabel("")
box_brand_price_compare.yaxis.set_major_formatter("${x}")
### Your code here

# Make a violin plot
### Your code here
violin_brand_price_compare = axs[2, 1]
sortedBoxData = bikeFile.sort_values(by=["purchase_price", "brand"])[
    ["purchase_price", "brand"]
]

data_to_plot = [sortedBoxData]

# sb.boxplot(  y="purchase_price", x= "brand", data=sortedBoxData, orient='v')
sb.violinplot(
    data=sortedBoxData,
    x="brand",
    y="purchase_price",
    order=["Giant", "GT", "Canyon", "Trek", "BMC", "Cdale"],
    palette="Blues",
    color="white",
    ax=axs[2, 1],
)
violin_brand_price_compare.set_title("Brand vs. Price")
violin_brand_price_compare.set_xlabel("")
violin_brand_price_compare.set_ylabel("")
violin_brand_price_compare.yaxis.set_major_formatter("${x}")
# Create some space between subplots
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Write out the plots as an image
plt.savefig("plots.png")

plt.show()

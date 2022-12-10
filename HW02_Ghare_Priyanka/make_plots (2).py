import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression

# Read in the employee data
df = pd.read_csv('bikes.csv')

# Read in stock price data
price_df = pd.read_csv('DOX.csv', parse_dates=['Date'])

# Divide the figure into subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Status pie chart
axs[0,0].set_title('Current Status')
vcounts = df['status'].value_counts()
patches, texts, pcts = axs[0,0].pie(vcounts, labels=vcounts.index, autopct='%1.1f%%', textprops=dict(color="w"))
for i, patch in enumerate(patches):
    texts[i].set_color(patch.get_facecolor())


# Histogram of purchase price
axs[0,1].set_title(f'Price Histogram ({len(df)} bikes)')
axs[0,1].set_xlabel('US Dollars')
axs[0,1].set_ylabel('Number of Bikes')
axs[0,1].hist(df['purchase_price'], histtype='step', bins=20, label=None)
axs[0,1].xaxis.set_major_formatter(lambda x, pos: f"${x:.0f}")

# Add meaningful vertical lines
price_stats = df['purchase_price'].describe()
ignored_cols = ['std', 'mean','count']
for stat in price_stats.index:
    if stat in ignored_cols:
        continue
    value = price_stats[stat]
    label = f"{stat}: ${value:,.0f}"
    axs[0,1].axvline(value, color='k', linestyle='dashed', label=stat)
    axs[0,1].text(value + 10, 10, label, rotation=90, va='bottom')

# Scatter plot of price vs weight
axs[1,0].set_title('Price vs. Weight')
axs[1,0].set_xlabel('Price')
axs[1,0].set_ylabel('Weight')
axs[1,0].xaxis.set_major_formatter(lambda x, pos: f"${x:.0f}")
axs[1,0].yaxis.set_major_formatter(lambda x, pos: f"{x:.0f} kg")

axs[1,0].scatter(df['purchase_price'], df['weight'], 5, marker="o", alpha=0.5, c='#005555', linewidths=0)

# Do linear regression for trendline
# Get data as numpy arrays
X =  df['purchase_price'].values.reshape(-1, 1)
y = df['weight'].values.reshape(-1, 1)

# Do linear regression
reg = LinearRegression()
reg.fit(X, y) 

# Get the parameters
slope = reg.coef_[0]
intercept = reg.intercept_
print(f"Slope: {slope}, Intercept: {intercept}")

# Add regression line
xends = [X.min(), X.max()]
yends = [slope * x + intercept for x in xends]
axs[1,0].plot(xends, yends, 'r')

# Plot the stock price vs date
axs[1,1].set_title('DOX')
axs[1,1].plot(price_df['Date'], price_df['Close'])
axs[1,1].yaxis.set_major_formatter(lambda x, pos: f"${x:.2f}")
hfmt = matplotlib.dates.DateFormatter('%m/%d')
axs[1,1].xaxis.set_major_formatter(hfmt)
axs[1,1].grid(color='0.5', linewidth=0.5)

# Box plot
axs[2,0].set_title('Brand vs. Price')
gb = df.groupby('brand')
g_df = pd.DataFrame({col: val['purchase_price'] for col, val in gb})
g_medians = g_df.median()
g_medians.sort_values(inplace=True)
g_df = g_df[g_medians.index]
g_df.boxplot(ax=axs[2,0], sym='')
axs[2,0].yaxis.set_major_formatter(lambda x, pos: f"${x:.0f}")

# Violin pot
axs[2,1].set_title('Brand vs. Price')
all_brands = g_df.columns.tolist()
ticks = [x + 1 for x in range(len(all_brands))]
brand_prices = []
for brand in all_brands:
    prices = df[df['brand'] == brand]['purchase_price']
    brand_prices.append(prices)
axs[2, 1].violinplot(brand_prices, widths=0.7, showmeans=True,
                     showextrema=True, showmedians=True, bw_method=0.5)
axs[2,1].yaxis.set_major_formatter(lambda x, pos: f"${x:.0f}")
axs[2,1].xaxis.set_tick_params(direction='out')
axs[2,1].xaxis.set_ticks_position('bottom')
axs[2,1].set_xticks(ticks)
axs[2,1].set_xticklabels(all_brands)
axs[2,1].set_xlim(0.25, len(all_brands) + 0.75)

# Create some space between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

# Write out the plots as an image
plt.savefig('plots.png')

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
from operator import le

# Deal with command-line
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv>")
    sys.exit(1)
infilename = sys.argv[1]

# Read in the basic data frame
df =  pd.read_csv(infilename, index_col='property_id')

X_basic = df.values[:, :-1]
labels_basic = df.columns[:-1]
Y = df.values[:, -1]
#df.head()


# Expand to a 2-degree polynynomials
## Your code here
polyn = PolynomialFeatures(2)
Xpoly = polyn.fit_transform(X_basic)
labels_polyn = polyn.get_feature_names(labels_basic)
#print(labels_polyn)
# Prepare for loop
residual = Y
feature_indices = ["1"]
reg = LinearRegression(fit_intercept=False)
X_list_updated = np.ones([len(Xpoly),1])

while len(feature_indices) < 3:
    
    rows,cols = Xpoly.shape
    input_list = []
   
    for i in range(1, cols):
        coeff, pval = pearsonr(Xpoly[:, i], residual)
        input_list.append({'feature':labels_polyn[i],'p_value':pval})

    #input_list
    # Sort by p-value
    input_list.sort(key=lambda x: x['p_value'])
    #input_list = pd.DataFrame(input_list, index=labels_polyn[1:])
    #print(input_list)
    #input_list = input_list.sort_values('p_value')
    #input_list

    # Print First time through: using original price data as the residual
    print("First time through: using original price data as the residual")
    # Print out all the p-values
    for i in range(len(input_list)):
        print(f'\"{input_list[i]["feature"]}\" vs residual: p-value={input_list[i]["p_value"]}')

    #Fitting with ["1" "sqft_hvac" ]
    #print(X_list_updated)
    feature_indices.append( input_list[0]['feature'])
    #f=feature[0]
    for i in range(len(input_list)):
        if labels_polyn[i]==input_list[0]['feature'] :
            index = i
            break
    print('**** Fitting with ', feature_indices, '****')

    X_list_updated= np.append(X_list_updated, Xpoly[:,index].reshape(-1,1), axis=1)
    #print(X_list_updated)

    # calculate the r2
    reg.fit(X_list_updated, Y)
    r2 =reg.score(X_list_updated, Y)
    print("R2 =", r2)
    y_new=reg.predict(X_list_updated)
    residual=Y-y_new
   ## print(residual)
   
    print("Residual is updated")
   

# Print out "sqft_hvac" r2
#print(feature_indices)
# We always need the column of zeros to be the first column
# include the intercept
#print(residual)

    ## Your code here

# Any relationship between the final residual and the unused variables?
print("Making scatter plot: age_of_roof vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:,3], residual, marker='+')
fig.savefig("ResidualRoof.png")

print("Making a scatter plot: miles_from_school vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:,4], residual, marker='+')
fig.savefig("ResidualMiles.png")


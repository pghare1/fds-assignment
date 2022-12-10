import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import warnings
warnings.filterwarnings('ignore')


data =  pd.read_csv('data.csv', index_col='property_id')

data=pd.DataFrame(data)
data_old=pd.DataFrame(data)
#data.head()

data["lot_size"] = data["lot_width"] * data["lot_depth"]

data["is_close_to_school"] = 0


for i in range(len(data)):
  if data["miles_to_school"].iloc[i]<2.0:
    data["is_close_to_school"].iloc[i]=1
  else:
    data["is_close_to_school"].iloc[i]=0


ones=np.ones((len(data),1))
#print(ones)
data_new=data.drop(columns=["age_of_roof","price","lot_width","lot_depth","miles_to_school"],inplace=True)


#np.append(ones, data.values["sqft_hvac"], axis=1)

X_value = data.values[:, :]
lab_values = data.columns[:]
Y = data_old.values[:,5]
#print(X_value)
#print(lab_values)
print("Making new features.....")
print("Using only the useful ones: ['sqft_hvac', 'lot_size', 'is_close_to_school']...")

residual = Y
feature_indices = ["1"]
reg = LinearRegression(fit_intercept=False)
X_updated_values = np.ones([len(X_value),1])

X_updated_values= np.append(X_updated_values, X_value, axis=1)
    #print(X_updated_values)

    # calculate the R2
reg.fit(X_updated_values, Y)
R2 =reg.score(X_updated_values, Y)
print("R2 = " , R2)
y_updated = reg.predict(X_updated_values)
residual=Y-y_updated
pred=reg.coef_
#print(residual)

#print(f"R2 ={R2.round(6)}")
print("****Prediction****")

print(
    f"Price = ${pred[0].round(2)} + $({lab_values[0]} X {pred[1].round(2)})+ $({lab_values[1]} X {pred[2].round(2)}) \n Less than 2 miles to school? you get ${pred[3].round(2)} added to the price"
)





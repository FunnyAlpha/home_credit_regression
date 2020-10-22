#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
import os

PATH = "D:\projects\home_credit_regression\data"
# print(os.listdir(PATH))

# upload data
app_train = pd.read_csv("./data/application_train.csv")
app_test = pd.read_csv("./data/application_test.csv")

#%%
pd.set_option('display.max_columns', None)
# print(app_train.TARGET.value_counts())
# print(app_train.head())

#%%
#plot

plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = [8,5]

plt.hist(app_train.TARGET)

plt.show()
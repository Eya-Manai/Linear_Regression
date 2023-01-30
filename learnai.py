import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
dataset=pd.read_csv('C:/Users/USER/Downloads/weather.csv')
print(dataset.shape)
print(dataset.describe())
dataset.plot(x='MinTemp' , y='MaxTemp',style='o')
plt.title('Mintemps vs max temps ')
plt.xlabel('MinTemps')
plt.ylabel('MaxTemps')
#plt.show()
plt.figure(figsize=(15,10))
plt.tight_layout()
sea.histplot(dataset['MaxTemp'],kde=True)
plt.show()




 


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
#plt.show()
#dataspicing
x=dataset['MinTemp'].values.reshape(-1,1)
y=dataset['MaxTemp'].values.reshape(-1,1)
x_train,x_test ,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
regressor=LinearRegression() 
regressor.fit(x_train,y_train)
print('intercept:',regressor.intercept_ )
print('coefficient ',regressor.coef_)
#print(y_test)
#print(y_test.flatten())
y_pred=regressor.predict(x_test)
arrayA=np.concatenate((y_test,y_pred),axis=1)
dataframe=pd.DataFrame(arrayA,columns=['Actuel','Predected'])
print(dataframe)
dataframe.plot(kind='bar',figsize=(16,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='minor',linestyle=':' , linewidth='0.5',color='black')
plt.show()
plt.scatter(x_test,y_test,color='gray')
plt.plot(x_test,y_pred,color='red',linewidth=2)
plt.show()
print ('Mean absolute error ', metrics.mean_absolute_error(y_test,y_pred))
print ('Mean squared error ', metrics.mean_squared_error(y_test,y_pred))
print ('Root Mean squared error ', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))









 










 




 


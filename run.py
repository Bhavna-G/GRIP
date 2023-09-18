import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model
df=pd.read_csv(r'C:\Users\HP\Downloads\task1.txt')
df
df.shape
df.columns
df.info()
df.describe()
df.groupby (['Hours'])['Scores'].mean()
plt.scatter(df['Hours'],df['Scores'],color='blue')
plt.title("Hours vs Scores")
plt.xlabel("Hours studied")
plt.ylabel("percentage scored")
plt.show()
df.corr()
sns.lmplot(x="Hours", y="Scores", data=df)
plt.title("plotting the regression line")
#sns.regplot(x="Hours",y="Scores",data=df)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
df1=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
df1
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('(training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('(testing set)')
plt.xlabel('Hours studied')
plt.ylabel('percentage scored')
plt.show()
Hours=np.array([[9.25]])
predict= regressor.predict(Hours)
print("No. of Hours={}".format(Hours))
print("predicted score={}".format(predict[0]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model
df=pd.read_csv(r'C:\Users\HP\Downloads\task1.txt')
df
    Hours  Scores
0     2.5      21
1     5.1      47
2     3.2      27
3     8.5      75
4     3.5      30
5     1.5      20
6     9.2      88
7     5.5      60
8     8.3      81
9     2.7      25
10    7.7      85
11    5.9      62
12    4.5      41
13    3.3      42
14    1.1      17
15    8.9      95
16    2.5      30
17    1.9      24
18    6.1      67
19    7.4      69
20    2.7      30
21    4.8      54
22    3.8      35
23    6.9      76
24    7.8      86
df.shape
(25, 2)
df.columns
Index(['Hours', 'Scores'], dtype='object')
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 532.0 bytes
df.describe()
           Hours     Scores
count  25.000000  25.000000
mean    5.012000  51.480000
std     2.525094  25.286887
min     1.100000  17.000000
25%     2.700000  30.000000
50%     4.800000  47.000000
75%     7.400000  75.000000
max     9.200000  95.000000
df.groupby (['Hours'])['Scores'].mean()
Hours
1.1    17.0
1.5    20.0
1.9    24.0
2.5    25.5
2.7    27.5
3.2    27.0
3.3    42.0
3.5    30.0
3.8    35.0
4.5    41.0
4.8    54.0
5.1    47.0
5.5    60.0
5.9    62.0
6.1    67.0
6.9    76.0
7.4    69.0
7.7    85.0
7.8    86.0
8.3    81.0
8.5    75.0
8.9    95.0
9.2    88.0
Name: Scores, dtype: float64
plt.scatter(df['Hours'],df['Scores'],color='blue')
plt.title("Hours vs Scores")
plt.xlabel("Hours studied")
plt.ylabel("percentage scored")
plt.show()
figure 1
df.corr()
           Hours    Scores
Hours   1.000000  0.976191
Scores  0.976191  1.000000
sns.lmplot(x="Hours", y="Scores", data=df)
plt.title("plotting the regression line")
#sns.regplot(x="Hours",y="Scores",data=df)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x
array([[2.5],
       [5.1],
       [3.2],
       [8.5],
       [3.5],
       [1.5],
       [9.2],
       [5.5],
       [8.3],
       [2.7],
       [7.7],
       [5.9],
       [4.5],
       [3.3],
       [1.1],
       [8.9],
       [2.5],
       [1.9],
       [6.1],
       [7.4],
       [2.7],
       [4.8],
       [3.8],
       [6.9],
       [7.8]])

y
array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30,
       24, 67, 69, 30, 54, 35, 76, 86], dtype=int64)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
array([17.04289179, 33.51695377, 74.21757747, 26.73351648, 59.68164043,
       39.33132858, 20.91914167, 78.09382734, 69.37226512])

df1=pd.DataFrame({'Actual':y_test,'predicted':y_pred})
df1
   Actual  predicted
0      20  17.042892
1      27  33.516954
2      69  74.217577
3      30  26.733516
4      62  59.681640
5      35  39.331329
6      24  20.919142
7      86  78.093827
8      76  69.372265
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('(training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
figure2

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('(testing set)')
plt.xlabel('Hours studied')
plt.ylabel('percentage scored')
plt.show()
figure 3
Hours=np.array([[9.25]])
predict= regressor.predict(Hours)
print("No. of Hours={}".format(Hours))
No. of Hours=[[9.25]]
print("predicted score={}".format(predict[0]))
predicted score=92.14523314523316

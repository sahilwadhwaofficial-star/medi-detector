#  Project : Medi-Detector 
# Made By : Sahil Wadhwa 

# Step1: First we will Import Libraries 

from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df=pd.read_csv('Drug200.csv')
df = df.head(201)
print(df)

df.info()

df.describe()

# Step 2 : For Gathering Information we will plot diffrent types of graph
#Initializing the Scatter Plot
def Scatterplot(df):
  x = df.Age
  y = df.Drug
  plt.figure(figsize=(10,10))
  
  plt.scatter(x,y)
  plt.xticks(np.arange(0, 110, 10))
  plt.xticks (rotation=35)
  plt.xticks (fontsize = 10)
  plt.yticks (fontsize = 24)
 
  plt.xlabel ("Age", fontsize = 24)
  plt.ylabel ("Medicine", fontsize = 24)
  plt.title ("Which Medicine is Suitable According to Age",fontsize = 24)
  plt.xticks (rotation=30, horizontalalignment='right')
  a = plt.show() 
  return a
  
#Printing Scatter Plot
print(Scatterplot(df))

#Initializing the Histogram
def Histogram(df):
  x = df.Drug
  y = df.Gender
  plt.figure(figsize=(10,10))
  plt.xticks (fontsize = 24)
  plt.yticks (fontsize = 24)
  plt.hist(y,bins=10)
  
  plt.ylabel("No of people",fontsize = 24)
  plt.xlabel("Gender",fontsize = 24)
  plt.title('Which Gender has More intake of Medicine',fontsize = 24)
  
  o = plt.show()
  return o
  
#Printing Histogram
print(Histogram(df))

# Step 3 : Now we will do Label encoding 

#Label Encoding of Medicine
print(df['Drug'].value_counts())
label_encode7 = {"Drug" : {'DrugY':0, 'drugC':1, 'drugX':2, 'drugA':3, 'drugB':4}}
df.replace(label_encode7,inplace=True)

#Label Encoding of Gender
print(df['Gender'].value_counts())
label_encode = {"Gender": {'F':1, 'M':0}}
df.replace(label_encode,inplace=True)

#Label Enccoding of Cholesterol
print(df['Cholesterol'].value_counts())
label_encode3 = {"Cholesterol": {'HIGH':1, 'NORMAL':0}}
df.replace(label_encode3,inplace=True)

#label Encoding of Blood Pressure (BP)
print(df['BP'].value_counts())
label_encode2 = {"BP": {'HIGH':1,'LOW':2,'NORMAL':0}}
df.replace(label_encode2,inplace=True)

#preprocessing before Standardising process
df.Drug.unique()
print(type(df['Age']))
df['Age'] = df['Age'].astype(float) 
y_values = df["Drug"]

# Step 4 : Now we will Standardise  
x_values = df[['Age','Gender','BP','Cholesterol']]
print(x_values.head())
standardise = StandardScaler() 
x_values = standardise.fit_transform(x_values)
x_values_df = pd.DataFrame(x_values)

#Now we'll split data
x_train, x_test, y_train, y_test = train_test_split(x_values,y_values,test_size=0.3,random_state=10)

# Step 5 : So lets make a Decision tree 
print("Initialising the Decision Tree")
dt = tree.DecisionTreeClassifier()
x = df[['Age','Gender','BP','Cholesterol']]
y = df['Drug']

print(x.head())
print(y.head())

dt = dt.fit(x.values,y)
y_predict = dt.predict(x_test)

#Printing the info
r = df.info()
print(r)

#Calculating and Printing the Accuracy of this AI Model
t = dt.score(x.values,y)
print("Accuracy = ",t*100,"%")

i = x.info()
print(i)

p = x.describe()
print(p)

#Intialising the Confusion Matrix
from sklearn.metrics import confusion_matrix  
print(confusion_matrix(y_test, y_predict))

y_predict = dt.predict(x_test)

#Taking input from User related his/her Age,Gender,BP Cholesterol
while True:
  st = int(input("0.Exit,1.To Continue : "))
  if st == 0:
    break
  else:
    test = pd.DataFrame()
    test["Age"]=[float(input("Enter Patient's Age : "))]
    test["Gender"]=[int(input("Enter Patient's Gender {0} for Male and {1} for Female : "))]
    test["BP"]=[int(input("Enter Patient's Blood Pressure Level {0} for Low Level ,{1} for Normal level & {2} for High Level : "))]
    test["Cholesterol"]=[int(input("Enter Patient's Cholesterol Level {0} for High Level & {1} for Normal Level : "))]
    test = standardise.transform(test)
    test = pd.DataFrame(test)
    print(test)

    #Fiting the Calculated Data & Predicting the Medicine
    dt.fit(x_values,y_values)
    prediction=dt.predict(test)
    if prediction == [0]:
        a = "Use Medicine A "
    elif prediction == [1]:
        a = "Use Medicine Y"
    elif prediction == [2]:
        a = "Use Medicine B"
    elif prediction == [3]:
        a = "Use Medicine C"
    else:
        a = "Use Medicine X"    
    print(prediction)
    print(a)

# Medicine are Suggested according to health of specific person

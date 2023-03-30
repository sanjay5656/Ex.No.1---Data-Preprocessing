# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:

#Developed by : SANJAY S
#Reg num      : 212221243002

import pandas as pd
import numpy as np

df = pd.read_csv("Churn_Modelling.csv")
df

df.isnull().sum()

df.duplicated()

df.describe()

df['Exited'].describe()

""" Normalize the data - There are range of values in different columns of x are different. 

To get a correct ne plot the data of x between 0 and 1 

LabelEncoder can be used to normalize labels.
It can also be used to transform non-numerical labels to numerical labels.
"""
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()

df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])

'''
MinMaxScaler - Transform features by scaling each feature to a given range. 
When we normalize the dataset it brings the value of all the features between 0 and 1 so that all the columns are in the same range, and thus there is no dominant feature.'''

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))

df1

df1.describe()

# Since values like Row Number, Customer Id and surname  doesn't affect the output y(Exited).
# So those are not considered in the x values
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)

y = df1.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print("Size of X_train: ",len(X_train))

print(X_test)
print("Size of X_test: ",len(X_test))

X_train.shape

## OUTPUT:
##Dataset:

![image](https://user-images.githubusercontent.com/115128955/228898426-94d842d4-9cdc-490a-a665-8c32639b83a6.png)

##Checking for null values:

![image](https://user-images.githubusercontent.com/115128955/228898543-8439bdf8-6aa4-40ff-a7f4-2c97f52b457f.png)

##Checking for duplicate values:

![image](https://user-images.githubusercontent.com/115128955/228898623-68a8c6cb-11fc-48e3-a71d-fbb24c205b5f.png)

##Describing Data:

![image](https://user-images.githubusercontent.com/115128955/228898718-5578abbe-5c84-45f0-b51b-e895885920d5.png)

##Checking for outliers in Exited Column:

![image](https://user-images.githubusercontent.com/115128955/228898782-aa6d47aa-ce64-48cf-a406-7736da4de82e.png)

##Normalized Dataset:

![image](https://user-images.githubusercontent.com/115128955/228899128-079e03dd-291f-473d-83e8-c62a944cd7a4.png)

##Describing Normalized Data:

![image](https://user-images.githubusercontent.com/115128955/228898955-d4fc98cc-b1d7-49a9-9aa2-d3bd1e7f7a19.png)

##X - Values:

![image](https://user-images.githubusercontent.com/115128955/228899280-dab6a778-6470-4f38-bda4-fe45c34d773c.png)

##Y - Value:

![image](https://user-images.githubusercontent.com/115128955/228899231-06b770e7-b568-4da0-a30e-f1c082e803f0.png)

##X_train values:

![image](https://user-images.githubusercontent.com/115128955/228899390-ad199c73-0836-47f7-b966-0edca49deb43.png)

##X_train Size:

![image](https://user-images.githubusercontent.com/115128955/228899448-37d2004f-2b2b-41d5-93b0-bc1fbc629f73.png)

##X_test values:

![image](https://user-images.githubusercontent.com/115128955/228899670-974df6a0-a03e-495e-97c4-d08934b7798b.png)

##X_test Size:

![image](https://user-images.githubusercontent.com/115128955/228899728-81765aca-8bb0-4e40-adb2-285486fab4ef.png)

##X_train shape:

![image](https://user-images.githubusercontent.com/115128955/228899855-5b1b4103-c23e-4afd-bd61-da85eddc5cf0.png)


## RESULT
Data preprocessing is performed in a data set downloaded from Kaggle

# -*- coding: utf-8 -*-
"""
Created on Mon May 16 18:23:45 2022

@author: user
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#deal with training data
data = pd.read_csv("train.csv")

#fill Pclass, fare with median
age_median = data['Pclass'].dropna().median() 
data['Pclass'].fillna(age_median, inplace = True)
Fare_median = data['Fare'].dropna().median() 
data['Fare'].fillna(Fare_median, inplace = True)

#make alphabetic parameters into numbers
dict_Embarked = {'S': 1, 'Q': 2, 'C': 3}
data['Embarked'] = data['Embarked'].map(dict_Embarked).astype(int)

#make a new sp feature combining 3 features
data.loc[(data.Sex=='female')&(data.Pclass==1),'sp']=6
data.loc[(data.Sex=='female')&(data.Pclass==2),'sp']=5
data.loc[(data.Sex=='female')&(data.Pclass==3),'sp']=4
data.loc[(data.Sex=='male')&(data.Pclass==1),'sp']=3
data.loc[(data.Sex=='male')&(data.Pclass==2),'sp']=2
data.loc[(data.Sex=='male')&(data.Pclass==3),'sp']=1

#put the same cabin into groups
data['Cabin']=data['Cabin'].apply( lambda x:str(x)[0] if not pd.isnull(x) else '1' )
data['Cabin'] = data.Cabin.replace( ['D', 'C', 'B', 'A','E', 'G','F', 'T'],'2')
print(data['Cabin'].unique())

data[ 'cp' ] = np.nan
data.loc[(data.Cabin=='1')&(data.Embarked==1),'cp']=6
data.loc[(data.Cabin=='1')&(data.Embarked==2),'cp']=5
data.loc[(data.Cabin=='1')&(data.Embarked==3),'cp']=4
data.loc[(data.Cabin=='2')&(data.Embarked==1),'cp']=3
data.loc[(data.Cabin=='2')&(data.Embarked==2),'cp']=2
data.loc[(data.Cabin=='2')&(data.Embarked==3),'cp']=1

#trying to fill out the age with names and titles
import re
regex = re. compile( ' ([A-Za-z]+)\.')
data['Title'] = data.Name.map( lambda x:regex.search(x)[0])
# Dropping the first and the Last words
data['Title'] = data.Title.map( lambda x:x[1:][:-1] )
data['Title'].unique()
        
data['Title'].count()
data['Title'] = data.Title.replace( ['Don', 'Rev', 'Dr', 'Major','Lady', 'Sir','Capt', 'Countess','Jonkheer','Dona','Col' ],'Rare')
data['Title'] = data.Title.replace( ['Ms','Mlle'],'Miss')
data['Title']= data.Title.replace(['Mme','Lady'],"Mrs")
data['Title']=data['Title'].map({'Mr':0,'Rare':1,'Master':2,'Miss':3,'Mrs':4})

Age_Mean = data[['Title','Age']].groupby( by=['Title' ] ).mean()
Age_Median = data[['Title','Age' ]].groupby( by=['Title'] ).median()
Age_Mean.columns = ['Age Mean' ]
Age_Median.columns = ['Age Median' ]
Age_Mean.reset_index( inplace=True )
Age_Median.reset_index( inplace=True )

tt=data.groupby('Title')['Age'].median()
#print(tt)
t=data.groupby('Title')['Age'].median().values
data['agee']=data['Age']
for i in range(0,5):
    data.loc[(data.agee.isnull()) & (data.Title==i),'agee']=t[i]
data['agee']=data['agee'].astype('int')
#print(data.isnull().sum())


#create family size for every person
data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
Survival_Rate = data[['Family_Size','Survived']].groupby(by=['Family_Size']).agg(np.mean)*100
Survival_Rate.columns = ['Survival Rate(%)']
Survival_Rate.reset_index()

#divide into sections
data[ 'Family_Class' ] = np.nan
data.loc[ data.Family_Size==0, 'Family_Class' ] = 2
data.loc[ (data.Family_Size==1), 'Family_Class' ] = 3
data.loc[ (data.Family_Size>=2) & (data.Family_Size<=4), 'Family_Class' ] = 2
data.loc[ (data.Family_Size>=5), 'Family_Class' ] = 1

#let sfare be sqrt of fare and fill nan with 0
data['sfare'] = np.sqrt((data['Fare']))
#data['log_fare'] = np.log10(data['Fare'])
data['sfare'] = data['sfare'].fillna(0).astype(int)

#print(data.sfare.quantile([0.25,0.5,0.75]))
data[ 'fare' ] = np.nan
data.loc[ (data.sfare<=2), 'fare' ] = 4
data.loc[ (data.sfare==3), 'fare' ] = 3
data.loc[ (data.sfare>=4) & (data.sfare<=5), 'fare' ] = 2
data.loc[ (data.sfare>=6), 'fare' ] = 1

#print(data.Age.quantile([0.25,0.5,0.75]))
data[ 'aged' ] = np.nan
data.loc[ (data.agee<18), 'aged' ] = 2
data.loc[ (data.agee>=18), 'aged' ] = 1

#drop data that are not needed
data.drop( 'Name', axis=1, inplace=True )
data.drop( 'Pclass', axis=1, inplace=True )
data.drop( 'Embarked', axis=1, inplace=True )
data.drop( 'SibSp', axis=1, inplace=True )
data.drop( 'Parch', axis=1, inplace=True )
data.drop( 'Ticket', axis=1, inplace=True )
data.drop( 'Fare', axis=1, inplace=True )
data.drop( 'fare', axis=1, inplace=True )
data.drop( 'Sex', axis=1, inplace=True )
data.drop( 'Age', axis=1, inplace=True )
data.drop( 'aged', axis=1, inplace=True )
data.drop( 'Family_Class', axis=1, inplace=True )


# Function to print plots
selected_cols = ['sp']
plt.figure( figsize=(10,len(selected_cols)*5) )
gs = gridspec.GridSpec(len(selected_cols),1)    
for i, col in enumerate(data[selected_cols] ) :        
    ax = plt.subplot( gs[i] )
    sns.countplot( data[col], hue=data.Survived, palette=['r','g'] )
    ax.set_yticklabels([])
    ax.set_ylabel( 'Number of People' )
plt.show()

#%%
#split training data into train and valid to calculate loss
train = data.sample(frac=0.6, random_state=25)
valid = data.drop(train.index)
train.reset_index(inplace=True,drop=True)
valid.reset_index(inplace=True,drop=True)

X1 = list(train['sp'])
X2 = list(train['agee'])
X3 = list(train['sfare'])
X4 = list(train['Family_Size'])
X5 = list(train['sp'])
Y = list(train['Survived'])

x1 = list(valid['sp'])
x2 = list(valid['agee'])
x3 = list(valid['sfare'])
x4 = list(valid['Family_Size'])
x5 = list(valid['sp'])
y = list(valid['Survived'])

X1 = np.array(X1)
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
X5 = np.array(X5)
Y = np.array(Y)

x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
x5 = np.array(x5)
y = np.array(y)

X = np.zeros((len(X1), 15))
Y = Y.reshape(len(Y),1)

X[:,0] = 1
X1 = X1.reshape(len(X1))
X2 = X2.reshape(len(X2))
X3 = X3.reshape(len(X3))
X4 = X4.reshape(len(X4))
X5 = X5.reshape(len(X5))
X[:,1] = X1
X[:,2] = X2
X[:,3] = X3
X[:,4] = X4
X[:,5] = X1**2
X[:,6] = X2**2
X[:,7] = X2**3
X[:,8] = X2**4
X[:,9] = X2**5
X[:,10] = X2**6
X[:,11] = X2**7
X[:,12] = X4**2
X[:,13] = X4**3
X[:,14] = X4**4


X_t = X.transpose()
matrix = np.dot(X_t,X)
matrix_inverse = np.linalg.inv(matrix)
para = np.dot(matrix_inverse,np.dot(X_t,Y))

y_pred = np.zeros(len(x1))
y_pred = y_pred + para[0]
y_pred = y_pred + para[1] * x1
y_pred = y_pred + para[2] * x2
y_pred = y_pred + para[3] * x3
y_pred = y_pred + para[4] * x4
y_pred = y_pred + para[5] * x1**2
y_pred = y_pred + para[6] * x2**2
y_pred = y_pred + para[7] * x2**3
y_pred = y_pred + para[8] * x2**4
y_pred = y_pred + para[9] * x2**5
y_pred = y_pred + para[10] * x2**6
y_pred = y_pred + para[11] * x2**7
y_pred = y_pred + para[12] * x4**2
y_pred = y_pred + para[13] * x4**3
y_pred = y_pred + para[14] * x4**4


loss = 0
for i in range(len(x2)):
    loss = loss + (y[i] - y_pred[i])**2
loss = loss/len(x2)
print(loss)

#the prediction output
prediction=[]
for i in y_pred:
    if i<=0.5:
        prediction.append(0)
    else:
        prediction.append(1)















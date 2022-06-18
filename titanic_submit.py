import numpy as np
import pandas as pd

train_data = pd.read_csv("train.csv")
age_median = train_data['Pclass'].dropna().median() 
train_data['Pclass'].fillna(age_median, inplace = True)
Fare_median = train_data['Fare'].dropna().median() 
train_data['Fare'].fillna(Fare_median, inplace = True)

#make alphabetic parameters into numbers
train_data['Embarked'].fillna( 'S', inplace=True )
dict_Embarked = {'S': 1, 'Q': 2, 'C': 3}
train_data['Embarked'] = train_data['Embarked'].map(dict_Embarked).astype(int)

#make a new sp feature combining 3 features
train_data.loc[(train_data.Sex=='female')&(train_data.Pclass==1),'sp']=6
train_data.loc[(train_data.Sex=='female')&(train_data.Pclass==2),'sp']=5
train_data.loc[(train_data.Sex=='female')&(train_data.Pclass==3),'sp']=4
train_data.loc[(train_data.Sex=='male')&(train_data.Pclass==1),'sp']=3
train_data.loc[(train_data.Sex=='male')&(train_data.Pclass==2),'sp']=2
train_data.loc[(train_data.Sex=='male')&(train_data.Pclass==3),'sp']=1


#put the same cabin into groups
train_data['Cabin']=train_data['Cabin'].apply( lambda x:str(x)[0] if not pd.isnull(x) else '1' )
train_data['Cabin'] = train_data.Cabin.replace( ['D', 'C', 'B', 'A','E', 'G','F', 'T'],'2')

train_data[ 'cp' ] = np.nan
train_data.loc[(train_data.Cabin=='1')&(train_data.Embarked==1),'cp']=2
train_data.loc[(train_data.Cabin=='1')&(train_data.Embarked==2),'cp']=2
train_data.loc[(train_data.Cabin=='1')&(train_data.Embarked==3),'cp']=2
train_data.loc[(train_data.Cabin=='2')&(train_data.Embarked==1),'cp']=3
train_data.loc[(train_data.Cabin=='2')&(train_data.Embarked==2),'cp']=1
train_data.loc[(train_data.Cabin=='2')&(train_data.Embarked==3),'cp']=3

#trying to fill out the age with names and titles
import re
regex = re. compile( ' ([A-Za-z]+)\.')
train_data['Title'] = train_data.Name.map( lambda x:regex.search(x)[0])
# Dropping the first and the Last words
train_data['Title'] = train_data.Title.map( lambda x:x[1:][:-1] )
train_data['Title'].unique()
        
train_data['Title'].count()
train_data['Title'] = train_data.Title.replace( ['Don', 'Rev', 'Dr', 'Major','Lady', 'Sir','Capt', 'Countess','Jonkheer','Dona','Col' ],'Rare')
train_data['Title'] = train_data.Title.replace( ['Ms','Mlle'],'Miss')
train_data['Title']= train_data.Title.replace(['Mme','Lady'],"Mrs")
train_data['Title']=train_data['Title'].map({'Mr':0,'Rare':1,'Master':2,'Miss':3,'Mrs':4})

Age_Mean = train_data[['Title','Age']].groupby( by=['Title' ] ).mean()
Age_Median = train_data[['Title','Age' ]].groupby( by=['Title'] ).median()
Age_Mean.columns = ['Age Mean' ]
Age_Median.columns = ['Age Median' ]
Age_Mean.reset_index( inplace=True )
Age_Median.reset_index( inplace=True )

tt=train_data.groupby('Title')['Age'].median()
#print(tt)
t=train_data.groupby('Title')['Age'].median().values
train_data['agee']=train_data['Age']
for i in range(0,5):
    train_data.loc[(train_data.agee.isnull()) & (train_data.Title==i),'agee']=t[i]
train_data['agee']=train_data['agee'].astype('int')
#print(train_data.isnull().sum())


#create family size for every person
train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch'] + 1
Survival_Rate = train_data[['Family_Size','Survived']].groupby(by=['Family_Size']).agg(np.mean)*100
Survival_Rate.columns = ['Survival Rate(%)']
Survival_Rate.reset_index()

#let sfare be sqrt of fare and fill nan with 0
train_data['sfare'] = np.sqrt((train_data['Fare']))
#train_data['log_fare'] = np.log10(train_data['Fare'])
train_data['sfare'] = train_data['sfare'].fillna(0).astype(int)


###################################################################################

data = pd.read_csv("test.csv")
age_median = data['Pclass'].dropna().median() 
data['Pclass'].fillna(age_median, inplace = True)
Fare_median = data['Fare'].dropna().median() 
data['Fare'].fillna(Fare_median, inplace = True)


#make alphabetic parameters into numbers
data['Embarked'].fillna( 'S', inplace=True )
dict_Embarked = {'S': 1, 'Q': 2, 'C': 3}
data['Embarked'] = data['Embarked'].map(dict_Embarked).astype(int)

#make a new sp feature combining 3 features
data.loc[(data.Sex=='female')&(data.Pclass==1),'sp']=6
data.loc[(data.Sex=='female')&(data.Pclass==2),'sp']=5
data.loc[(data.Sex=='female')&(data.Pclass==3),'sp']=4
data.loc[(data.Sex=='male')&(data.Pclass==1),'sp']=3
data.loc[(data.Sex=='male')&(data.Pclass==2),'sp']=2
data.loc[(data.Sex=='male')&(data.Pclass==3),'sp']=1


data['Cabin']=data['Cabin'].apply( lambda x:str(x)[0] if not pd.isnull(x) else '1' )
data['Cabin'] = data.Cabin.replace( ['D', 'C', 'B', 'A','E', 'G','F', 'T'],'2')

data[ 'cp' ] = np.nan
data.loc[(data.Cabin=='1')&(data.Embarked==1),'cp']=2
data.loc[(data.Cabin=='1')&(data.Embarked==2),'cp']=2
data.loc[(data.Cabin=='1')&(data.Embarked==3),'cp']=2
data.loc[(data.Cabin=='2')&(data.Embarked==1),'cp']=3
data.loc[(data.Cabin=='2')&(data.Embarked==2),'cp']=1
data.loc[(data.Cabin=='2')&(data.Embarked==3),'cp']=3

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

#let sfare be sqrt of fare and fill nan with 0
data['sfare'] = np.sqrt((data['Fare']))
#data['log_fare'] = np.log10(data['Fare'])
data['sfare'] = data['sfare'].fillna(0).astype(int)


train = train_data
valid = data

X1 = list(train['sp'])
X2 = list(train['agee'])
X3 = list(train['sfare'])
X4 = list(train['Family_Size'])
X5 = list(train['cp'])
Y = list(train['Survived'])

x1 = list(valid['sp'])
x2 = list(valid['agee'])
x3 = list(valid['sfare'])
x4 = list(valid['Family_Size'])
x5 = list(valid['cp'])


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


X = np.zeros((len(X1), 18))
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
X[:,5] = X5
X[:,6] = X2**2
X[:,7] = X2**3
X[:,8] = X2**4
X[:,9] = X3**2
X[:,10] = X3**3
X[:,11] = X3**4
X[:,12] = X3**5
X[:,13] = X4**2
X[:,14] = X4**3
X[:,15] = X4**4
X[:,16] = X4**5
X[:,17] = X5**2

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
y_pred = y_pred + para[5] * x5
y_pred = y_pred + para[6] * x2**2
y_pred = y_pred + para[7] * x2**3
y_pred = y_pred + para[8] * x2**4
y_pred = y_pred + para[9] * x3**2
y_pred = y_pred + para[10] * x3**3
y_pred = y_pred + para[11] * x3**4
y_pred = y_pred + para[12] * x3**5
y_pred = y_pred + para[13] * x4**2
y_pred = y_pred + para[14] * x4**3
y_pred = y_pred + para[15] * x4**4
y_pred = y_pred + para[16] * x4**5
y_pred = y_pred + para[17] * x5**2


#Rewrite the result to 'Team_4.csv'
import csv
headers=['PassengerId','Survived']
Id = []
id = 0
rows = []
prediction=[]
for i in y_pred:
    if i<=0.5:
        prediction.append(0)
    else:
        prediction.append(1)
for i in range(len(prediction)):
    id = 892+i
    Id.append(id)
    rows.append([Id[i],prediction[i]])
with open('Team_4.csv', 'w', encoding = 'UTF8', newline = '') as f:
    writer=csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)
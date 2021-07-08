import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams



# 必要なモジュールをインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor

#data = pd.read_csv('./data/kaggle/nba_final.csv')
#print(data.head())

import seaborn as sns #importing seaborn module 
import warnings

import random as rnd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
# Support Vector Machines
import pickle 
import chainer
import chainer
from chainer.datasets import split_dataset_random, get_cross_validation_datasets_random
from chainer import iterators, optimizers, serializers, cuda
from chainer.optimizer_hooks import WeightDecay, Lasso

import mlp
import submlp
# import training
import saving_data
import ie_11_14
import calc
import running

# library
import datetime
import os
import argparse
import random
import pickle
import sys
import copy
import math
import pandas as pd
import numpy as np
import csv
import time
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
# data = pd.read_csv("./data/sklearn_data/diabesets_normalize_row_hdlunt.csv")
# colormap = plt.cm.RdYlGn
# plt.figure(figsize=(7,7))
# sns.heatmap(data.corr(), 
#             square=True,cmap=colormap, annot=True, robust = True)
# plt.savefig('./data/image/soukan_diabetes.png')
# plt.show()

df= pd.read_csv("./data/kaggle/heart_attack_3.csv")


with open("./result/train/pkl/IEmod_77233_nashi_0.001_max_min_1_mse_2_relu_1_adam_0.001_units_breast-cancer_2_monotony_ie_True_0_2_mno1.pkl", "rb") as mod: 
    model = pickle.load(mod)

x=np.array([1,2,3])
b=x.reshape(-1,1)# b列ベクトルになる
# retu = []
# for i in range(1001):
#     retu.append(i/1000)
retu = np.linspace(0,1,1000)
retu_np = np.array(retu, dtype=np.float32).reshape(-1,1)

for i in range(10):
    result = chainer.functions.sigmoid(model.fb[i](model.fa[i](retu_np))).array.flatten()
    plt.plot(retu, result, label="value{}".format(i))
    plt.show()
# import matplotlib.pyplot as plt
# retu = np.linspace(-1,1,1000)
# retu_np = np.array(retu, dtype=np.float32).reshape(-1,1)

# for i in range(10):
#     result = mod(retu_np).array.flatten()
#     plt.plot(retu, result, label="value{}".format(i))
#     plt.show()



for i in range(5):
    data = pd.read_csv('./data/Titanic/Titanic_shaplay/5cross_add{}_Tit.csv'.format(i+2), index_col=0)
    colormap = plt.cm.RdYlGn
    plt.figure(figsize=(7,7))
    #plt.title('Pearson Correlation of Features', y=1.05, size=15)
    #sns.set(font_scale=2)
    sns.heatmap(data, 
                square=True,cmap=colormap, annot=True, robust = True)
    plt.savefig('./data/image/5cross_add{}_Tit.png'.format(i+2))
#plt.show()
# 図の保存と図示



X_train = data.drop("Y", axis=1)
y_train = data["Y"]

# reg_linear = SVR(kernel='linear', C=1, epsilon=0.1)
# reg_poly = SVR(kernel='poly', C=1, epsilon=0.1)
# reg_rbf = SVR(kernel='rbf', C=1, epsilon=0.1)

# reg_linear.fit(X_train, np.ravel(y_train))
# reg_poly.fit(X_train, np.ravel(y_train))
# reg_rbf.fit(X_train, np.ravel(y_train))

# scores = (reg_linear.score(X_train, y_train),
# reg_poly.score(X_train, y_train),
# reg_rbf.score(X_train, y_train))

# plt.bar(("Linear", "poly", "RBF"), scores)
# plt.xlabel("Kernel")
# plt.ylabel("$R^2$ score")
# plt.show()

# pre_linear = reg_linear.predict(X_train)
# pre_poly = reg_poly.predict(X_train)
# pre_rbf = reg_rbf.predict(X_train)

# print("mse linear:{}".format(mean_squared_error(y_train, pre_linear)))
# print("mse poly:{}".format(mean_squared_error(y_train, pre_poly)))
# print("mse rbf:{}".format(mean_squared_error(y_train, pre_rbf)))
# print("mae linear:{}".format(mean_absolute_error(y_train, pre_linear)))
# print("mae poly:{}".format(mean_absolute_error(y_train, pre_poly)))
# print("mae rbf:{}".format(mean_absolute_error(y_train, pre_rbf)))
# print("R2:{}".format(scores))
# Gaussian Naive Bayes

# gaussian = GaussianNB()
# gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
# acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
# acc_gaussian

# Perceptron

# perceptron = Perceptron()
# perceptron.fit(X_train, Y_train)
# Y_pred = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# acc_perceptron

# Linear SVC

# linear_svc = LinearSVC()
# linear_svc.fit(X_train, Y_train)
# Y_pred = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
# acc_linear_svc

# Stochastic Gradient Descent

# sgd = SGDClassifier()
# sgd.fit(X_train, Y_train)
# Y_pred = sgd.predict(X_test)
# acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
# acc_sgd

# Decision Tree

decision_tree = DecisionTreeRegressor(max_depth=20)
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_train)
print("mse decision:{}".format(mean_squared_error(y_train, Y_pred)))
print("mae decision:{}".format(mean_absolute_error(y_train, Y_pred)))
print("R2  decision:{}".format(decision_tree.score(X_train,y_train)))
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# acc_decision_tree

# Random Forest

random_forest = RandomForestRegressor(n_estimators=100, max_depth=20)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_train)
print("mse decision:{}".format(mean_squared_error(y_train, Y_pred)))
print("mae decision:{}".format(mean_absolute_error(y_train, Y_pred)))
print("randomR2: {}".format(random_forest.score(X_train, y_train)))
# acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# acc_random_forest



xgb_params = {
                'max_depth':20,
                'objective': 'reg:linear',
                'eval_metric': 'rmse',
                # 'min_child_weight':2,
                # 'max_delta_step':6,
                # 'alpha':1/6,
}
dtrain = xgb.DMatrix(X_train, label=y_train)
watch = [(dtrain,'train'),(dtrain,'eval')]
evals_result = {}
bst = xgb.train(xgb_params,
        dtrain,
        num_boost_round=100,
        early_stopping_rounds=10,
        evals=watch,
        )
Y_pred = bst.predict(dtrain, ntree_limit = bst.best_ntree_limit)

print("mse xgb:{}".format(mean_squared_error(y_train, Y_pred)))
print("mae xgb:{}".format(mean_absolute_error(y_train, Y_pred)))
print("xgbR2: {}".format(r2_score(y_train, Y_pred)))
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

train_df = pd.read_csv('./data/Titanic/train_row.csv')
test_df = pd.read_csv('./data/Titanic/test_row.csv')
combine = [train_df, test_df]



print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head())

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

print(train_df.head())

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
print(train_df.head())

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

print(train_df.head())

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print(train_df.head())

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print(test_df.head())

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
test_df = test_df.drop(['Survived'], axis=1)
combine = [train_df, test_df]
    
print(train_df.head(10))

print(test_df.head(10))

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook
#plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
plt.rcParams['figure.figsize']=[6,3]
plt.rcParams['figure.dpi']=80

print(data.isnull().sum())

plt.figure(figsize=(10,10))
sns.heatmap(data.isnull(),cbar=False,cmap='YlGnBu')
plt.ioff()
plt.show()

df=data.copy()
df.drop(['Player.x', 'Player_ID', 'Pos1', 'Pos2', 'Tm', 'Season', 'Conference', 'Role', 'Play', 'X3P.', 'X2P', 'eFG.', 'FT.', 'Salary', 'mean_views', 'Pvot', 'PRank', 'Mvot', 'MRank'],axis=1,inplace=True)

sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

print(df.head())

print(df.isnull().sum())

df.to_csv('./data/kaggle/nba_num.csv')

df.info()#データの情報が見れる
df.describe()#最大最小偏差平均中央値などがわかる
#df.describe(include=['0']) 引数指定でいろいろ表示できる

data['Title']=0
for i in data:
    data['Title']=data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

pd.crosstab(data.Title,data.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex

data['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Other'],inplace=True)

#pd.crosstab(data.Title,data.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex

#data.groupby('Title')['Age'].mean() #lets check the average age by Initials

## Assigning the NaN Values with the Ceil values of the mean ages
data.loc[(data.Age.isnull())&(data.Title=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Title=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Title=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Title=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Title=='Other'),'Age']=46

print(data.Age.isnull().any()) #So no null values left finally 

data['Embarked'].fillna('S',inplace=True)
print(data.Embarked.isnull().any())# Finally No NaN values

cols = data.columns
print(list(cols))

data['Fare_Range']=pd.qcut(data['Fare'],4)
data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')

data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
data.head(2)

data['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')#checking the number of passenegers in each band

data["FamilySize"] = data["SibSp"] + data["Parch"]+1
#titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1
#data["FamilySize"].value_counts())
#sns.countplot('FamilySize',data=data)
pass

import re
#GettingLooking the prefix of all Passengers
#data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

#defining the figure size of our graphic
#plt.figure(figsize=(12,5))

#Plotting the result
# sns.countplot(x='Title', data=data, palette="hls")
# plt.xlabel("Title", fontsize=16) #seting the xtitle and size
# plt.ylabel("Count", fontsize=16) # Seting the ytitle and size
# plt.title("Title Name Count", fontsize=20) 
# plt.xticks(rotation=45)
# plt.show()

Title_Dictionary = {
        "Capt":       "Officer",
        "Col":        "Officer",
        "Major":      "Officer",
        "Dr":         "Officer",
        "Rev":        "Officer",
        "Jonkheer":   "Royalty",
        "Don":        "Royalty",
        "Sir" :       "Royalty",
        "the Countess":"Royalty",
        "Dona":       "Royalty",
        "Lady" :      "Royalty",
        "Mme":        "Mrs",
        "Ms":         "Mrs",
        "Mrs" :       "Mrs",
        "Mlle":       "Miss",
        "Miss" :      "Miss",
        "Mr" :        "Mr",
        "Master" :    "Master"
                   }
data['Title']=data.Title.map(Title_Dictionary)

print('Chance of Survival based on Titles:')
print(data.groupby("Title")["Survived"].mean())
#plt.figure(figsize(12,5))
# sns.countplot(x='Title',data=data,palette='hls',hue='Survived')
# plt.xlabel('Title',fontsize=15)
# plt.ylabel('Count',fontsize=15)
# plt.title('Count by Title',fontsize=20)
# plt.xticks(rotation=30)
# plt.show()

data['Sex'] = data['Sex'].astype(str)
data['Embarked'] = data['Embarked'].astype(str)
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Title'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
data.head(2)

df=data.copy()
df.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

df.drop(['SibSp','Parch'],axis=1,inplace=True)
df.head()

print(df.isnull().sum())

df = df.dropna()

df.to_csv('./data/Titanic/test_analys.csv')

#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

X = df.drop(labels='Survived',axis=1)
y = df['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state= 1234,stratify=y)

print('Training Set:',len(X_train))
print('Test Set:',len(X_test))
print('Training labels:',len(y_train))
print('Test labels:',len(y_test))

model=svm.SVC(kernel='linear',C=0.1,gamma=0.1)
model.fit(X_train,y_train)
prediction1=model.predict(X_test)
print('Accuracy for linear SVM is',metrics.accuracy_score(prediction1,y_test))

df=pd.read_csv('https://web.stanford.edu/~hastie/Papers/LARS/diabetes.sdata', delim_whitespace=True)
df2 = np.genfromtxt("./data/sklearn_data/diabesets_v6.csv", delimiter=",", filling_values=0).astype(np.float32)
df2 = pd.DataFrame(data=df2,columns=['y','age','sex','bmi','map','tc','ldl','hdl','tch','ltg','glu'], dtype='float')
import seaborn as sns
df.head()
cols = ['y','age','sex','bmi','map','tc']
cols2 = ['y','ldl','hdl','tch','ltg','glu']
sns.pairplot(df2)
plt.show()
# データの取得
# sklearnのデータセットload_breast_cancerについて軽く解説
# 乳房塊の微細針吸引物（FNA）のデジタル画像中に存在する細胞核の特徴を捉えたものを特徴量として置き。その患者が陰性か陽性かを判定するというもの

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target


#特徴量とその重要度を含むDFを作成
def get_feature_importances(X, y, shuffle=False):
    # 必要ならば目的変数をシャッフル
    if shuffle:
        y = np.random.permutation(y)

    # モデルの学習
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)

    # 特徴量の重要度を含むデータフレームを作成
    imp_df = pd.DataFrame()
    imp_df["feature"] = X.columns
    imp_df["importance"] = clf.feature_importances_
    return imp_df.sort_values("importance", ascending=False)

# 実際の目的変数でモデルを学習し、特徴量の重要度を含むデータフレームを作成
actual_imp_df = get_feature_importances(X, y, shuffle=False)

# 目的変数をシャッフルした状態でモデルを学習し、特徴量の重要度を含むデータフレームを作成
#シャッフル重要度の母数を増やすためにN_RUNS回学習を行う
N_RUNS = 100
null_imp_df = pd.DataFrame()
for i in range(N_RUNS):
    imp_df = get_feature_importances(X, y, shuffle=True)
    imp_df["run"] = i + 1
    null_imp_df = pd.concat([null_imp_df, imp_df])

#重要度を可視化する関数
def display_distributions(actual_imp_df, null_imp_df, feature):
    # ある特徴量に対する重要度を取得
    actual_imp = actual_imp_df.query(f"feature == '{feature}'")["importance"].mean()
    null_imp = null_imp_df.query(f"feature == '{feature}'")["importance"]

    # 可視化
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    a = ax.hist(null_imp, label="Null importances")
    ax.vlines(x=actual_imp, ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
    ax.legend(loc="upper right")
    ax.set_title(f"Importance of {feature.upper()}", fontweight='bold')
    plt.xlabel(f"Null Importance Distribution for {feature.upper()}")
    plt.ylabel("Importance")
    plt.show()



# 実データにおいて特徴量の重要度が高かった上位5位を表示
for feature in actual_imp_df["feature"][:5]:
    display_distributions(actual_imp_df, null_imp_df, feature)
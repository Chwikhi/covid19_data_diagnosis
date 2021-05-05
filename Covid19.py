# -*- coding: utf-8 -*-
"""
Created on Wed May  5 23:25:28 2021

@author: Wadie
"""
### Chargement des données
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data = pd.read_excel('dataset.xlsx')
data.head()

### Exploration et visualisation
df = data.copy()
print(df.shape)
pd.set_option('display.max_row', 111)
print(df.dtypes.value_counts())
plt.figure(figsize=(20, 15))
sns.heatmap(df.isna(), cbar=False)
print(((df.isna().sum())/df.shape[0]).sort_values(ascending=True))

df = df[df.columns[df.isna().sum()/df.shape[0] < 0.9]]
plt.figure(figsize=(20, 15))
sns.heatmap(df.isna(), cbar=False)

df = df.drop('Patient ID', axis=1)
print(df.shape)
print(df.dtypes.value_counts())
print(df.head())

# Target
df['SARS-Cov-2 exam result'].value_counts(normalize=True).plot.pie()

# float dtypes features
for c in df.select_dtypes('float'):
    plt.figure()
    sns.distplot(df[c])

#patient age    
sns.distplot(df['Patient age quantile'])

# object dtypes features
for c in df.select_dtypes('object'):
    print(df[c].value_counts(normalize=True))
    plt.figure()
    df[c].value_counts().plot.pie()

## target / features
df_pos = df[df['SARS-Cov-2 exam result'] == 'positive']
df_neg = df[df['SARS-Cov-2 exam result'] == 'negative']
NaN_rate = df.isna().sum()/df.shape[0]
blood_features = df.columns[(NaN_rate < 0.9) & (NaN_rate > 0.88)]
viral_features = df.columns[(NaN_rate < 0.88) & (NaN_rate > 0.75)]

# target / blood features
for c in blood_features:
    plt.figure(figsize=(12, 8))
    sns.distplot(df_pos[c], label='positive')
    sns.distplot(df_neg[c], label='negative')
    plt.legend()

#target / patient age
sns.countplot(x='Patient age quantile', hue='SARS-Cov-2 exam result', data=df)

# target / viral features
for c in viral_features:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[c]), annot=True, fmt='d')
    
## features/features
# blood features
sns.heatmap(df[blood_features].corr())

plt.figure(figsize=(15,15))
sns.pairplot(df[blood_features])
plt.show()

# blood features / patient age
for c in df[blood_features]:
    sns.lmplot(x='Patient age quantile', y=c, hue='SARS-Cov-2 exam result', data=df)
df.corr()['Patient age quantile'].sort_values(ascending=True)

# blood features / viral features
df['malade'] = np.sum(df[viral_features[:-2]] == 'detected', axis=1) >=1
print(df.head())
df_malade = df[df['malade'] == True]
df_non_malade = df[df['malade'] == False]
for c in blood_features:
    plt.figure(figsize=(7, 5))
    sns.distplot(df_malade[c], label='malade')
    sns.distplot(df_non_malade[c], label='Non malade')
    plt.legend()
    
def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'
    if df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'soin semi_intensive'
    if df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'soin intensive'
    else:
        return 'inconnu'
    
df['status'] = df.apply(hospitalisation, axis=1)
for c in df[blood_features]:
    plt.figure()
    for soin in df['status'].unique():
        sns.distplot(df[df['status'] == soin][c], label=soin)
    plt.legend()
    plt.show()
    


### Prétraitement

df = data.copy()
NaN_rate = df.isna().sum()/df.shape[0]
blood_features = list(df.columns[(NaN_rate < 0.9) & (NaN_rate > 0.88)])
viral_features = list(df.columns[(NaN_rate < 0.88) & (NaN_rate > 0.75)])
viral_features = viral_features[:-2]

df = df[['Patient age quantile', 'SARS-Cov-2 exam result'] + blood_features + viral_features]
print(df.head())

from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)
print(trainset['SARS-Cov-2 exam result'].value_counts())
print(testset['SARS-Cov-2 exam result'].value_counts())

def encodage(df):
    code = {'positive':1, 'negative':0, 'detected':1, 'not_detected':0}
    for c in df.select_dtypes('object'):
        df[c] = df[c].map(code)
    return df

def feature_engeneering(df):
    df['malade'] = df[viral_features].sum(axis=1) >= 1
    df = df.drop(viral_features, axis=1)
    return df

def imputation(df):
    #df = df.fillna(-999)
    df = df.dropna(axis=0)
    #try missing indicator
    return df

def preprocessing(df):
    df = encodage(df)
    feature_engeneering(df)
    df = imputation(df)
    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']
    print(y.value_counts())
    return X, y

Xtrain, ytrain = preprocessing(trainset)
Xtest, ytest = preprocessing(testset)


## Modélisation et evaluation préliminaire pour prétraitement utile

from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

def evaluation(model):
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    print(confusion_matrix(ytest, ypred))
    print(classification_report(ytest, ypred))
    N, train_score, validation_score = learning_curve(model, Xtrain, ytrain,
                                                      cv=4, scoring='f1',
                                                      train_sizes=np.linspace(0.1, 1, 10))
    plt.figure()
    plt.plot(N, train_score.mean(axis=1), label='training score')
    plt.plot(N, validation_score.mean(axis=1), label='validation score')
    plt.title('learning curve')
    plt.legend()
    plt.show()

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)
evaluation(model)

features_select = pd.DataFrame(model.feature_importances_, index=Xtrain.columns)
features_select.sort_values(by= features_select.columns[0], ascending=False).plot.bar(figsize=(12,8))    

from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(random_state=0)
evaluation(model1)

features_select = pd.DataFrame(model1.feature_importances_, index=Xtrain.columns)
features_select.sort_values(by= features_select.columns[0], ascending=False).plot.bar(figsize=(12,8))

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

model2 = make_pipeline(PolynomialFeatures(2, include_bias=False),
                       PCA(), SelectKBest(f_classif, k=10),
                       DecisionTreeClassifier(random_state=0))
evaluation(model2)



### Modélisation

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), PCA(),
                             StandardScaler(), SelectKBest(f_classif, k=6))

AdaBoostClassifier = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
GradientboostingClassifier = make_pipeline(preprocessor,
                                           GradientBoostingClassifier(random_state=0))
SVC = make_pipeline(preprocessor, SVC(random_state=0))
Knn = make_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=1))
Tree = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=0))
MLPClassifier = make_pipeline(preprocessor, MLPClassifier(random_state=0,
                                                          max_iter=400, solver='adam',
                                                          shuffle=True))

model_list = [AdaBoostClassifier, GradientboostingClassifier, SVC, Knn, Tree, MLPClassifier]

for model in model_list:
    print(model)
    evaluation(model)
    
## On continue avec SCV et MLPClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

Hyper_params_SVC = {'svc__gamma':[1e-3,1e-4], 'svc__C':[1, 10, 100, 1000, 10000]}
grid_SVC = GridSearchCV(SVC, param_grid=Hyper_params_SVC, cv=4, scoring='recall')
grid_SVC.fit(Xtrain, ytrain)
print(grid_SVC.best_params_)
evaluation(grid_SVC.best_estimator_)

Hyper_params_MLP = {'mlpclassifier__beta_1':np.linspace(1e-3, 0.99, 3), 
                    'mlpclassifier__beta_2':np.linspace(0.5, 0.99, 3),
                    'mlpclassifier__alpha':[1e-2, 1e-3]}
grid_MLP = GridSearchCV(MLPClassifier, param_grid=Hyper_params_MLP, cv=4, scoring='recall')
grid_MLP.fit(Xtrain, ytrain)
print(grid_MLP.best_params_)
evaluation(grid_MLP.best_estimator_)

## Prise de Décision

from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(ytest,
                                                      grid_SVC.best_estimator_.decision_function(Xtest))
plt.figure(figsize=(12, 8))
plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.title('SVC')
plt.legend()
plt.show

def optimized_model(model, X, threshold=0):
    return model.decision_function(X) > threshold

ypred = optimized_model(grid_SVC.best_estimator_, Xtest, threshold=-1.7)
print('final model f1_score : ', f1_score(ytest,ypred))
print('final model recall_score : ', recall_score(ytest,ypred))

### fin


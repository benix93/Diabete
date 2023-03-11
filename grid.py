import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

start_time = time.time()
data = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
print("_____________________________________________________________________________")
print("SHAPE0: " + str(data.shape))
print("_____________________________________________________________________________")
print(data.head())
print("_____________________________________________________________________________")
print(data.info())
print("_____________________________________________________________________________")
print(data.describe())
print("_____________________________________________________________________________")
print(data.isnull().sum())

print(data.duplicated().sum())
data.drop_duplicates(inplace=True)
print("SHAPE1: " + str(data.shape))
data = data.astype(int)
print("_____________________________________________________________________________")
print(data.head())
print(data.info())

unique = {}
for col in data.columns:
    unique[col] = data[col].value_counts().shape[0]
print(pd.DataFrame(unique, index=['Count']).transpose())


def v_counts(dataframe):
    for i in dataframe:
        print(dataframe[i].value_counts())
        print("_____________________________________________________________________________")


print(v_counts(data))

plt.figure(figsize=(14, 14))
for i, col in enumerate(['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']):
    plt.subplot(4, 2, i + 1)
    plt.boxplot(x=col, data=data, labels=[col], patch_artist=True)
plt.show()

# Facciamo un drop degli outliers caratterizzati da un BMI maggiore di 50 e minore di 13 in quanto sono sicuramente
# valori non veritieri

mask = (data['BMI'] > 50) | (data['BMI'] < 13)
data = data.drop(index=data[mask].index)
print("SHAPE2: " + str(data.shape))


# Suddividiamo i BMI in fasce
def bmi_map(x):
    if x < 18.5:
        return 1
    elif 18.5 <= x <= 24.9:
        return 2
    elif 25 <= x <= 29.9:
        return 3
    elif 30 <= x <= 34.9:
        return 4
    elif 35 <= x <= 39.9:
        return 5
    else:
        return 6


# Riduciamo i range piu ampi
def age_map(x):
    if 1 <= x <= 3:
        return 1
    elif 4 <= x <= 5:
        return 2
    elif 6 <= x <= 7:
        return 3
    elif 8 <= x <= 9:
        return 4
    elif 10 <= x <= 11:
        return 5
    else:
        return 6


def days_map(x):
    if x <= 10:
        return 1
    elif 11 <= x <= 20:
        return 2
    else:
        return 3


data2 = data.copy()
data2['BMI'] = data2["BMI"].apply(bmi_map)
data2['Age'] = data2["Age"].apply(age_map)
data2['PhysHlth'] = data2["PhysHlth"].apply(days_map)
data2['MentHlth'] = data2["MentHlth"].apply(days_map)
data2['Diabetes_binary'] = data2['Diabetes_binary'].replace({0: 'No', 1: 'Si'})
data2['Smoker'] = data2['Smoker'].replace({0: 'No', 1: 'Si'})
data2['Sex'] = data2['Sex'].replace({0: 'Uomo', 1: 'Donna'})
data2['HighBP'] = data2['HighBP'].replace({0: 'No', 1: 'Si'})
data2['HighChol'] = data2['HighChol'].replace({0: 'No', 1: 'Si'})
data2['CholCheck'] = data2['CholCheck'].replace({0: 'No', 1: 'Si'})
data2['DiffWalk'] = data2['DiffWalk'].replace({0: 'No', 1: 'Si'})
data2['AnyHealthcare'] = data2['AnyHealthcare'].replace({0: 'No', 1: 'Si'})
data2['HvyAlcoholConsump'] = data2['HvyAlcoholConsump'].replace({0: 'No', 1: 'Si'})
data2['Veggies'] = data2['Veggies'].replace({0: 'No', 1: 'Si'})
data2['Fruits'] = data2['Fruits'].replace({0: 'No', 1: 'Si'})
data2['PhysActivity'] = data2['PhysActivity'].replace({0: 'No', 1: 'Si'})
data2['HeartDiseaseorAttack'] = data2['HeartDiseaseorAttack'].replace({0: 'No', 1: 'Si'})
data2['Stroke'] = data2['Stroke'].replace({0: 'No', 1: 'Si'})
data2['NoDocbcCost'] = data2['Smoker'].replace({0: 'No', 1: 'Si'})

data2['BMI'] = data2['BMI'].replace(
    {1: '18.5 o meno', 2: '18.5-24.9', 3: '25-29.9', 4: '30-34.9', 5: '35-39.9', 6: '40+'})
data2['Age'] = data2['Age'].replace({1: '18-34', 2: '35-44', 3: '45-54', 4: '55-64', 5: '65-74', 6: '75+'})
data2['Education'] = data2['Education'].replace(
    {1: '1-Nessuna', 2: '2-Elementari', 3: '3-Medie', 4: '4-Superiori', 5: '5-Triennale', 6: '6-Magistrale'})
data2['Income'] = data2['Income'].replace(
    {1: '10k o meno', 2: '10k-15k', 3: '15-20k', 4: '20-25k', 5: '25-35k', 6: '35-50k', 7: '50-75k', 8: '80+'})
data2['GenHlth'] = data2['GenHlth'].replace(
    {1: '1-Eccellente', 2: '2-Molto buona', 3: '3-Buona', 4: '4-Discreta', 5: '5-Pessima'})
data2['PhysHlth'] = data2['PhysHlth'].replace(
    {1: '1-10', 2: '11-20', 3: '21-30'})
data2['MentHlth'] = data2['MentHlth'].replace(
    {1: '1-10', 2: '11-20', 3: '21-30'})

plt.figure(figsize=(10, 10))
colors = ['dodgerblue', 'crimson']
data2['Diabetes_binary'].value_counts().plot.bar(color=colors)
plt.ylabel('Popolazione')
plt.xlabel('Diabete')
plt.title('Distribuzione diabete')
plt.show()

# Piechart BMI
data2['BMI'].value_counts().plot(figsize=(10, 10), kind='pie', autopct='%.1f', colors=sn.color_palette('Set2'),
                                 pctdistance=0.9, radius=1)
plt.title('Distribuzione BMI')
plt.axis('off')
plt.legend(title='BMI', loc='upper right', bbox_to_anchor=(1, 1))
plt.show()

# Valutiamo le percentuali con cui si presentano le features binarie
fig, axes = plt.subplots(4, 3, figsize=(15, 15))
for i, ax in enumerate(axes.flatten()):
    if i < len(['Diabetes_binary', 'Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
                'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke']):
        col = ['Diabetes_binary', 'Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
               'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke'][i]
        labels = data2[col].unique()
        ax.pie(data2[col].value_counts(), labels=labels, autopct='%.1f', colors=colors)
        ax.set_title(col)
plt.show()

fig, axes = plt.subplots(4, 3, figsize=(15, 20))
for i, ax in enumerate(axes.flatten()):
    if i < len(['Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
                'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke']):
        col = ['Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
               'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke'][i]
        tab = pd.crosstab(data2[col], data2.Diabetes_binary, normalize='index') * 100
        ax = tab.plot(kind="bar", stacked=True, figsize=(20, 20), color=colors, ax=ax)
        ax.set_title(f'{col} x Diabete')
        ax.set_xlabel(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel('Percentuale')
        ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1.10, 1))
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for i, ax in enumerate(axes.flatten()):
    if i < len(['Smoker', 'AnyHealthcare', 'NoDocbcCost']):
        col = ['Smoker', 'AnyHealthcare', 'NoDocbcCost'][i]
        tab = pd.crosstab(data2[col], data2.Diabetes_binary, normalize='index') * 100
        ax = tab.plot(kind="bar", stacked=True, figsize=(18, 6), color=colors, ax=ax)
        ax.set_title(f'{col} x Diabete')
        ax.set_xlabel(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel('Percentuale')
        ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1.10, 1))
plt.show()

# Analizziamo il rapporto tra BMI e incidenza del diabete
tab = pd.crosstab(data2.BMI, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x BMI')
plt.xlabel('BMI')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Istruzione e incidenza del Diabete
data2['Education'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
plt.xticks(rotation=0)
plt.ylabel('Popolazione')
plt.xlabel('Livelli di Istruzione')
plt.title('Distribuzione Livello Istruzione')
plt.show()

tab = pd.crosstab(data2.Education, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(18, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Istruzione')
plt.xlabel('Istruzione')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Reddito e incidenza del diabete
data2['Income'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
plt.xticks(rotation=0)
plt.ylabel('Popolazione')
plt.xlabel('Fasce di reddito')
plt.title('Distribuzione Redditi')

tab = pd.crosstab(data2.Income, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(18, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Fasce di Reddito')
plt.xlabel('Reddito')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Salute Generale e incidenza del diabete
data2['GenHlth'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
plt.xticks(rotation=0)
plt.ylabel('Popolazione')
plt.xlabel('Livello di salute')
plt.title('Distribuzione Salute Generale')

tab = pd.crosstab(data2.GenHlth, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Salute Generale')
plt.xlabel('Salute Generale')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Salute Mentale e incidenza del diabete
data2['MentHlth'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
plt.ylabel('Popolazione')
plt.xlabel('Giorni')
plt.xticks(rotation=0)
plt.title('Distribuzione Salute Mentale')

tab = pd.crosstab(data2.MentHlth, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Salute Mentale')
plt.xlabel('Giorni')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.1f}%' for x in p.datavalues])
plt.show()

# Crosstab tra diabete e Age
data2['Age'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
plt.ylabel('Popolazione')
plt.xlabel('Fasce di etá')
plt.xticks(rotation=0)
plt.title('Distribuzione Age')
plt.show()

tab = pd.crosstab(data2.Age, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
ax.legend(title='Diabete')
plt.title('Distribuzione frequenze Diabete x Age')
plt.xlabel('Fasce di etá')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Crosstab tra diabete e Salute fisica
data2['PhysHlth'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
plt.ylabel('Popolazione')
plt.xlabel('Giorni')
plt.xticks(rotation=0)
plt.title('Distribuzione Problemi Fisici')

tab = pd.crosstab(data2.PhysHlth, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Problemi fisici')
plt.xlabel('Giorni')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Matrice correlazione
plt.figure(figsize=(15, 15))
sn.heatmap(data.corr(), square=True, annot=True, cmap='YlGnBu', fmt='.2f', robust=True)
plt.title("Matrice Correlazione")
plt.show()

# Istogramma correlazione con Diabetes_Binary
plt.figure(figsize=(15, 15))
corr = data.corr().sort_values(by='Diabetes_binary', ascending=False)
corr = corr[corr.index != 'Diabetes_binary']
corr['Diabetes_binary'].plot(kind='bar', color='firebrick')
plt.xlabel('Features')
plt.ylabel('Correlazione')
plt.title('Correlazione con Diabete')
plt.axhline(y=0.05, linestyle='--', color='gray')
plt.axhline(y=-0.05, linestyle='--', color='gray')
plt.show()

# Vengono eliminate le features meno significative
X = data.drop(['Diabetes_binary', 'AnyHealthcare', 'NoDocbcCost', 'Fruits', 'Veggies', 'Sex'], axis=1)
y = data['Diabetes_binary']

# names = ["Decision Tree", 'Random Forest', 'Logistic Regression', 'Nearest Neighbors', "Naive Bayes", 'GradientBoost',
#          'XGB', 'LGBM', 'CatBoost', 'AdaBoost']
names = ['AdaBoost']

classifiers = [
    # DecisionTreeClassifier(),
    # RandomForestClassifier(random_state=42),
    # LogisticRegression(random_state=42),
    # KNeighborsClassifier(),
    # GaussianNB(),
    # GradientBoostingClassifier(random_state=42),
    # XGBClassifier(random_state=42),
    # # LGBMClassifier(),
    # CatBoostClassifier(),
    AdaBoostClassifier()
]

params = [
    # {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [5, 9, 15, None],
    #     'min_samples_split': [2, 3, 4],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['sqrt', 'log2']
    # },  # Decision Tree
    #
    # {
    #     'n_estimators': [150, 200, 250],
    #     'max_depth': [7, 9, 12, 15, 20, None],
    #     'criterion': ['gini', 'entropy'],
    #     'max_features': ['sqrt', 'log2'],
    #     'min_samples_leaf': [1, 2, 4],
    #     'min_samples_split': [2, 3, 4],
    # },  # Random Forest
    #
    # {
    #     'penalty': ['l1', 'l2'],
    #     'C': [0.25, 0.5, 0.75, 1, 10],
    #     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    # },  # Logistic Regression
    #
    # {
    #     'n_neighbors': [5, 7, 9],
    #     'weights': ['uniform', 'distance'],
    #     'algorithm': ['auto', 'kd_tree']
    # },  # KNN
    #
    # {},  # Naive Bayes
    #
    # {
    #     "learning_rate": [0.05, 0.1, 0.2],
    #     "max_depth": [3, 5, 7],
    #     "max_features": ["log2", "sqrt"],
    #     "n_estimators": [100, 150, 250],
    #     'min_samples_split': [2, 3, 4],
    #     'min_samples_leaf': [1, 2, 4]
    # },  # GradientBoost
    #
    # {
    #     "n_estimators": [100, 150, 250],
    #     'min_child_weight': [1, 3, 5],
    #     'max_depth': [3, 5, 7],
    #     'subsample': [0.8, 1.0],
    #     'colsample_bytree': [0.8, 1.0],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'eval_metric': ['error']
    # },  # XGB
    #
    # {
    #     'learning_rate': [0.05, 0.1, 0.2],
    #     'max_depth': [3, 5, 7],
    #     'n_estimators': [50, 100, 200],
    #     'num_leaves': [6, 8, 12],
    #     'boosting_type': ['gbdt', 'dart'],
    #     'objective': ['binary'],
    #     'subsample': [0.5, 0.7, 1],
    #     'reg_alpha': [0, 0.1, 0.5],
    #     'reg_lambda': [0, 0.1, 0.5],
    #     'min_child_samples': [10, 20, 30]
    # },  # LGBM

    # {
    #     'iterations': [500, 1000],
    #     'depth': [4, 5, 6],
    #     'loss_function': ['Logloss', 'CrossEntropy'],
#'learning_rate': [0.05, 0.1, 0.15],
#'l2_leaf_reg': [1, 3, 5],  # Coefficiente di regolarizzazione L2
    #     'leaf_estimation_iterations': [10],
    #     'logging_level': ['Silent'],
    #     'random_seed': [42]
    # },  # CatBoost

    {
        'estimator': [DecisionTreeClassifier()],
        'estimator__min_samples_split': [2, 3, 4],
        'estimator__min_samples_leaf': [1, 2, 3],
        'estimator__max_depth': [3, 4, 5],
        'estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1]
    }  # AdaBoost
]

param_results = pd.DataFrame(columns=["Classifier", "Best Parameters"])
results = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score", "Time"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# SMOTE
smote = SMOTE(sampling_strategy=1, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("_____________________________________________________________________________")
for name, clf, param in zip(names, classifiers, params):
    start = time.time()
    print('\n -> ' + name)
    clf = GridSearchCV(clf, param_grid=param, cv=3, n_jobs=-1, scoring='recall')
    clf.fit(X_train, y_train)
    print(f'Migliori parametri per {name}: {clf.best_params_}')

    param_results = pd.concat([param_results, pd.DataFrame({"Classifier": name,
                                                            "Best Parameters": [clf.best_params_]},
                                                           index=[0])], ignore_index=True)

    print("Tempo: ", round(time.time() - start, 3))

# Stampa dei risultati in una tabella
print()
print("_____________________________________________________________________________")
print("_______________________________ RISULTATI ___________________________________")
print(results)
print("_____________________________________________________________________________\n")
results.to_csv("results.csv")
param_results.to_csv("best_params_acc.csv")
print("Tempo di esecuzione --- %s secondi ---" % (time.time() - start_time))

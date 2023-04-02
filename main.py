import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import shap
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from lime import lime_tabular
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
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


# Facciamo un drop degli outliers caratterizzati da un BMI maggiore di 80 e minore di 13 in quanto sono quasi sicuramente
# valori non veritieri

# mask = (data['BMI'] > 75) | (data['BMI'] < 15)
# data = data.drop(index=data[mask].index)
# print("SHAPE2: " + str(data.shape))


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


#
# data2 = data.copy()
# data2['BMI'] = data2["BMI"].apply(bmi_map)
# data2['Age'] = data2["Age"].apply(age_map)
# data2['PhysHlth'] = data2["PhysHlth"].apply(days_map)
# data2['MentHlth'] = data2["MentHlth"].apply(days_map)
# data2['Diabetes_binary'] = data2['Diabetes_binary'].replace({0: 'No', 1: 'Si'})
# data2['Smoker'] = data2['Smoker'].replace({0: 'No', 1: 'Si'})
# data2['Sex'] = data2['Sex'].replace({0: 'Donna', 1: 'Uomo'})
# data2['HighBP'] = data2['HighBP'].replace({0: 'No', 1: 'Si'})
# data2['HighChol'] = data2['HighChol'].replace({0: 'No', 1: 'Si'})
# data2['CholCheck'] = data2['CholCheck'].replace({0: 'No', 1: 'Si'})
# data2['DiffWalk'] = data2['DiffWalk'].replace({0: 'No', 1: 'Si'})
# data2['AnyHealthcare'] = data2['AnyHealthcare'].replace({0: 'No', 1: 'Si'})
# data2['HvyAlcoholConsump'] = data2['HvyAlcoholConsump'].replace({0: 'No', 1: 'Si'})
# data2['Veggies'] = data2['Veggies'].replace({0: 'No', 1: 'Si'})
# data2['Fruits'] = data2['Fruits'].replace({0: 'No', 1: 'Si'})
# data2['PhysActivity'] = data2['PhysActivity'].replace({0: 'No', 1: 'Si'})
# data2['HeartDiseaseorAttack'] = data2['HeartDiseaseorAttack'].replace({0: 'No', 1: 'Si'})
# data2['Stroke'] = data2['Stroke'].replace({0: 'No', 1: 'Si'})
# data2['NoDocbcCost'] = data2['Smoker'].replace({0: 'No', 1: 'Si'})
#
# data2['BMI'] = data2['BMI'].replace(
#     {1: '18.5 o meno', 2: '18.5-24.9', 3: '25-29.9', 4: '30-34.9', 5: '35-39.9', 6: '40+'})
# data2['Age'] = data2['Age'].replace({1: '18-34', 2: '35-44', 3: '45-54', 4: '55-64', 5: '65-74', 6: '75+'})
# data2['Education'] = data2['Education'].replace(
#     {1: '1-Nessuna', 2: '2-Elementari', 3: '3-Medie', 4: '4-Superiori', 5: '5-Triennale', 6: '6-Magistrale'})
# data2['Income'] = data2['Income'].replace(
#     {1: '10k o meno', 2: '10k-15k', 3: '15-20k', 4: '20-25k', 5: '25-35k', 6: '35-50k', 7: '50-75k', 8: '80+'})
# data2['GenHlth'] = data2['GenHlth'].replace(
#     {1: '1-Eccellente', 2: '2-Molto buona', 3: '3-Buona', 4: '4-Discreta', 5: '5-Pessima'})
# data2['PhysHlth'] = data2['PhysHlth'].replace(
#     {1: '1-10', 2: '11-20', 3: '21-30'})
# data2['MentHlth'] = data2['MentHlth'].replace(
#     {1: '1-10', 2: '11-20', 3: '21-30'})
#
# plt.figure(figsize=(10, 10))
# colors = ['dodgerblue', 'crimson']
# data2['Diabetes_binary'].value_counts().plot.bar(color=colors)
# plt.ylabel('Popolazione')
# plt.xlabel('Diabete')
# plt.title('Distribuzione diabete')
# plt.show()
#
# # Piechart BMI
# data2['BMI'].value_counts().plot(figsize=(10, 10), kind='pie', autopct='%.1f', colors=sn.color_palette('Set2'),
#                                  pctdistance=0.9, radius=1)
# plt.title('Distribuzione BMI')
# plt.axis('off')
# plt.legend(title='BMI', loc='upper right', bbox_to_anchor=(1, 1))
# plt.show()
#
# fig, axes = plt.subplots(4, 3, figsize=(15, 15))
# for i, ax in enumerate(axes.flatten()):
#     if i < len(['Diabetes_binary', 'Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
#                 'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke']):
#         col = ['Diabetes_binary', 'Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
#                'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke'][i]
#         labels = data2[col].unique()
#         values = data2[col].value_counts()
#         ax.bar(labels, values, color=colors, width=0.5)
#         ax.set_title(col)
# plt.tight_layout()
# plt.show()
#
# fig, axes = plt.subplots(4, 3, figsize=(15, 15))
# for i, ax in enumerate(axes.flatten()):
#     if i < len(['Smoker', 'Stroke', 'NoDocbcCost']):
#         col = ['Smoker', 'Stroke', 'NoDocbcCost'][i]
#         labels = data2[col].unique()
#         values = data2[col].value_counts()
#         ax.bar(labels, values, color=colors, width=0.5)
#         ax.set_title(col)
# plt.tight_layout()
# plt.show()
#
#
# fig, axes = plt.subplots(4, 3, figsize=(20, 25))
# for i, ax in enumerate(axes.flatten()):
#     if i < len(['Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
#                 'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke']):
#         col = ['Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
#                'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke'][i]
#         tab = pd.crosstab(data2[col], data2.Diabetes_binary, normalize='index') * 100
#         ax = tab.plot(kind="bar", stacked=True, figsize=(20, 20), color=colors, ax=ax)
#         ax.set_title(f'{col} x Diabete')
#         ax.set_xlabel(col)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#         ax.set_ylabel('Percentuale')
#         ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1.10, 1))
# plt.tight_layout()
# plt.show()
#
# fig, axes = plt.subplots(4, 3, figsize=(15, 15))
# for i, ax in enumerate(axes.flatten()):
#     if i < len(['Smoker', 'AnyHealthcare', 'NoDocbcCost']):
#         col = ['Smoker', 'AnyHealthcare', 'NoDocbcCost'][i]
#         tab = pd.crosstab(data2[col], data2.Diabetes_binary, normalize='index') * 100
#         ax = tab.plot(kind="bar", stacked=True, color=colors, ax=ax)
#         ax.set_title(f'{col} x Diabete')
#         ax.set_xlabel(col)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#         ax.set_ylabel('Percentuale')
#         ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1.10, 1))
# plt.tight_layout()
# plt.show()
#
# # Analizziamo il rapporto tra BMI e incidenza del diabete
# tab = pd.crosstab(data2.BMI, data2.Diabetes_binary, normalize='index') * 100
# ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
# ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
# plt.title('Diabete x BMI')
# plt.xlabel('BMI')
# plt.xticks(rotation=0)
# plt.ylabel('Percentuale')
# for p in ax.containers:
#     ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
# plt.show()
#
# # Analizziamo il rapporto tra Istruzione e incidenza del Diabete
# data2['Education'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
# plt.xticks(rotation=0)
# plt.ylabel('Popolazione')
# plt.xlabel('Livelli di Istruzione')
# plt.title('Distribuzione Livello Istruzione')
# plt.show()
#
# tab = pd.crosstab(data2.Education, data2.Diabetes_binary, normalize='index') * 100
# ax = tab.plot(kind="bar", figsize=(18, 6), color=colors)
# ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
# plt.title('Diabete x Istruzione')
# plt.xlabel('Istruzione')
# plt.xticks(rotation=0)
# plt.ylabel('Percentuale')
# for p in ax.containers:
#     ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
# plt.show()
#
# # Analizziamo il rapporto tra Reddito e incidenza del diabete
# data2['Income'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
# plt.xticks(rotation=0)
# plt.ylabel('Popolazione')
# plt.xlabel('Fasce di reddito')
# plt.title('Distribuzione Redditi')
#
# tab = pd.crosstab(data2.Income, data2.Diabetes_binary, normalize='index') * 100
# ax = tab.plot(kind="bar", figsize=(18, 6), color=colors)
# ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
# plt.title('Diabete x Fasce di Reddito')
# plt.xlabel('Reddito')
# plt.xticks(rotation=0)
# plt.ylabel('Percentuale')
# for p in ax.containers:
#     ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
# plt.show()
#
# # Analizziamo il rapporto tra Salute Generale e incidenza del diabete
# data2['GenHlth'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
# plt.xticks(rotation=0)
# plt.ylabel('Popolazione')
# plt.xlabel('Livello di salute')
# plt.title('Distribuzione Salute Generale')
#
# tab = pd.crosstab(data2.GenHlth, data2.Diabetes_binary, normalize='index') * 100
# ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
# ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
# plt.title('Diabete x Salute Generale')
# plt.xlabel('Salute Generale')
# plt.xticks(rotation=0)
# plt.ylabel('Percentuale')
# for p in ax.containers:
#     ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
# plt.show()
#
# # Analizziamo il rapporto tra Salute Mentale e incidenza del diabete
# data2['MentHlth'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
# plt.ylabel('Popolazione')
# plt.xlabel('Giorni')
# plt.xticks(rotation=0)
# plt.title('Distribuzione Salute Mentale')
#
# tab = pd.crosstab(data2.MentHlth, data2.Diabetes_binary, normalize='index') * 100
# ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
# ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
# plt.title('Diabete x Salute Mentale')
# plt.xlabel('Giorni')
# plt.xticks(rotation=0)
# plt.ylabel('Percentuale')
# for p in ax.containers:
#     ax.bar_label(p, label_type='edge', labels=[f'{x:.1f}%' for x in p.datavalues])
# plt.show()
#
# # Crosstab tra diabete e Age
# data2['Age'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
# plt.ylabel('Popolazione')
# plt.xlabel('Fasce di etá')
# plt.xticks(rotation=0)
# plt.title('Distribuzione Age')
# plt.show()
#
# tab = pd.crosstab(data2.Age, data2.Diabetes_binary, normalize='index') * 100
# ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
# ax.legend(title='Diabete')
# plt.title('Distribuzione frequenze Diabete x Age')
# plt.xlabel('Fasce di etá')
# plt.xticks(rotation=0)
# plt.ylabel('Percentuale')
# for p in ax.containers:
#     ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
# plt.show()
#
# # Crosstab tra diabete e Salute fisica
# data2['PhysHlth'].value_counts().sort_values(ascending=True).plot(figsize=(10, 10), kind='bar', color='dodgerblue')
# plt.ylabel('Popolazione')
# plt.xlabel('Giorni')
# plt.xticks(rotation=0)
# plt.title('Distribuzione Problemi Fisici')
#
# tab = pd.crosstab(data2.PhysHlth, data2.Diabetes_binary, normalize='index') * 100
# ax = tab.plot(kind="bar", figsize=(15, 6), color=colors)
# ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
# plt.title('Diabete x Problemi fisici')
# plt.xlabel('Giorni')
# plt.xticks(rotation=0)
# plt.ylabel('Percentuale')
# for p in ax.containers:
#     ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
# plt.show()
#
# # Matrice correlazione
# plt.figure(figsize=(15, 15))
# sn.heatmap(data.corr(), square=True, annot=True, cmap='YlGnBu', fmt='.2f', robust=True)
# plt.title("Matrice Correlazione")
# plt.show()
#
# # Istogramma correlazione con Diabetes_Binary
# plt.figure(figsize=(20,15))
# corr = data.corr().sort_values(by='Diabetes_binary', ascending=False)
# corr = corr[corr.index != 'Diabetes_binary']
# corr['Diabetes_binary'].plot(kind='bar', color='dodgerblue')
# plt.xlabel('Features')
# plt.ylabel('Correlazione')
# plt.title('Correlazione con la variabile target')
# plt.axhline(y=0.05, linestyle='--', color='gray')
# plt.axhline(y=-0.05, linestyle='--', color='gray')
#
# plt.show()

# Vengono eliminate le features meno significative
X = data.drop(['Diabetes_binary', 'AnyHealthcare', 'NoDocbcCost', 'Smoker', 'Fruits', 'Veggies', 'Stroke', 'PhysActivity'],axis=1)
y = data['Diabetes_binary']

names = ["Decision Tree", 'Random Forest', 'Logistic Regression', 'Nearest Neighbors', "Naive Bayes", 'GradientBoost',
         'XGB', 'LGBM', 'ExtraTree', 'AdaBoost']

# classifiers = [
#     DecisionTreeClassifier(criterion='gini', max_depth=40, min_samples_leaf=2, max_features=None,
#                            min_samples_split=3,
#                            splitter='best'),
#     RandomForestClassifier(criterion='entropy', max_depth=50, max_features='log2', n_estimators=200,
#                            min_samples_leaf=1, min_samples_split=4,
#                            n_jobs=-1),
#     LogisticRegression(C=0.5, max_iter=300, class_weight='balanced', fit_intercept=True, penalty='l1',
#                        solver='liblinear', n_jobs=-1,
#                        tol=0.001),
#     KNeighborsClassifier(leaf_size=20, weights='distance', n_neighbors=4, n_jobs=-1),
#     GaussianNB(),
#     GradientBoostingClassifier(learning_rate=0.2, max_depth=5, n_estimators=150, max_features=None,
#                                min_samples_leaf=2, min_samples_split=2),
#     XGBClassifier(learning_rate=0.4, max_depth=4, min_child_weight=3, booster='gbtree', colsample_bytree=0.9,
#                   n_estimators=200),
#     LGBMClassifier(boosting_type='gbdt', learning_rate=0.3, max_depth=5, n_estimators=200,
#                    num_leaves=21, objective='binary'),
#     ExtraTreeClassifier(criterion='log_loss', max_features='sqrt'),
#     AdaBoostClassifier(learning_rate=1, n_estimators=200)
# ]  # Recall

# classifiers = [
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     LogisticRegression(),
#     KNeighborsClassifier(),
#     GaussianNB(),
#     GradientBoostingClassifier(),
#     XGBClassifier(),
#     LGBMClassifier(),
#     ExtraTreeClassifier(),
#     AdaBoostClassifier()
# ]

# classifiers = [
#     DecisionTreeClassifier(criterion='entropy', max_depth=40, min_samples_leaf=2, max_features=None,
#                            min_samples_split=1,
#                            splitter='best'),
#     RandomForestClassifier(criterion='entropy', max_depth=50, max_features='log2', n_estimators=200,
#                            min_samples_leaf=1, min_samples_split=4,
#                            n_jobs=-1),
#     LogisticRegression(C=1, max_iter=200, class_weight='balanced', fit_intercept=True, penalty='l2',
#                        solver='saga', n_jobs=-1,
#                        tol=0.001),
#     KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, weights='distance', n_neighbors=4, n_jobs=-1),
#     GaussianNB(),
#     GradientBoostingClassifier(learning_rate=0.2, max_depth=5, n_estimators=150, max_features=None,
#                                min_samples_leaf=2, min_samples_split=2),
#     XGBClassifier(learning_rate=0.3, max_depth=5, min_child_weight=1, booster='gbtree', colsample_bytree=0.8,
#                   n_estimators=200, subsample=1),
#     LGBMClassifier(boosting_type='gbdt', learning_rate=0.2, max_depth=7, n_estimators=200,
#                    num_leaves=41, objective='binary', subsample=0.8, colsample_bytree=0.8),
#     ExtraTreeClassifier(criterion='log_loss', max_features='sqrt', max_depth=50),
#     AdaBoostClassifier(learning_rate=1, n_estimators=200)
# ]  # Accuracy

classifiers = [
    DecisionTreeClassifier(criterion='gini', max_depth=30, min_samples_leaf=1, max_features=None,
                           min_samples_split=2,
                           splitter='best'),
    RandomForestClassifier(criterion='gini', max_depth=30, max_features='log2', n_estimators=200,
                           min_samples_leaf=1, min_samples_split=2,
                           n_jobs=-1),
    LogisticRegression(C=0.1, max_iter=200, class_weight='balanced', fit_intercept=False, penalty='l1',
                       solver='liblinear', n_jobs=-1,
                       tol=0.001),
    KNeighborsClassifier(leaf_size=30, weights='distance', n_neighbors=5, n_jobs=-1),
    GaussianNB(),
    GradientBoostingClassifier(learning_rate=0.05, max_depth=7, n_estimators=150, max_features='sqrt',
                               min_samples_leaf=1, min_samples_split=2),
    XGBClassifier(learning_rate=0.2, max_depth=4, min_child_weight=3, booster='gbtree', colsample_bytree=1,
                  n_estimators=75),
    LGBMClassifier(boosting_type='gbdt', learning_rate=0.05, max_depth=5, n_estimators=100,
                   num_leaves=41, objective='binary', subsample=0.8, colsample_bytree=1),
    ExtraTreeClassifier(criterion='log_loss', max_features=None),
    AdaBoostClassifier(learning_rate=1, n_estimators=200)
]  # Recall class 1

results_cv = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score", "Time"])
results_test = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score", "Time"])
results_class1 = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score", "Time"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Trovo gli indici delle prime righe di y_test con valore pari a 0 e 1
idx_0 = np.where(y_test == 0)[0][0]
idx_1 = np.where((y_test == 1) & (X_test['BMI'] > 40) & (X_test['HighBP'] == 1))[0][0]
# Seleziono le righe corrispondenti in X_test
rows = [X_test.iloc[idx_0], X_test.iloc[idx_1]]

fig, axs = plt.subplots(2, 2, figsize=(20, 20))
# Grafico per la classe 0
axs[0][0].bar(rows[0].index, rows[0].values)
axs[0][0].set_title('Valori delle feature per la classe 0')
axs[0][0].set_xticklabels(rows[0].index, rotation=90)
# Grafico per la classe 1
axs[1][0].bar(rows[1].index, rows[1].values)
axs[1][0].set_title('Valori delle feature per la classe 1')
axs[1][0].set_xticklabels(rows[1].index, rotation=90)

# # specifico le colonne non binarie da scalare
# non_binary_cols = ['Age', 'BMI', 'PhysHlth', 'GenHlth', 'Education', 'Income', 'MentHlth']
# # creo un oggetto scaler della classe RobustScaler
# scaler = RobustScaler()
# X_train[non_binary_cols] = scaler.fit_transform(X_train[non_binary_cols])
# X_test[non_binary_cols] = scaler.transform(X_test[non_binary_cols])

scaler = RobustScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

rows_sc = [X_test.iloc[idx_0], X_test.iloc[idx_1]]
axs[0][1].bar(rows_sc[0].index, rows_sc[0].values)
axs[0][1].set_title('Valori delle feature per la classe 0 dopo lo scaling')
axs[0][1].set_xticklabels(rows_sc[0].index, rotation=90)

# Visualizza la riga dopo lo scaling come barplot
axs[1][1].bar(rows_sc[1].index, rows_sc[1].values)
axs[1][1].set_title('Valori delle feature per la classe 1 dopo lo scaling')
axs[1][1].set_xticklabels(rows_sc[1].index, rotation=90)
plt.tight_layout()
plt.show()

# SMOTE
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

###########################################################################################

# Create a SHAP explainer
model = XGBClassifier(learning_rate=0.2, max_depth=4, min_child_weight=3, booster='gbtree', colsample_bytree=1,
                      n_estimators=75)
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for test set
shap_values = explainer.shap_values(X_test)
pred = model.predict(X_test.iloc[[idx_0]])
# Plot waterfall chart y=0
plt.figure()
shap.waterfall_plot(shap.Explanation(values=shap_values[idx_0],
                                     base_values=explainer.expected_value,
                                     data=rows[0],
                                     feature_names=X_test.columns.tolist()),
                    max_display=16, show=False)
plt.title(f"XGB - Waterfall plot per un NON Diabetico. Predizione: {pred[0]}")
plt.tight_layout()
plt.show()

plt.figure()
pred = model.predict(X_test.iloc[[idx_1]])
shap.waterfall_plot(shap.Explanation(values=shap_values[idx_1],
                                     base_values=explainer.expected_value,
                                     data=rows[1],
                                     feature_names=X_test.columns.tolist()),
                    max_display=16, show=False)
plt.title(f"XGB - Waterfall plot per un Diabetico. Predizione: {pred[0]}")
plt.tight_layout()
plt.show()

# Violin plot of SHAP values
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="violin", show=False)
plt.title("XGB - Violin plot")
plt.tight_layout()
plt.show()

# Bar plot of SHAP values
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("XGB - Bar plot")
plt.tight_layout()
plt.show()

###########################################################################################

model = LGBMClassifier(boosting_type='gbdt', learning_rate=0.05, max_depth=5, n_estimators=100,
                       num_leaves=41, objective='binary', subsample=0.8, colsample_bytree=1)
model.fit(X_train, y_train)
# Create a SHAP explainer
explainer = shap.TreeExplainer(model)
# Calculate SHAP values for test set
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("LGBM - Summary plot")
plt.tight_layout()
plt.show()

pred = model.predict(X_test.iloc[[idx_0]])
plt.figure()
# Genera un grafico waterfall plot per la classe 0
shap.waterfall_plot(shap.Explanation(values=shap_values[1][idx_0],
                                     base_values=explainer.expected_value[0],
                                     data=rows[0],
                                     feature_names=X_test.columns.tolist()),
                    max_display=16, show=False)
plt.title(f"LGBM - Waterfall plot per un NON Diabetico. Predizione: {pred[0]}")
plt.tight_layout()
plt.show()

pred = model.predict(X_test.iloc[[idx_1]])
plt.figure()
# Genera un grafico waterfall plot per la classe 1
shap.waterfall_plot(shap.Explanation(values=shap_values[1][idx_1],
                                     base_values=explainer.expected_value[1],
                                     data=rows[1],
                                     feature_names=X_test.columns.tolist()),
                    max_display=16, show=False)
plt.title(f"LGBM - Waterfall plot per un Diabetico. Predizione: {pred[0]}")
plt.tight_layout()
plt.show()

###########################################################################################

model = RandomForestClassifier(criterion='gini', max_depth=30, max_features='log2', n_estimators=200,
                               min_samples_leaf=1, min_samples_split=2, n_jobs=-1)
model.fit(X_train, y_train)
# Create a SHAP explainer
explainer = shap.TreeExplainer(model)
# Calculate SHAP values for test set
shap_values = explainer.shap_values(X_test[:350])

plt.figure()
shap.summary_plot(shap_values[:350], X_test[:350], show=False)
plt.title("RandomForest - Summary plot")
plt.tight_layout()
plt.show()

pred = model.predict(X_test.iloc[[idx_0]])
plt.figure()
# Genera un grafico waterfall plot per ogni classe presente nella variabile di interesse
shap.waterfall_plot(shap.Explanation(values=shap_values[1][idx_0],
                                     base_values=explainer.expected_value[0],
                                     data=rows[0],
                                     feature_names=X_test.columns.tolist()),
                    max_display=16, show=False)
plt.title(f"RF - Waterfall plot per un NON Diabetico. Predizione: {pred[0]}")
plt.tight_layout()
plt.show()

pred = model.predict(X_test.iloc[[idx_1]])
plt.figure()
# Genera un grafico waterfall plot per ogni classe presente nella variabile di interesse
shap.waterfall_plot(shap.Explanation(values=shap_values[1][idx_1],
                                     base_values=explainer.expected_value[1],
                                     data=rows[1],
                                     feature_names=X_test.columns.tolist()),
                    max_display=16, show=False)
plt.title(f"RF - Waterfall plot per un Diabetico. Predizione: {pred[0]}")
plt.tight_layout()
plt.show()

###########################################################################################

print("_____________________________________________________________________________")
cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
for name, clf in zip(names, classifiers):
    start = time.time()
    print('\n -> ' + name)

    # Valutazione delle performance utilizzando la cross-validation
    predictions = cross_val_predict(clf, X_train, y_train, cv=cv, n_jobs=-1, verbose=10)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_train, predictions)
    acc = round(accuracy_score(y_train, predictions), 3)

    print('\nAccuracy con cross-validation:', acc)
    print('Precision con cross-validation:', precision.mean())
    print('Recall con cross-validation:', recall.mean())
    print('F1 con cross-validation:', f1_score.mean())
    print('Supporto con cross-validation:', support)

    print('Report con cross-validation:\n', classification_report(y_train, predictions))

    # Plot della matrice di confusione come heat map
    sn.set(font_scale=1.4)  # Imposta la dimensione del font
    sn.heatmap(confusion_matrix(y_train, predictions), annot=True, annot_kws={"size": 16}, cmap='Blues',
               fmt="d")  # Annota ogni cella con il valore della matrice di confusione
    plt.title(f'Matrice di Confusione per {name}')
    plt.xlabel('Predetto')
    plt.ylabel('Effettivo')
    plt.tight_layout()
    plt.show()

    results_cv = pd.concat([results_cv, pd.DataFrame({"Classifier": name,
                                                      "Accuracy": round(acc, 3),
                                                      "Precision": round(precision.mean(), 3),
                                                      "Recall": round(recall.mean(), 3),
                                                      "F1-Score": round(f1_score.mean(), 3),
                                                      "Time": round(time.time() - start, 3)},
                                                     index=[0])], ignore_index=True)

    results_class1 = pd.concat([results_class1, pd.DataFrame({"Classifier": name,
                                                              "Accuracy": round(acc, 3),
                                                              "Precision": round(precision[1], 3),
                                                              "Recall": round(recall[1], 3),
                                                              "F1-Score": round(f1_score[1], 3),
                                                              "Time": round(time.time() - start, 3)},
                                                             index=[0])], ignore_index=True)
    print("_____________________________________________________________________________")

    clf.fit(X_train.values, y_train.values)
    # Predizione sui dati di test
    y_pred = clf.predict(X_test.values)
    acc = round(accuracy_score(y_test, y_pred), 3)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred)
    # Valutazione delle performance sui dati di test
    print('Accuracy sul test set:', acc)
    print('Precision test:', precision.mean())
    print('Recall test:', recall.mean())
    print('F1 test:', f1_score.mean())
    print('Supporto test:', support)
    print('Report sul test set:\n', classification_report(y_test, y_pred))
    print('Matrice di confusione sul test set:\n', confusion_matrix(y_test, y_pred))

    results_test = pd.concat([results_test, pd.DataFrame({"Classifier": name,
                                                          "Accuracy": round(acc, 2),
                                                          "Precision": round(precision.mean(), 2),
                                                          "Recall": round(recall.mean(), 2),
                                                          "F1-Score": round(f1_score.mean(), 2),
                                                          "Time": round(time.time() - start, 2)},
                                                         index=[0])], ignore_index=True)

    print("_____________________________________________________________________________")
    print("Tempo: ", round(time.time() - start, 3))

    # # LIME
    # explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values,
    #                                               feature_names=X_train.columns.tolist(),
    #                                               mode='classification',
    #                                               discretize_continuous=True,
    #                                               discretizer='decile')
    #
    # for row in rows_sc:
    #     # Creo l'explainer
    #     exp = explainer.explain_instance(row.values, clf.predict_proba, num_features=12)
    #
    #     # Recupero l'esito della predizione e la classe vera
    #     # pred_label = clf.predict([row])[0]
    #     true_label = y_test.iloc[idx_1] if row.equals(X_test.iloc[idx_1]) else y_test.iloc[idx_0]
    #     # pred_label = 'SI Diabete' if pred_label == 1 else 'NO Diabete'
    #     true_label = 'SI Diabete' if true_label == 1 else 'NO Diabete'
    #
    #     # Visualizzazione come barplot con il nome del modello e la classe predetta e vera
    #     fig = exp.as_pyplot_figure()
    #     fig.set_size_inches(20, 6)
    #     plt.title(
    #         f' Classificatore: {name} |'
    #         f' Classe vera: {true_label} |'
    #         f' Probabilitá NO: {exp.predict_proba[0]:.2f} | '
    #         f' Probabilitá SI: {exp.predict_proba[1]:.2f}')
    #     plt.show()
    #     # exp.save_to_file(f'{name}_{true_label}_{pred_label}.html')

# Stampa dei risultati in una tabella
print()
print("_____________________________________________________________________________")
print("_____________________________ RISULTATI(CV) _________________________________")
print(results_cv)
print("_____________________________________________________________________________\n")
results_cv.to_csv("results_cv.csv")
print()
print("_____________________________________________________________________________")
print("___________________________ RISULTATI(Test) _________________________________")
print(results_test)
print("_____________________________________________________________________________\n")
results_test.to_csv("results_test.csv")

print()
print("_____________________________________________________________________________")
print("___________________________ RISULTATI(SI) _________________________________")
print(results_class1)
print("_____________________________________________________________________________\n")
results_class1.to_csv("results_test.csv")

print("Tempo di esecuzione --- %s secondi ---" % (time.time() - start_time))

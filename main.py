import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from lime import lime_tabular
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold, \
    GridSearchCV
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

names = ["Decision Tree", 'Random Forest', 'Logistic Regression', 'Nearest Neighbors', "Naive Bayes", 'GradientBoost',
         'XGB', 'LGBM', 'CatBoost', 'AdaBoost']

# classifiers = [
#     DecisionTreeClassifier(criterion='gini', max_depth=15, splitter='best', min_samples_leaf=2),
#     RandomForestClassifier(criterion='gini', max_depth=20, max_features='log2', n_estimators=200, min_samples_leaf=1, min_samples_split=2, random_state=42),
#     LogisticRegression(C=0.25, penalty='l1', solver='liblinear', random_state=42),
#     KNeighborsClassifier(algorithm='kd_tree', n_neighbors=5, weights='distance'),
#     GaussianNB(var_smoothing=1e-09),
#     GradientBoostingClassifier(learning_rate=0.5, max_depth=5, n_estimators=200, criterion='friedman_mse', max_features='sqrt', random_state=42),
#     XGBClassifier(eval_metric='error', learning_rate=0.1, max_depth=5, n_estimators=200, min_child_weight=10, random_state=42),
#     LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, max_depth=5, n_estimators=200, num_leaves=12, objective='binary'),
#     CatBoostClassifier(depth=6, iterations=500, leaf_estimation_iterations=10, logging_level='Silent', loss_function='Logloss', random_seed=42),
#     AdaBoostClassifier(estimator=DecisionTreeClassifier(), algorithm='SAMME.R', learning_rate=0.5, n_estimators=200)
# ]

classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(random_state=42),
    LogisticRegression(random_state=42),
    KNeighborsClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(random_state=42),
    XGBClassifier(random_state=42),
    LGBMClassifier(),
    CatBoostClassifier(random_state=42, verbose=False),
    AdaBoostClassifier(estimator=DecisionTreeClassifier())
] # Recall

results_cv = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score", "Time"])
results_test = pd.DataFrame(columns=["Classifier", "Accuracy", "Precision", "Recall", "F1-Score", "Time"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# SMOTE
smote = SMOTE(sampling_strategy=1, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
print("_____________________________________________________________________________")

for name, clf in zip(names, classifiers):
    start = time.time()
    print('\n -> ' + name)
    clf.fit(X_train, y_train)

    predictions = cross_val_predict(clf, X_train, y_train, cv=10, n_jobs=-1)
    # Valutazione delle performance utilizzando la predizione con cross-validation
    print('\nAccuracy con cross-validation:', round(accuracy_score(y_train, predictions), 3))
    print('Report con cross-validation:\n', classification_report(y_train, predictions))
    print('Matrice di confusione con cross-validation:\n', confusion_matrix(y_train, predictions))
    report = classification_report(y_train, predictions, output_dict=True)
    results_cv = pd.concat([results_cv, pd.DataFrame({"Classifier": name,
                                                      "Accuracy": round(report['accuracy'], 3),
                                                      "Precision": round(report['macro avg']['precision'], 3),
                                                      "Recall": round(report['macro avg']['recall'], 3),
                                                      "F1-Score": round(report['macro avg']['f1-score'], 3),
                                                      "Time": round(time.time() - start, 3)},
                                                     index=[0])], ignore_index=True)
    print("_____________________________________________________________________________")

    # Predizione sui dati di test
    y_pred = clf.predict(X_test)
    # Valutazione delle performance sui dati di test
    print('Accuracy sul test set:', round(accuracy_score(y_test, y_pred), 3))
    print('Report sul test set:\n', classification_report(y_test, y_pred))
    print('Matrice di confusione sul test set:\n', confusion_matrix(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    results_test = pd.concat([results_test, pd.DataFrame({"Classifier": name,
                                                          "Accuracy": round(report['accuracy'], 3),
                                                          "Precision": round(report['macro avg']['precision'], 3),
                                                          "Recall": round(report['macro avg']['recall'], 3),
                                                          "F1-Score": round(report['macro avg']['f1-score'], 3),
                                                          "Time": round(time.time() - start, 3)},
                                                         index=[0])], ignore_index=True)
    print("_____________________________________________________________________________")
    print("Tempo: ", round(time.time() - start, 3))

    # LIME
    # Seleziono la prima riga con classe pari a 1
    test_instance = X_test.loc[y_test].iloc[0]

    explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values,
                                                  feature_names=X_train.columns.values,
                                                  class_names=[': NO Diabete', ': SI Diabete'],
                                                  mode='classification')

    exp = explainer.explain_instance(test_instance.values, clf.predict_proba, num_features=len(X_train.columns))

    # Recupero l'esito della predizione
    pred_label = clf.predict([test_instance])[0]
    pred_label = 'SI diabete' if pred_label == 0 else 'NO diabete'
    coef = pd.DataFrame(exp.as_list())
    # print(coef)
    coef_sum = coef[1].sum()
    # print("Somma dei coefficienti: ", round(coef_sum,2))

    # Visualizzazione come barplot con il nome del modello e la classe predetta
    fig = exp.as_pyplot_figure()
    fig.suptitle(
        f'     Classificatore: {name}   |   Classe predetta: {pred_label}   |   Valore totale: {round(coef_sum, 2)}')
    fig.set_size_inches(20, 6)
    plt.show()

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

print("Tempo di esecuzione --- %s secondi ---" % (time.time() - start_time))

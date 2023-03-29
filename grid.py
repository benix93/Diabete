import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
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


# Vengono eliminate le features meno significative
X = data.drop(
    ['Diabetes_binary', 'AnyHealthcare', 'NoDocbcCost', 'Fruits', 'Veggies', 'Sex', 'Smoker'],
    axis=1)
y = data['Diabetes_binary']

names = ["Decision Tree", 'Random Forest', 'Logistic Regression', 'Nearest Neighbors', "Naive Bayes", 'GradientBoost',
         'XGB', 'LGBM', 'ExtraTrees', 'AdaBoost']

classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    KNeighborsClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(),
    XGBClassifier(),
    LGBMClassifier(),
    ExtraTreeClassifier(),
    AdaBoostClassifier()
]

params = [
    {
        'criterion': ['gini', 'entropy'],  # gini
        'splitter': ['best', 'random'],  # best
        'max_depth': [25, 30, 35, 40, 45, None],  # none
        'min_samples_split': [2, 3],  # 2
        'min_samples_leaf': [1, 2, 3, 5, 10],  # 1
        'max_features': ['sqrt', 'log2', None],  # None= n_features
        'max_leaf_nodes': [None, 50, 100, 200]
    },  # Decision Tree

    {
        'n_estimators': [100, 150, 200],  # 100
        'max_depth': [35, 40, 50, None],  # None
        'criterion': ['gini', 'entropy'],  # gini
        'min_samples_leaf': [1, 2, 3],  # 1
        'min_samples_split': [2, 3, 4, 10],  # 2
        'max_features': ['sqrt', 'log2'],  # sqrt
        'max_leaf_nodes': [None, 50, 100, 200],
        'max_samples': [None, 0.5, 0.7, 0.9],
        'n_jobs': [-1],
    },  # Random Forest

    {
        'penalty': ['l1', 'l2', 'elasticnet'],  # l2
        'C': [0.1, 0.5, 1, 5, 10],  # 1
        'fit_intercept': [True, False],
        'solver': ['lbfgs', 'liblinear', 'saga'],  # lbfgs
        'max_iter': [100, 200, 300],  # 100
        'n_jobs': [-1],
        'class_weight': [None, 'balanced'],  # None
        'tol': [0.0001, 0.001]  # 0.0001
    },  # Logistic Regression

    {
        'n_neighbors': range(4, 6),  # 5
        'weights': ['uniform', 'distance'],  # uniform
        'algorithm': ['auto', 'ball_tree'],  # auto
        'leaf_size': [20, 30, 40],  # 30
        'n_jobs': [-1]
    },  # KNN

    {},  # Naive Bayes

    {
        "learning_rate": [0.05, 0.1, 0.2],  # 0.1
        "max_depth": [3, 5, 7],  # 3
        "max_features": [None, "log2", "sqrt"],  # None
        "n_estimators": [50, 100, 150],  # 100
        'min_samples_split': [2, 3],  # 2
        'min_samples_leaf': [1, 2],  # 1
    },  # GradientBoost

    {
        'booster': ['gbtree'],  # default gbtree
        "n_estimators": [75, 100, 150, 200],  # 100
        'min_child_weight': [1, 2, 3],  # default 1
        'max_depth': [4, 5, 6, 7],  # default 6
        'learning_rate': [0.2, 0.3, 0.4, 0.5],  # default 0.3
        'colsample_bytree': [0.8, 0.9, 1],  # 1
        'subsample': [0.5, 0.8, 1],  # default 1
    },  # XGB

    {
        'learning_rate': [0.05, 0.1, 0.2, 0.3],  # 0.1
        'max_depth': [3, 4, 5, 7, 9, None],  # -1 = None
        'n_estimators': [50, 100, 150, 200],  # 100
        'boosting_type': ['gbdt', 'dart', 'rf'],  # gbdt
        'objective': ['binary', None],  # None = regression
        'num_leaves': [21, 31, 41],  # default 31
        'subsample': [0.5, 0.8, 1],  # default 1
    },  # LGBM

    {
        'max_depth': [15, 25, 35, 40, 50, None],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3, 4],
        'max_features': ['sqrt', 'log2'],
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
        'min_impurity_decrease': [0.0, 0.1, 0.2],
        'ccp_alpha': [0.0, 0.1, 0.2],
    },  # ExtraTree

    {
        'n_estimators': [50, 100, 200],  # 50
        'learning_rate': [0.1, 0.8, 0.9, 1],  # 1
        'algorithm': ['SAMME.R', 'SAMME'],  # SAMME.R
    }  # AdaBoost
]

param_results = pd.DataFrame(columns=["Classifier", "Best Parameters"])
results = pd.DataFrame(columns=["Classifier", "Recall", "Time"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# specifico le colonne non binarie da scalare
non_binary_cols = ['Age', 'BMI', 'PhysHlth', 'GenHlth', 'Education', 'Income', 'MentHlth']
# creo un oggetto scaler della classe RobustScaler
scaler = RobustScaler()
X_train[non_binary_cols] = scaler.fit_transform(X_train[non_binary_cols])
X_test[non_binary_cols] = scaler.transform(X_test[non_binary_cols])

# scaler = RobustScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
# X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# SMOTE
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

scoring = make_scorer(recall_score, pos_label=1)
cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
print("_____________________________________________________________________________")
for name, clf, param in zip(names, classifiers, params):
    start = time.time()
    print('\n -> ' + name)
    clf = GridSearchCV(clf, param_grid=param, cv=cv, n_jobs=-1, scoring=scoring)
    clf.fit(X_train.values, y_train.values)

    print(f'Migliori parametri per {name}: {clf.best_params_}')
    print(f'Miglior score per {name}: {clf.best_score_}')

    param_results = pd.concat([param_results, pd.DataFrame({"Classifier": name,
                                                            "Best Parameters": [clf.best_params_]},
                                                           index=[0])], ignore_index=True)

    results = pd.concat([results, pd.DataFrame({"Classifier": name,
                                                "Recall": round(clf.best_score_, 3),
                                                "Time": round(time.time() - start, 3)},
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

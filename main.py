import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split

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

data2 = data.copy()
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
data2['GenHlth'] = data2['GenHlth'].replace(
    {1: 'Eccellente', 2: 'Molto Buona', 3: 'Buona', 4: 'Discreta', 5: 'Pessima'})

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


data2['BMI'] = data2["BMI"].apply(bmi_map)
data2['BMI'] = data2['BMI'].replace({1: '18.5-', 2: '18.5-24.9', 3: '25-29.9', 4: '30-34.9', 5: '35-39.9', 6: '40+'})
print(data2['BMI'].value_counts().unique())

# Valutiamo le percentuali con cui si presentano le features binarie
fig, axes = plt.subplots(4, 3, figsize=(15, 15))
for i, ax in enumerate(axes.flatten()):
    if i < len(['Diabetes_binary', 'Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
                'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke']):
        col = ['Diabetes_binary', 'Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
               'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke'][i]
        labels = data2[col].unique()
        explode = (0.02, 0.02)
        ax.pie(data2[col].value_counts(), labels=labels, autopct='%.2f', explode=explode)
        ax.set_title(col)
plt.show()

fig, axes = plt.subplots(4, 3, figsize=(15, 15))
for i, ax in enumerate(axes.flatten()):
    if i < len(['Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
                'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke']):
        col = ['Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
               'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke'][i]
        tab = pd.crosstab(data2[col], data2.Diabetes_binary, normalize='index') * 100
        ax = tab.plot(kind="bar", stacked=True, figsize=(25, 25), ax=ax)
        ax.set_title(f'{col} x Diabete')
        ax.set_xlabel(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel('Percentuale')
        ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1.10, 1))
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    if i < len(['Smoker', 'AnyHealthcare', 'NoDocbcCost']):
        col = ['Smoker', 'AnyHealthcare', 'NoDocbcCost'][i]
        tab = pd.crosstab(data2[col], data2.Diabetes_binary, normalize='index') * 100
        ax = tab.plot(kind="bar", stacked=True, figsize=(15, 15), ax=ax)
        ax.set_title(f'{col} x Diabete')
        ax.set_xlabel(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel('Percentuale')
        ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1.10, 1))
plt.show()


# Analizziamo il rapporto tra BMI e incidenza del diabete
tab = pd.crosstab(data2.BMI, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(20, 6))
plt.title('Distribuzione frequenze Diabete x BMI')
plt.xlabel('BMI')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Crosstab tra diabete e Age
labels = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79',
          '80+']
tab = pd.crosstab(data2.Age, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(20, 6))
plt.title('Distribuzione frequenze Diabete x Age')
plt.xlabel('Age')
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Istruzione e incidenza del Diabete
tab = pd.crosstab(data2.Education, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(20, 6))
plt.title('Distribuzione frequenze Diabete x Istruzione')
plt.xlabel('Istruzione')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Reddito e incidenza del diabete
tab = pd.crosstab(data2.Income, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(20, 6))
plt.title('Distribuzione frequenze Diabete x Fasce di Reddito')
plt.xlabel('Reddito')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra GenHealth e incidenza del diabete
tab = pd.crosstab(data2.GenHlth, data2.Diabetes_binary, normalize='index') * 100
ax = tab.plot(kind="bar", figsize=(20, 6))
plt.title('Distribuzione frequenze Diabete x Salute Generale')
plt.xlabel('Salute Generale')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()


# Matrice correlazione
plt.figure(figsize=(20, 15))
sn.heatmap(data.corr(), annot=True, cmap='YlGnBu')
plt.title("Matrice Correlazione")
plt.show()

# Istogramma correlazione con Diabetes_Binary
plt.figure(figsize=(15, 15))
corr = data.corr().sort_values(by='Diabetes_binary', ascending=False)
corr = corr[corr.index != 'Diabetes_binary']
corr['Diabetes_binary'].plot(kind='bar', color='orange')
plt.xlabel('Features')
plt.title('Correlazione con Diabete')
plt.show()


# plt.figure(figsize=(10,10))
# corr = data.corr().sort_values(by='BMI', ascending=False)
# corr['BMI'].plot(kind='bar', color='orange')
# plt.xlabel('Features')
# plt.ylabel('Correlation with BMI')
# plt.title('Correlation of Features with BMI')
# plt.show()

# Analizziamo il rapporto tra educazione e BMI
# tab = pd.crosstab(data2.Education, data2.BMI, normalize='index')*100
# ax = tab.plot(kind="bar", figsize=(20, 6))
# plt.title('Distribuzione frequenze BMI x Education')
# plt.xlabel('Education')
# plt.xticks(rotation=0)
# plt.ylabel('Percentuale')
# for p in ax.containers:
#     ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
# plt.show()

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
data2['BMI'] = data2['BMI'].replace(
    {1: '18.5 o meno', 2: '18.5-24.9', 3: '25-29.9', 4: '30-34.9', 5: '35-39.9', 6: '40+'})
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


fig, axes = plt.subplots(4, 3, figsize=(15, 20))
for i, ax in enumerate(axes.flatten()):
    if i < len(['Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
                'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke']):
        col = ['Sex', 'HighBP', 'HighChol', 'CholCheck', 'DiffWalk', 'AnyHealthcare',
               'HvyAlcoholConsump', 'Veggies', 'Fruits', 'PhysActivity', 'HeartDiseaseorAttack', 'Stroke'][i]
        tab = pd.crosstab(data2[col], data2.Diabetes_binary, normalize='index') * 100
        colors = ['dodgerblue', 'crimson']
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
        colors = ['dodgerblue', 'crimson']
        ax = tab.plot(kind="bar", stacked=True, figsize=(18, 6), color=colors, ax=ax)
        ax.set_title(f'{col} x Diabete')
        ax.set_xlabel(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel('Percentuale')
        ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1.10, 1))
plt.show()

# Analizziamo il rapporto tra BMI e incidenza del diabete
tab = pd.crosstab(data2.BMI, data2.Diabetes_binary, normalize='index') * 100
colors = ['dodgerblue', 'crimson']
ax = tab.plot(kind="bar", figsize=(20, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x BMI')
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
colors = ['dodgerblue', 'crimson']
ax = tab.plot(kind="bar", figsize=(20, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Age')
plt.xlabel('Age')
plt.gca().set_xticklabels(labels)
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Istruzione e incidenza del Diabete
labels = ['1-Nessuna', '2-Elementari', '3-Medie', '4-Superiori', '5-Triennale', '6-Magistrale']
tab = pd.crosstab(data2.Education, data2.Diabetes_binary, normalize='index') * 100
colors = ['dodgerblue', 'crimson']
ax = tab.plot(kind="bar", figsize=(20, 6), color=colors)
ax.set_xticklabels(labels)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Istruzione')
plt.xlabel('Istruzione')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Reddito e incidenza del diabete
labels = ['10k o meno', '10k-15k', '15-20k', '20-25k', '25-35k', '35-50k', '50-75k', '75k+']
tab = pd.crosstab(data2.Income, data2.Diabetes_binary, normalize='index') * 100
colors = ['dodgerblue', 'crimson']
ax = tab.plot(kind="bar", figsize=(20, 6), color=colors)
ax.set_xticklabels(labels)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Fasce di Reddito')
plt.xlabel('Reddito')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Salute Generale e incidenza del diabete
labels = ['Eccellente', 'Molto buona', 'Buona', 'Discreta', 'Pessima']
tab = pd.crosstab(data2.GenHlth, data2.Diabetes_binary, normalize='index') * 100
colors = ['dodgerblue', 'crimson']
ax = tab.plot(kind="bar", figsize=(20, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
ax.set_xticklabels(labels)
plt.title('Diabete x Salute Generale')
plt.xlabel('Salute Generale')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Salute Fisica e incidenza del diabete
tab = pd.crosstab(data2.PhysHlth, data2.Diabetes_binary, normalize='index') * 100
colors = ['dodgerblue', 'crimson']
ax = tab.plot(kind="bar", figsize=(25, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Problemi Fisici')
plt.xlabel('Giorni')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Analizziamo il rapporto tra Salute Mentale e incidenza del diabete
tab = pd.crosstab(data2.MentHlth, data2.Diabetes_binary, normalize='index') * 100
colors = ['dodgerblue', 'crimson']
ax = tab.plot(kind="bar", figsize=(25, 6), color=colors)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Salute Mentale')
plt.xlabel('Giorni')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Riduciamo il range delle eta' e dei problemi fisici0
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


data2['Age'] = data2['Age'].apply(age_map)
print(data2['Age'].value_counts().unique())


def physHlth_map(x):
    if x <= 5:
        return 1
    elif 6 <= x <= 10:
        return 2
    elif 11 <= x <= 15:
        return 3
    elif 16 <= x <= 20:
        return 4
    elif 21 <= x <= 25:
        return 5
    else:
        return 6


data2['PhysHlth'] = data2['PhysHlth'].apply(age_map)
print(data2['PhysHlth'].value_counts().unique())

# Crosstab tra diabete e Age aggiornata
labels = ['18-34', '35-44', '45-54', '55-64', '65-74', '75+']
tab = pd.crosstab(data2.Age, data2.Diabetes_binary, normalize='index') * 100
colors = ['dodgerblue', 'crimson']
ax = tab.plot(kind="bar", figsize=(20, 6), color=colors)
ax.set_xticklabels(labels)
ax.legend(title='Diabete')
plt.title('Distribuzione frequenze Diabete x Age 2.0')
plt.xlabel('Age')
plt.xticks(rotation=0)
plt.ylabel('Percentuale')
for p in ax.containers:
    ax.bar_label(p, label_type='edge', labels=[f'{x:.2f}%' for x in p.datavalues])
plt.show()

# Crosstab tra diabete e Salute fisica aggiornata
labels = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30']
tab = pd.crosstab(data2.PhysHlth, data2.Diabetes_binary, normalize='index') * 100
colors = ['dodgerblue', 'crimson']
ax = tab.plot(kind="bar", figsize=(20, 6), color=colors)
ax.set_xticklabels(labels)
ax.legend(title='Diabete', loc='upper right', bbox_to_anchor=(1, 1))
plt.title('Diabete x Problemi fisici 2.0')
plt.xlabel('Giorni')
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
corr['Diabetes_binary'].plot(kind='bar', color='salmon')
plt.xlabel('Features')
plt.title('Correlazione con Diabete')
plt.axhline(y=0.05, linestyle='--', color='gray')
plt.axhline(y=-0.05, linestyle='--', color='gray')
plt.show()

data['BMI'] = data["BMI"].apply(bmi_map)
data['Age'] = data['Age'].apply(age_map)
data['PhysHlth'] = data['PhysHlth'].apply(age_map)
print("_____________________________________________________________________________")
print(data['BMI'].value_counts().unique())
print(data['Age'].value_counts().unique())
print(data['PhysHlth'].value_counts().unique())


# Modelli

































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

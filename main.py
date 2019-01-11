import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pickle as pk

print('****************************************************')
print('Lecture des données')
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
tempData = data

input("prompt")
print('****************************************************')
print('Résumé des données')
# resume des datas
print(tempData.head())
# Colonnes
print(tempData.columns)

input("prompt")
#normaliser les noms de colonnes
#               0              1            2     3       4       5        6        7         8        9      10         11
colnames = ['passengerId', 'survived', 'class', 'name', 'sex', 'age', 'sibling', 'parch', 'ticket', 'fare', 'cabin', 'embarked']
tempData.columns = colnames
test_data.columns = ['passengerId', 'class', 'name',
                                'sex', 'age', 'sibling', 'parch', 'ticket', 'fare', 'cabin', 'embarked']

#extraction du y d'entrainement
y = tempData.survived

#columns id -- no magic number
passId = 0
survivedId = 1
classId = 2
nameId = 3
sexId = 4
ageId = 5
siblingId = 6
parchId = 7
ticketId = 8
fareId = 9
cabinId = 10
embarkedId = 11
numSexId = 12
numEmbarkedId = 13

# ********************************** EXPLORATION DES DONNEES *****************************************
print('****************************************************')
print('Statistique des données originales')
#Statistique de base
print(tempData.describe())
#
#      PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
# count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
# mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
# std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
# min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
# 25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
# 50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
# 75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
# max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

input("prompt")
#Correlation
print('****************************************************')
print('Matrice de corrélation')
print(tempData.corr())
# Pclass et fare on la plus forte correlation , par contre il manque des colonnes qui sont des categories
#              PassengerId  Survived    Pclass       Age     SibSp     Parch      Fare
# PassengerId     1.000000 -0.005007 -0.035144  0.036847 -0.057527 -0.001652  0.012658
# Survived       -0.005007  1.000000 -0.338481 -0.077221 -0.035322  0.081629  0.257307
# Pclass         -0.035144 -0.338481  1.000000 -0.369226  0.083081  0.018443 -0.549500
# Age             0.036847 -0.077221 -0.369226  1.000000 -0.308247 -0.189119  0.096067
# SibSp          -0.057527 -0.035322  0.083081 -0.308247  1.000000  0.414838  0.159651
# Parch          -0.001652  0.081629  0.018443 -0.189119  0.414838  1.000000  0.216225
# Fare            0.012658  0.257307 -0.549500  0.096067  0.159651  0.216225  1.000000

input("prompt")
print('****************************************************')
print('Heatmap des données originales')
ax = sn.heatmap(tempData.corr())
plt.show()

input("prompt")
print('****************************************************')
print('Histogramme des données originales')
# Histogramme
data.hist()
#plt.savefig('histogramme_1.png')
plt.show()

#histogramme Survived
#femme vs homme
sn.barplot(x=data.sex, y=data.survived, palette="rocket")
plt.show()

# femme homme survived dead
sn.countplot(x="sex", hue="survived", data=tempData)
plt.show()

input("prompt")
print('****************************************************')
print('Prétraitement des données originales')
#***************************************PRETRAITEMENT ************************************************
#Combien de NAN dans les features significative******************************************************
print('Traitement des NAN')
print('Nombre de  NAN : ', tempData.isna().sum())
print('Nombre de  NAN : ', test_data.isna().sum())

input("prompt")
print('****************************************************')
print('Conversion des valeurs qualitatives en numérique')
print('Conversion de la colonne Sexe')
#Conversion numerique de certain champs categorie
#sexe
tempData['num_sex'] = tempData.sex.replace({'male': 0, 'female': 1})
test_data['num_sex'] = test_data.sex.replace({'male': 0, 'female': 1})

print('Conversion de la colonne Embarked')
#Port d'embarcation
tempData['num_embarked'] = tempData.embarked.replace({'C': 0, 'Q': 1, 'S': 2})
test_data['num_embarked'] = test_data.embarked.replace({'C': 0, 'Q': 1, 'S': 2})

# corriger les  NAN **********************************************************************************
# pour lage on met a zero  ce qui n'aura pas vraiment de valeur, si on utilisait la moyenne qui est de 30ans, il est possible que je donne 30 a une vielle personne de 70 ou un enfant de 5 ans ce qui fait pas de sens du tout
# apres les test, on pourrait decider de juste ignorer les individu dont l'age nest pas presente si cela cause trop de probleme
print('****************************************************')
print('Correction des NAN de AGE, remplace par des 0')
tempData.age = tempData.age.fillna(0).values
test_data.age = test_data.age.fillna(0).values

#pour l'embarcation il y en a seulement que deux nous remplaceron par -1
tempData.num_embarked = tempData.num_embarked.fillna(-1).values
test_data.num_embarked = test_data.num_embarked.fillna(-1).values
test_data.fare = test_data.fare.fillna(-1).values
#pour le reste acutellement cela ne semble pas interessant . PAr exemple. le nom . difficile de convertir en donn/e numerique
# Cabine est aussi difficle et il y a beaucoup de NAN

#avec une nouvelle correlation on constate que le sexe est la donnee qui a la plus forte correlation .
# La class et la gare dembargquement  suit de pres avec une correlation negative
coefficients = tempData.corr()
print('Nouveau coefficient suite au correction ', coefficients)
print('Nombre de  NAN : ', tempData.isna().sum())

input("prompt")
print('****************************************************')
print('Heatmap des données suite au prétraitement')
plt.figure()
g = sn.heatmap(coefficients, vmin=-1, vmax=1)
plt.show()

input("prompt")
print('****************************************************')
print('Split des données')
# print(tempData)
x_train, x_test, y_train, y_test = train_test_split(
    tempData.iloc[:, [classId, ageId, siblingId, parchId, fareId, numSexId, numEmbarkedId]], y, random_state=42, test_size=0.20)
# clf.fit(tempData.iloc[:, [classId, ageId, siblingId,
#                           parchId, fareId, numSexId, numEmbarkedId]], y)
# pred = clf.predict(test_data.iloc[:, [1, 4, 5, 6, 8, 11, 12]])

input("prompt")
print('Création du modèle - Arbre de décision')
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

input("prompt")
print('Prédiction avec les données de test')
predict = clf.predict(x_test)

input("prompt")
print('Test de la précision')
print(accuracy_score(y_test, predict))

input("prompt")
print('Sauvegarde du modèle pour déploiement')
f = open('tree.txt', 'wb')
pk.dump(clf, f)

# print(x_train)
#Visualiser les patterns avec le clustering
#doit prendre le numerique seulement  Survived, age, num_sexe, pclass
# print(tempData)
# kmeanData = tempData.iloc[:, [1,2,5,12]]
# t = kmeanData.age.fillna(0)
# kmeanData.age = t.values

# kmeans = KMeans(n_clusters=3).fit(kmeanData)
# print(kmeans.labels_)







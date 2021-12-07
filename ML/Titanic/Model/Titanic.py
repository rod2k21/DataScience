# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:50:22 2021
@title: Predicción de supervivencia en el Titanic
@author: Rodrigo Rodríguez
"""
###########################LIBRERÍAS A UTILIZAR#############################
##Procesamiento de Datos
import numpy as np
import pandas as pd
import os
from pathlib import Path

##Algoritmos de ML

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier




#############################IMPORTANDO DATA################################
print(os.environ['USERPROFILE'])
data_folder = Path("/GitHub/DataScience/ML/Titanic/Dataset/")
df_test = pd.read_csv(data_folder / "test.csv")
df_train = pd.read_csv(data_folder /"train.csv")

print(df_test.head())
print(df_train.head())

###########################ENTENDIENDO LA DATA##############################
#Verificar la cantidad de Datos en los DataSet
print('Cantidad de Datos:')
print(df_test.shape)
print(df_train.shape)

#Verificar el tipo de Datos en ambos DataSet
print('Tipos de Datos:')
print(df_test.info())
print(df_train.info())

#Verificar Datos faltantes
print('Datos faltantes:')
print(pd.isnull(df_test).sum())
print(pd.isnull(df_train).sum())

#Verificar las estadísticas de cada DataSet
print('Estadísticas del DataSet:')
print(df_test.describe())
print(df_train.describe())


#############################PROCESANDO LA DATA#############################
#Cambiar datos de sexo a números
df_test['Sex'].replace(['female','male'],[0,1],inplace=True)
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)

#Cambiar datos de embarque a números
df_test['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)
df_train['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)

#Reemplazar los Datos faltantes en la edad por la media de esta columna
print(df_test['Age'].mean())
print(df_train['Age'].mean())
promedio = 30
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)
df_train['Age'] = df_train['Age'].replace(np.nan, promedio)

#Crear grupos por rango de edad
#Grupos: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100
bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7'] 
df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)

#Eliminar columna de "Cabin" ya que tiene muchos datos perdidos
df_test.drop(['Cabin'], axis = 1, inplace = True)
df_train.drop(['Cabin'], axis = 1, inplace = True)

#Eliminar columnas que no se consideran necesarias para el análisis
df_test = df_test.drop(['Name', 'Ticket', 'WikiId', 'Name_wiki', 'Age_wiki', 'Hometown', 'Boarded', 'Destination', 'Lifeboat', 'Body', 'Class'], axis = 1)
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'WikiId', 'Name_wiki', 'Age_wiki', 'Hometown', 'Boarded', 'Destination', 'Lifeboat', 'Body', 'Class'], axis = 1)

#Eliminar filas con datos perdidos
df_test.dropna(axis=0, how='any', inplace = True)
df_train.dropna(axis=0, how='any', inplace = True)

#Verificar los Datos
print(pd.isnull(df_test).sum())
print(pd.isnull(df_train).sum())

print(df_test.shape)
print(df_train.shape)

print(df_test.head())
print(df_train.head())
##########################APLICANDO ALGORITMOS DE ML########################
#Armando Modelos

#Separar el target con la infromación de supervivencia
x = np.array(df_train.drop(['Survived'], 1))
y = np.array(df_train['Survived'])
print(x)
print(y)

#Separar los Datos de entrenamiento
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

##Regresión logística
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)
print('Precision Regresion Logistica:')
print(logreg.score(x_train, y_train))

##Support Vector Machines
svc = SVC()
svc.fit(x_train, y_train)
Y_pred = svc.predict(x_test)
print('Precision Soporte de Vectores:')
print(svc.score(x_train, y_train))

##K Neighbors
Knn = KNeighborsClassifier(n_neighbors = 3)
Knn.fit(x_train, y_train)
Y_pred = Knn.predict(x_test)
print('Precision Vecinos mas cercanos:')
print(Knn.score(x_train, y_train))

#####################PREDICCIÓN UTILIZANDO LOS MODELOS#######################
ids = df_test['PassengerId']

##Regresión logística
prediccion_logreg = logreg.predict(df_test.drop('PassengerId', axis=1))
out_logreg = pd.DataFrame({'PassengerId' : ids, 'Survived': prediccion_logreg })
print('Prediccion Regresion Logistica:')
print(out_logreg.head())

##Support Vector Machines
prediccion_svc = svc.predict(df_test.drop('PassengerId', axis=1))
out_svc = pd.DataFrame({'PassengerId' : ids, 'Survived': prediccion_svc })
print('Prediccion RSoporte de Vectores:')
print(out_svc.head())

##K Neighbors
prediccion_knn = Knn.predict(df_test.drop('PassengerId', axis=1))
out_knn = pd.DataFrame({'PassengerId' : ids, 'Survived': prediccion_knn })
print('Prediccion Vecinos mas cercanos:')
print(out_knn.head())

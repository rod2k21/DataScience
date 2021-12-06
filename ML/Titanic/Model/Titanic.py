# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:50:22 2021
@title: Predicción de supervivencia en el Titanic
@author: Rodrigo Rodríguez
"""
###########################LIBRERÍAS A UTILIZAR#############################
import numpy as np
import pandas as pd
import os
from pathlib import Path

#############################IMPORTANDO DATA################################
print(os.environ['USERPROFILE'])
data_folder = Path("/GitHub/DataScience/ML/Titanic/Dataset/")
df_test = pd.read_csv(data_folder / "test.csv")
df_train = pd.read_csv(data_folder /"train.csv")

print(df_test.head())
print(df_train.head())

###########################ENTENDIENDO LA DATA##############################


#############################PROCESANDO LA DATA#############################


##########################APLICANDO ALGORITMOS DE ML########################


#####################PREDICCIÓN UTILIZANDO LOS MODELOS#######################


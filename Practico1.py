"""
Práctico 1 Diplomatura Datos
Clara Cabrera - Marianela Luján - Nicolás Salles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


#Generamos un objeto DataFrame de pandas
tabla=pd.read_csv('heart_diseases.csv')
print('Se muestran las primeras 5\n', tabla.head())

#Generamos una pequeña estadística
print('Resumen \n',tabla.describe())
#tabla.info()# Son todos datos numéricos

#Limpieza de los datos
tabla.drop('education', axis = 1,inplace=True)#Prescindimos de la columna Education
#print(tabla) Funciona

"""
correlaciones=tabla.corr()
#print(correlaciones)
sns.heatmap(correlaciones, annot=True)#Funciona
plt.show()
"""

"""
#15/07/24: Eliminación de la columna education
           Generación de la matriz de correlaciones
           Generación del heatmap de correlaciones
           Inicio de la limpieza del dataset
"""
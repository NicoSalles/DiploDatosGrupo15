"""
Práctico 1 Diplomatura Datos
Clara Cabrera - Marianela Luján - Nicolás Salles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Generamos un objeto DataFrame de pandas
tabla=pd.read_csv('heart_diseases.csv')
print('Se muestran las primeras 5 y las últimas 5 filas\n', tabla)

#Generamos una pequeña estadística
print('Resumen \n',tabla.describe())

"""
Práctico 1 Diplomatura Datos
Clara Cabrera - Marianela Luján - Nicolás Salles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import funciones as fun


#Generamos un objeto DataFrame de pandas
tabla=pd.read_csv('heart_diseases.csv')
print('Se muestran las primeras 5\n', tabla.head())

#Generamos una pequeña estadística
print('Resumen \n',tabla.describe())
#tabla.info()# Son todos datos numéricos

#Limpieza de los datos
print('Se suprime la columna "education" porque la consideramos prescindible')
tabla.drop('education', axis = 1,inplace=True)#Prescindimos de la columna Education

salida=fun.correlacion(tabla)
print('Valores más altos de correlación: \n', salida)

fun.cajas(tabla)
fun.histogramas(tabla)
fun.prueba_hipotesis(tabla)
resultado, predicciones_clases=fun.regresion_logistica(tabla)
if predicciones_clases is not None:
    fun.confusion(tabla, predicciones_clases)


"""
#15/07/24: Eliminación de la columna education
           Generación de la matriz de correlaciones
           Generación del heatmap de correlaciones
           Inicio de la limpieza del dataset

16/07/24: Creo el archivo funciones.py
17/07/24: Escribí funciones de gráficos en funciones.py 
19/07/24: Escribí algo para ordenar los valores de la matriz de correlaciones
21/07/24: Creé la función para la correlación
22/07/24: Hice los histogramas
23/07/24: Trabajé en la función prueba_de_hipótesis
24/07/24: Matriz de confusión
25/07/24: Regresión logística
26/07/24: 
"""
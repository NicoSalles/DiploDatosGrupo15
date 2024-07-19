"""
Funciones varias
Clara Cabrera - Marianela Luján - Nicolás Salles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def grafico_simple():
	print('Gráfico individual')
	ventana=plt.figure(figsize=(8,5))
	x=np.linspace(0,20,100)
	y=np.sin(x)
	plt.plot(x,y)
	plt.show()
	print('Fin función gráficos')

def graficos_multiples_ventana_simple():
	print('Función que dibuja varias curvas en una sola ventana')
	x=np.linspace(0.0001,20,100)
	y=np.array([np.sin(x),np.sin(x/2),1/x])
	print(y.shape)
	plt.plot(x,y.T)
	plt.show()
	print('Fin función gráficos')

def graficos_en_multiples_paneles():
	print('Función que dibuja una curva en cada panel')
	x=np.linspace(0.0001,20,100)
	ventana, eje=plt.subplots(1,2,figsize=(8,5))
	eje[0].plot(x,np.sin(x))
	eje[1].plot(x, np.cos(x))
	plt.show()



def limpiar_datos():
	pass


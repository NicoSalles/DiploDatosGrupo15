"""
Funciones varias
Clara Cabrera - Marianela Luján - Nicolás Salles
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm

def histogramas(tabla):
	print('Función de histogramas, cerrar la figura para continuar')
	ventana, ejes=plt.subplots(1,2,figsize=(8,5))
	ejes[0].hist(tabla["totChol"], bins=40)
	ejes[0].set(ylim=(0,600), xlabel='Colesterol total', ylabel='N° de personas', title='Distribución de colesterol')
	ejes[0].grid()
	ejes[1].hist(tabla["glucose"], bins=40)
	ejes[1].set(ylim=(0,180), xlabel='Glucosa', ylabel='N° de personas', title='Distribución de glucosa')
	ejes[1].grid()
	plt.show()
	

def cajas(tabla):
	pass

def limpiar_datos():
	pass


def correlacion(tabla):
	correlaciones=tabla.corr() #Obtenemos la matirz de correlaciones
	lista_correlaciones=correlaciones.values.flatten()  #La matriz se convierte en lista
	lista_correlaciones_filtrada=lista_correlaciones[ (lista_correlaciones != 1) 
	                            & (~np.isnan(lista_correlaciones)) 
	                            & (lista_correlaciones > 0.6)]  #Eliminamos los menores a 0.6 e iguales a 1
	lista_correlaciones_ordenada=np.sort(lista_correlaciones_filtrada)[::-1] #Se ordena de mayor a menor
    #sns.heatmap(correlaciones,annot=True)
    #plt.show()
	return lista_correlaciones_ordenada

def prueba_hipotesis(tabla):
	print('Prueba de hipótesis')
	mu_sys=tabla["sysBP"].mean()
	print("Presión sistólica media: ", mu_sys)
	var_sys=tabla["sysBP"].var()
	print("Varianza: ", var_sys)
	S_sys=tabla["sysBP"].std()
	print("Desviación estandar: ", S_sys)
	#Creación de una nueva columna para los que tienen mayor a 140
	tabla['AltaSys']=tabla['sysBP']>140

	#Creamos dos grupos: mayores y menores o iguales a 140
	grupo_alta_sys=tabla[tabla['AltaSys']]
	grupo_baja_sys=tabla[~tabla['AltaSys']]

	#Datos de los grupos
	n_grupo_altasys=len(grupo_alta_sys)
	n_grupo_bajasys=len(grupo_baja_sys)

	print(f'Tamaño del grupo con presión sistólica > 140mmHg: {n_grupo_altasys}')
	print(f'Tamaño del grupo con presión sistólica ≤ 140mmHg: {n_grupo_bajasys}')

	#Número de ataques cardíacos en cada grupo
	cuantos_ataques_altasys = grupo_alta_sys['TenYearCHD'].sum()
	cuantos_ataques_bajasys = grupo_baja_sys['TenYearCHD'].sum()

	tasa_grupo_altasys=grupo_alta_sys['TenYearCHD']/n_grupo_altasys
	tasa_grupo_bajasys=grupo_baja_sys['TenYearCHD']/n_grupo_bajasys

	print(f'Tasa de ataque cardíaco alta presión sistólica\n {tasa_grupo_altasys}')
	print(f'Tasa de ataque cardíaco baja presión sistólica\n {tasa_grupo_bajasys}')

	#Calcular la proporción pooled (combinada)
	p_combinada = (cuantos_ataques_altasys + cuantos_ataques_bajasys) / (n_grupo_altasys + n_grupo_bajasys)

	#Calcular el error estándar
	se = np.sqrt(p_combinada * (1 - p_combinada) * (1/n_grupo_altasys + 1/n_grupo_bajasys))

	#Calcular el estadístico Z
	z = (tasa_grupo_altasys - tasa_grupo_bajasys) / se

	#Calcular el valor p (doble cola)
	p_valor = 2 * (1 - norm.cdf(abs(z)))

	print(f"Proporción en grupo con presión > 140mmHg: {tasa_grupo_altasys}")
	print(f"Proporción en grupo con presión ≤ 140mmHg: {tasa_grupo_bajasys}")
	print(f"Estadístico Z: {z}")
	print(f"p valor: {p_valor}")

	








"""
def graficos_multiples_ventana_simple():
	print('Función que dibuja varias curvas en una sola ventana')
	x=np.linspace(0.0001,20,100)
	y=np.array([np.sin(x),np.sin(x/2),1/x])
	print(y.shape)
	plt.plot(x,y.T)
	plt.show()
	print('Fin función gráficos')
"""
	Gráfico simple
	x=np.linspace(0,20,100)
	y=np.sin(x)
	plt.plot(x,y)
	plt.show()
	print('Fin función gráficos')
	"""

def graficos_en_multiples_paneles():
	print('Función que dibuja una curva en cada panel')
	x=np.linspace(0.0001,20,100)
	ventana, eje=plt.subplots(1,2,figsize=(8,5))
	eje[0].plot(x,np.sin(x))
	eje[1].plot(x, np.cos(x))
	plt.show()
"""
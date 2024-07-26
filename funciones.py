"""
Funciones varias
Clara Cabrera - Marianela Luján - Nicolás Salles
"""
import numpy as np #Para análisis numérico
import pandas as pd #Para trabajar con tablas
import matplotlib.pyplot as plt #Para la mayoría de los gráficos
import seaborn as sns #Para algunos gráficos
import scipy.stats as stats
from scipy.stats import norm #Para el test de hipótesis
from sklearn import metrics #Para la matriz de confusión
import statsmodels.formula.api as smf

def histogramas(tabla):
	print('Función de histogramas, cerrar la figura para continuar')
	ventana, ejes=plt.subplots(1,2,figsize=(5,3))
	ejes[0].hist(tabla["totChol"], bins=40)
	ejes[0].set(ylim=(0,600), xlabel='Colesterol total', ylabel='N° de personas', title='Distribución de colesterol')
	ejes[0].grid()
	ejes[1].hist(tabla["glucose"], bins=40)
	ejes[1].set(ylim=(0,180), xlabel='Glucosa', ylabel='N° de personas', title='Distribución de glucosa')
	ejes[1].grid()
	plt.show()
	

def cajas(tabla):
	pass


def limpiar_datos(tabla):
	#Rellenar valores nulos con la media de cada columna
	rellenados_con_media = tabla.fillna(df.mean())

	#Eliminar filas con valores nulos restantes (si los hay)
	eliminados_con_ceros= tabla.dropna()

	#Filtrar datos (ejemplo: filas donde 'A' es mayor que 1)
	#filtrados_ = tabla.loc[df['A'] > 1]
	print(tabla)



def correlacion(tabla):
	correlaciones=tabla.corr() #Obtenemos la matirz de correlaciones
	lista_correlaciones=correlaciones.values.flatten()  #La matriz se convierte en lista
	lista_correlaciones_filtrada=lista_correlaciones[ (lista_correlaciones != 1) 
	                            & (~np.isnan(lista_correlaciones)) 
	                            & (lista_correlaciones > 0.6)]  #Eliminamos los menores a 0.6 e iguales a 1
	lista_correlaciones_ordenada=np.sort(lista_correlaciones_filtrada)[::-1] #Se ordena de mayor a menor
	figura=plt.figure(figsize=(6,6))
	sns.heatmap(correlaciones,annot=False) #Se superponen los valores si dejo en True
	print('Función de correlación, cerrar la figura para continuar')
	plt.show()
	return lista_correlaciones_ordenada

def prueba_hipotesis(tabla):
	print('Prueba de hipótesis')
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

	tasa_grupo_altasys=cuantos_ataques_altasys/n_grupo_altasys
	tasa_grupo_bajasys=cuantos_ataques_bajasys/n_grupo_bajasys
	
	#Calcular la proporción pooled (combinada)
	p_combinada = (cuantos_ataques_altasys + cuantos_ataques_bajasys) / (n_grupo_altasys + n_grupo_bajasys)
	#Calcular el error estándar
	se = np.sqrt(p_combinada * (1 - p_combinada) * (1/n_grupo_altasys + 1/n_grupo_bajasys))
	#Calcular el estadístico Z
	z = (tasa_grupo_altasys - tasa_grupo_bajasys) / se
	#Calcular el valor p (doble cola, por eso el 2)
	p_valor = 2 * (1 - norm.cdf(abs(z)))
	print(f"Proporción en grupo con presión > 140mmHg: {tasa_grupo_altasys}")
	print(f"Proporción en grupo con presión ≤ 140mmHg: {tasa_grupo_bajasys}")
	print(f"Estadístico Z: {z}")
	print(f"p valor: {p_valor}")




def regresion_logistica(tabla):
	"""
    Realiza una regresión logística en los datos proporcionados.
    
    Parámetros:
    tabla (DataFrame): DataFrame que contiene los datos para el modelo.

    Retorna:
    resultado: Objeto con el modelo ajustado.
    predicciones: Predicciones del modelo ajustado.
    """
	formula ='TenYearCHD ~ sysBP + age + glucose + totChol' #Fórmula del modelo
	#Ajuste del modelo de regresión logística
	modelo=smf.logit(formula, data=tabla)
	print('Listo el modelo')
	resultado=modelo.fit()
    # Obtener predicciones
	predicciones = resultado.predict(tabla)
	print('Predicciones obtenidas')
	#Presentación del modelo ajustado
	print(resultado.summary())
	# Convertir predicciones a clases (0 o 1) usando un umbral de 0.5
	predicciones_clases = (predicciones >= 0.5).astype(int)
	return resultado, predicciones_clases




def confusion(tabla, prediccion):
	"""
    Genera y muestra la matriz de confusión.

    Parámetros:
    tabla (DataFrame): DataFrame que contiene los datos observados.
    prediccion (array): Predicciones del modelo.
    """
	datos_observados=tabla['TenYearCHD']
	#datos_predichos=np.random.choice([0, 1], size=len(datos_observados), p=[0.8, 0.2])
	datos_predichos=prediccion
	matriz_confusion=metrics.confusion_matrix(datos_observados,datos_predichos)
	# Crear la visualización de la matriz de confusión
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matriz_confusion, display_labels = [False, True])
	# Graficar la matriz de confusión
	figura, ejes = plt.subplots()
	cm_display.plot(ax=ejes)
	# Cambiar los rótulos de los ejes
	ejes.set_xlabel('Etiqueta Predicha')
	ejes.set_ylabel('Etiqueta Verdadera')
	ejes.set_title('Matriz de Confusión')
	print('Función Matriz de Confusión, cerrar la figura para continuar')
	plt.show()










"""
def graficos_multiples_ventana_simple():
	print('Función que dibuja varias curvas en una sola ventana')
	x=np.linspace(0.0001,20,100)
	y=np.array([np.sin(x),np.sin(x/2),1/x])
	print(y.shape)
	plt.plot(x,y.T)
	plt.show()
	print('Fin función gráficos')

	Gráfico simple
	x=np.linspace(0,20,100)
	y=np.sin(x)
	plt.plot(x,y)
	plt.show()
	print('Fin función gráficos')

def graficos_en_multiples_paneles():
	print('Función que dibuja una curva en cada panel')
	x=np.linspace(0.0001,20,100)
	ventana, eje=plt.subplots(1,2,figsize=(8,5))
	eje[0].plot(x,np.sin(x))
	eje[1].plot(x, np.cos(x))
	plt.show()


def
	# Crear una figura con subplots para cada combinación de pares de variables
variables = df.columns
num_vars = len(variables)
fig, axes = plt.subplots(nrows=num_vars, ncols=num_vars, figsize=(20, 20))

# Generar los gráficos
for i, var1 in enumerate(variables):
    for j, var2 in enumerate(variables):
        if i != j:
            axes[i, j].scatter(df[var1], df[var2])
            axes[i, j].set_xlabel(var1)
            axes[i, j].set_ylabel(var2)
        else:
            axes[i, j].axis('off')  # No mostrar gráfico si es la misma variable

# Ajustar el layout para que no se superpongan los gráficos
plt.tight_layout()
plt.show()

"""
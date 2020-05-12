# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:11:57 2020

@author: nicol
"""

#### Filtro inicial: eliminar variables segun correlacion

##### Imports
import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
import multiprocessing
import pingouin
#%%
from sklearn.base import clone
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score 
from sklearn.metrics import roc_auc_score

## Evitar warnings de convergencia
import warnings
warnings.filterwarnings("ignore")
#%%
### Modelo base
os.chdir('../..')
sys.path.insert(0, os.getcwd())
#%%
#Define el archivo en el que se guararan los logs del codigo
import logging
from logging.handlers import RotatingFileHandler

file_name = 'resampling'
logger = logging.getLogger()
dir_log = f'logs/{file_name}.log'

### Crea la carpeta de logs
if not os.path.isdir('logs'):
    os.makedirs('logs', exist_ok=True) 

handler = RotatingFileHandler(dir_log, maxBytes=2000000, backupCount=10)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
                    handlers = [handler])
#%%
import scripts.funciones as funciones
version = 'verFinal'    
mod_version = funciones.carga_model('.', f'models/{version}', version)
mod = mod_version['model'].steps[0][1]

classifier = mod_version['model'].steps[1][1]

### Carga random forest
# classifier  = funciones.carga_model_ind('.', f'models/ver012', 'rforest_20200505_1249')
#%%
########## LECTURA DE DATOS ############
d_ini = dt.datetime(2017,6,1)
d_fin = dt.datetime(2019,8,1) 

### params
freq1 = '1D'
freq2 = '3D'
freq3 = '7D'
freq4 = '14D'
freq5 = '30D'
freq6 = '60D'
n_proc = multiprocessing.cpu_count() -1

### Realizamos la lectura de la informacion climatica en el rango de fechas
### especificado, incluye la etiqueta de si ocurre o no un accidente. 
### Posteriormente, en la organizacion de la informacion climatica, lo
### que se hace es agregar las variables con la informacion distribucional
### de las ultimas 5 horas de la info climatica
data = funciones.read_clima_accidentes(d_ini, d_fin, poblado = True)
data_org = funciones.organizar_data_infoClima(data)


### agregamos la informacion relacionada a la cantidad de accidentes ocurridas
### en las ultimas X horas
### Agregar senales
senales = [freq1, freq2, freq3, freq4, freq5, freq6]
d_ini_acc = d_ini - dt.timedelta(days = int(freq6.replace('D', '')))  ### freq mayor
raw_accidentes = funciones.read_accidentes(d_ini_acc, d_fin)
for fresen in senales:
    data_org = funciones.obtener_accidentes_acumulados(data_org, 
                                                        raw_accidentes, 
                                                        freq = fresen)


### Convertimos la bariable de Barrios en variable dummy para ser incluida
### en el modelo
data_org['poblado'] = data_org['BARRIO']
data_org= pd.get_dummies(data_org, columns=['poblado'])

### Relizamos la particion del conjunto de datos en las variables
### explicativas (X) y la variable respuesta (Y)
X = data_org.drop(columns = ['TW','BARRIO','Accidente','summary'])
Y = data_org['Accidente']  



############# Partir en train y validation
from sklearn.model_selection import train_test_split
x_tra, x_val, y_tra, y_val = train_test_split(X,Y,test_size=0.2, random_state=42)



########## Aqui se dejaria la X y Y de train con undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy = 2/3, random_state = 42)
X_rus, y_rus = rus.fit_sample(x_tra, y_tra)




#### Calculamos correlaciones, correlaciones parciales y point biserial correlations
X_rus['Accidente'] = y_rus.copy()
corr_y = X_rus.corr()['Accidente']
pcorr_y = X_rus.pcorr()['Accidente']

#### Con valores absolutos
abs_corr = []
abs_pcorr = []
for i in range(len(corr_y)):
    abs_corr.append(abs(corr_y[i]))
    abs_pcorr.append(abs(pcorr_y[i]))
corr_full = pd.DataFrame(corr_y)
pcorr_full = pd.DataFrame(pcorr_y)
corr_full['abs'] = abs_corr.copy()
pcorr_full['abs'] = abs_pcorr.copy()

### Ordenrarlas
corr_full = corr_full.sort_values(by='abs', ascending=False)
pcorr_full = pcorr_full.sort_values(by='abs', ascending=False)


### Ordeno la matriz X segun las correlaciones parciales de forma ascendente (para que funcione
### bien el siguiente paso)
vars_ord = list(pcorr_full.index)
X_rus_ord = X_rus[vars_ord[1:][::-1]]

#### Calcula matriz de correlacion de las variables explicativas
corr_x = X_rus_ord.corr()

### Valor de correlacion umbral (si algun par de variables explicativas tiene
### una correlacion mayor a esta, se elimina la variable de este par que 
### tenga una menor correlacion parcial con la variable de salida)
cor_umb = 0.95

variabs = list(X_rus_ord.columns)
X_rus_aux = X_rus_ord.copy()

for va in variabs:
    esta_col = corr_x[va]
    if np.sort(esta_col)[-2]>cor_umb:
        X_rus_aux = X_rus_aux.drop(columns=va)
        corr_x = X_rus_aux.corr()


### X_rus_aux tendria la X, luego del filtro de correlacion
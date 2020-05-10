# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:19:05 2020

@author: nicol
"""

######### Script para probar varios distintos metodos de resampling ###########

##### Imports
import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import multiprocessing


from sklearn.base import clone
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score 
from sklearn.metrics import roc_auc_score


### Modelo base
os.chdir('../..')
sys.path.insert(0, os.getcwd())
import scripts.funciones as funciones
version = 'verFinal'    
mod_version = funciones.carga_model('.', f'models/{version}', version)
mod = mod_version['model'].steps[0][1]

classifier = mod_version['model'].steps[1][1][1]







########## LECTURA DE DATOS ############
d_ini = dt.datetime(2017,6,1)
d_fin = dt.datetime(2019,8,1) 

### params
cv = 3
freq1 = '4D'
freq2 = '14D'
balance = 'rus'
score = 'roc_auc'
prop_deseada_under = 0.4
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
d_ini_acc = d_ini - dt.timedelta(days = int(freq2.replace('D', '')))
raw_accidentes = funciones.read_accidentes(d_ini_acc, d_fin)

### Agrega senal a corto plazo
data_org = funciones.obtener_accidentes_acumulados(data_org, 
                                                    raw_accidentes, 
                                                    freq = freq1)

### Agrega senal a largo plazo
data_org = funciones.obtener_accidentes_acumulados(data_org, 
                                                    raw_accidentes, 
                                                    freq = freq2)

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





############ Metodo 1: random undersampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy = 2/3, random_state = 42)
X_rus, y_rus = rus.fit_sample(x_tra, y_tra)
classifier1 = clone(classifier)
classifier1.fit(X_rus, y_rus)
pred1 = classifier1.predict(x_val)
prob1 = classifier1.predict_proba(x_val)[:,1]
print('\n\n Random undersampling: ')
print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
print('F1 score: ' +str(f1_score(y_val, pred1)))
print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
print('Precission score: ' +str(precision_score(y_val, pred1)))
print('Recall score: ' +str(recall_score(y_val, pred1)))



############ Metodo 1: random undersampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy = 2/3, random_state = 42)
X_rus, y_rus = rus.fit_sample(x_tra, y_tra)
classifier1 = clone(classifier)
classifier1.fit(X_rus, y_rus)
pred1 = classifier1.predict(x_val)
prob1 = classifier1.predict_proba(x_val)[:,1]
print('\n\nRandom undersampling: ')
print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
print('F1 score: ' +str(f1_score(y_val, pred1)))
print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
print('Precission score: ' +str(precision_score(y_val, pred1)))
print('Recall score: ' +str(recall_score(y_val, pred1)))




############ Metodo 2: random oversampling
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy = 2/3, random_state = 42)
X_rus, y_rus = ros.fit_sample(x_tra, y_tra)
classifier2 = clone(classifier)
classifier2.fit(X_rus, y_rus)
pred1 = classifier2.predict(x_val)
prob1 = classifier2.predict_proba(x_val)[:,1]
print('\n\nRandom oversampling: ')
print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
print('F1 score: ' +str(f1_score(y_val, pred1)))
print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
print('Precission score: ' +str(precision_score(y_val, pred1)))
print('Recall score: ' +str(recall_score(y_val, pred1)))




############ Metodo 3: tomek link undersampling
from imblearn.under_sampling import TomekLinks

rus = TomekLinks()
X_rus, y_rus = rus.fit_sample(x_tra, y_tra)
classifier3 = clone(classifier)
classifier3.fit(X_rus, y_rus)
pred1 = classifier1.predict(x_val)
prob1 = classifier1.predict_proba(x_val)[:,1]
print('\n\nRandom undersampling: ')
print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
print('F1 score: ' +str(f1_score(y_val, pred1)))
print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
print('Precission score: ' +str(precision_score(y_val, pred1)))
print('Recall score: ' +str(recall_score(y_val, pred1)))
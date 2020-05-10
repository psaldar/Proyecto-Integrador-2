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

## Evitar warnings de convergencia
import warnings
warnings.filterwarnings("ignore")

### Modelo base
os.chdir('../..')
sys.path.insert(0, os.getcwd())
import scripts.funciones as funciones
version = 'verFinal'    
mod_version = funciones.carga_model('.', f'models/{version}', version)
mod = mod_version['model'].steps[0][1]

classifier = mod_version['model'].steps[1][1]







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
x_tra, x_val, y_tra, y_val = train_test_split(X,Y,test_size=0.7, random_state=42)




#### Fijar unos iniciales (Tomek, Nearest neighbors, etc)
### para ejecutarlos solo una vez
### En el ciclo, luego de estos filtros iniciales, se aplica random undersampling
### o smote para balancear el resto de las observaciones
### Tomek
from imblearn.under_sampling import TomekLinks
rus = TomekLinks()
X_tom, y_tom = rus.fit_sample(x_tra, y_tra)
#### ENN
from imblearn.under_sampling import EditedNearestNeighbours
enn = EditedNearestNeighbours()
X_enn, y_enn = enn.fit_sample(x_tra, y_tra)





### Proporciones a evaluar entre 1s y 0s
sampling_strategies = [0.1/0.9, 0.2/0.8, 0.3/0.7, 0.4/0.6, 0.5/0.5]

### Evaluar distintas proporciones
for samp in sampling_strategies:
    print('\n\n\nProporcion entre 1s y 0s: '+str(samp))
    ############ Metodo 1: random undersampling
    from imblearn.under_sampling import RandomUnderSampler
    
    rus = RandomUnderSampler(sampling_strategy = samp, random_state = 42)
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
    
    ros = RandomOverSampler(sampling_strategy = samp, random_state = 42)
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
    
    
    
    
    ########### Metodo 3: tomek link undersampling + Random
    rustom = RandomUnderSampler(sampling_strategy = samp, random_state = 42)    
    X_rus, y_rus = rustom.fit_sample(X_tom, y_tom)
    classifier3 = clone(classifier)
    classifier3.fit(X_rus, y_rus)
    pred1 = classifier3.predict(x_val)
    prob1 = classifier3.predict_proba(x_val)[:,1]
    print('\n\nTomek Link + Random Undersampling: ')
    print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
    print('F1 score: ' +str(f1_score(y_val, pred1)))
    print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
    print('Precission score: ' +str(precision_score(y_val, pred1)))
    print('Recall score: ' +str(recall_score(y_val, pred1)))
    
    
    
    
    
    ############ Metodo 4: cluster centroids
    from imblearn.under_sampling import ClusterCentroids
    
    cus = ClusterCentroids(sampling_strategy = samp, random_state = 42,  n_jobs=n_proc)
    X_rus, y_rus = cus.fit_sample(x_tra, y_tra)
    classifier4 = clone(classifier)
    classifier4.fit(X_rus, y_rus)
    pred1 = classifier4.predict(x_val)
    prob1 = classifier4.predict_proba(x_val)[:,1]
    print('\n\nUndersampling cluster centroids: ')
    print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
    print('F1 score: ' +str(f1_score(y_val, pred1)))
    print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
    print('Precission score: ' +str(precision_score(y_val, pred1)))
    print('Recall score: ' +str(recall_score(y_val, pred1)))
    
    
    
    
    ############ Metodo 5: nearest to cluster centroids    
    cus = ClusterCentroids(sampling_strategy = samp, random_state = 42, voting='hard',  n_jobs=n_proc)
    X_rus, y_rus = cus.fit_sample(x_tra, y_tra)
    classifier5 = clone(classifier)
    classifier5.fit(X_rus, y_rus)
    pred1 = classifier5.predict(x_val)
    prob1 = classifier5.predict_proba(x_val)[:,1]
    print('\n\nUndersampling nearest to cluster centroids: ')
    print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
    print('F1 score: ' +str(f1_score(y_val, pred1)))
    print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
    print('Precission score: ' +str(precision_score(y_val, pred1)))
    print('Recall score: ' +str(recall_score(y_val, pred1)))
    
    
    ############ Metodo 6: Near Miss
    from imblearn.under_sampling import NearMiss
    
    near = NearMiss(sampling_strategy = samp,  n_jobs=n_proc)
    X_rus, y_rus = near.fit_sample(x_tra, y_tra)
    classifier6 = clone(classifier)
    classifier6.fit(X_rus, y_rus)
    pred1 = classifier6.predict(x_val)
    prob1 = classifier6.predict_proba(x_val)[:,1]
    print('\n\nNear Miss: ')
    print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
    print('F1 score: ' +str(f1_score(y_val, pred1)))
    print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
    print('Precission score: ' +str(precision_score(y_val, pred1)))
    print('Recall score: ' +str(recall_score(y_val, pred1)))
    
    
    
    
    ############ Metodo 7: EditedNearestNeighbours + Random undersampling
    rusenn = RandomUnderSampler(sampling_strategy = samp, random_state = 42)    
    X_rus, y_rus = rusenn.fit_sample(X_enn, y_enn)
    classifier7 = clone(classifier)
    classifier7.fit(X_rus, y_rus)
    pred1 = classifier7.predict(x_val)
    prob1 = classifier7.predict_proba(x_val)[:,1]
    print('\n\nENN + Random Undersampling: ')
    print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
    print('F1 score: ' +str(f1_score(y_val, pred1)))
    print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
    print('Precission score: ' +str(precision_score(y_val, pred1)))
    print('Recall score: ' +str(recall_score(y_val, pred1)))
    
    


    
    ############ Metodo 8: Smote oversampling
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(sampling_strategy = samp, random_state = 42,  n_jobs=n_proc)
    X_rus, y_rus = smote.fit_sample(x_tra, y_tra)
    classifier8 = clone(classifier)
    classifier8.fit(X_rus, y_rus)
    pred1 = classifier8.predict(x_val)
    prob1 = classifier8.predict_proba(x_val)[:,1]
    print('\n\nSMOTE: ')
    print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
    print('F1 score: ' +str(f1_score(y_val, pred1)))
    print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
    print('Precission score: ' +str(precision_score(y_val, pred1)))
    print('Recall score: ' +str(recall_score(y_val, pred1)))
    
    
    
    ############ Metodo 9: Smote oversampling + tomek undersampling
    from imblearn.combine import SMOTETomek
    
    stom = SMOTETomek(sampling_strategy = samp, random_state = 42,  n_jobs=n_proc)
    X_rus, y_rus = stom.fit_sample(x_tra, y_tra)
    classifier9 = clone(classifier)
    classifier9.fit(X_rus, y_rus)
    pred1 = classifier9.predict(x_val)
    prob1 = classifier9.predict_proba(x_val)[:,1]
    print('\n\nSMOTETomek: ')
    print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
    print('F1 score: ' +str(f1_score(y_val, pred1)))
    print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
    print('Precission score: ' +str(precision_score(y_val, pred1)))
    print('Recall score: ' +str(recall_score(y_val, pred1)))
    
    
    
    
    
    ############ Metodo 10: Smote oversampling + edited nearest neighbors undersampling
    from imblearn.combine import SMOTEENN
    
    senn = SMOTEENN(sampling_strategy = samp, random_state = 42, n_jobs=n_proc)
    X_rus, y_rus = senn.fit_sample(x_tra, y_tra)
    classifier10 = clone(classifier)
    classifier10.fit(X_rus, y_rus)
    pred1 = classifier10.predict(x_val)
    prob1 = classifier10.predict_proba(x_val)[:,1]
    print('\n\nSMOTEENN: ')
    print('ROC AUC score: ' +str(roc_auc_score(y_val, prob1)))
    print('F1 score: ' +str(f1_score(y_val, pred1)))
    print('Balanced accuracy score: ' +str(balanced_accuracy_score(y_val, pred1)))
    print('Precission score: ' +str(precision_score(y_val, pred1)))
    print('Recall score: ' +str(recall_score(y_val, pred1)))
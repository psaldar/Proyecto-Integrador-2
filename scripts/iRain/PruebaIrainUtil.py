# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:23:27 2020

@author: nicol
"""

import sqlite3
import pandas as pd
import os 
import sys
 

db_name = '../../data/irain_info.sqlite3'

conn = sqlite3.connect(db_name)

query = ' SELECT *                       FROM irain  WHERE LAT <= 6.205 AND LAT >= 6.195 AND LON <= -75.555 AND LON >=-75.565'

data = pd.read_sql_query(query,conn)



### Leer datos de train y validation
data_train_u = pd.read_csv('../../data/train.csv') 
data_val = pd.read_csv('../../data/validation.csv') 

### Modelo usado
os.chdir('../..')
sys.path.insert(0, os.getcwd())
import scripts.funciones as funciones
version = 'verFinal'    
mod_version = funciones.carga_model('.', f'models/{version}', version)
mod = mod_version['model'].steps[0][1]

classifier = mod_version['model'].steps[1][1][1]

### Nuevos modelos 
from sklearn.base import clone
classifier_viejo = clone(classifier)
classifier_nuevo = clone(classifier)


### Entrenar y validar modelo sin irain
from sklearn.metrics import roc_auc_score
data_tr1 = data_train_u.drop(['BARRIO', 'Accidente', 'TW'], axis=1)
data_va1 = data_val.drop(['BARRIO', 'Accidente', 'TW'], axis=1)
classifier_viejo.fit(data_tr1,data_train_u['Accidente'])
preds1 = classifier_viejo.predict_proba(data_va1)[:,1]
print('ROC AUC sin Irain: '+str(roc_auc_score(data_val['Accidente'], preds1)))




### Estandarizo data precip
datapre =  data['PRECIP'].values
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
datapreci = ss.fit_transform(datapre.reshape(-1,1))

### Crear dataframes nuevas (para poder pegar la columna de precipitacion de irain)
dicto = {}
voyv=0
for i in data['TW']:
    dicto[i] = datapreci[voyv]
    voyv=voyv+1

precips_new = []
for k in data_train_u['TW']:
    try:
        preci = dicto[k]
    except:
        preci = 0
    precips_new.append(preci)

precips_new2 = []
for k in data_val['TW']:
    try:
        preci = dicto[k]
    except:
        preci = 0
    precips_new2.append(preci)



### Entrenar y validar modelo con Irain
from sklearn.metrics import roc_auc_score
data_tr2 = data_tr1.copy()
data_tr2['preci'] = precips_new.copy()
data_va2 = data_va1.copy()
data_va2['preci'] = precips_new2.copy()
classifier_nuevo.fit(data_tr2,data_train_u['Accidente'])
preds2 = classifier_nuevo.predict_proba(data_va2)[:,1]
print('ROC AUC con Irain: '+str(roc_auc_score(data_val['Accidente'], preds2)))


### No se ven mejoras considerables al incluir datos de Irain
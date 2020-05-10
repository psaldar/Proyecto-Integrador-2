# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:56:46 2020

@author: Pablo Saldarriaga
"""
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
#%%
### Realizamos el cambio de directoroi de trabajo al "Directorio Base"
current_dir = os.getcwd()
base_path = os.path.dirname(current_dir)
base_path = os.path.dirname(base_path)
#%%

train = pd.read_csv('data/train.csv')
y_train = train.Accidente

validation = pd.read_csv('data/validation.csv')
y_val = validation.Accidente

test = pd.read_csv('data/test.csv')
y_test = test.Accidente

test= test[['precipIntensity',
 'precipProbability',
 'temperature',
 'apparentTemperature',
 'dewPoint',
 'humidity',
 'windSpeed',
 'cloudCover',
 'uvIndex',
 'visibility',
 'poblado_alejandria',
 'cumAcc_96H',
 'cumAcc_336H',
 'poblado_altosdelpoblado',
 'poblado_astorga',
 'poblado_castropol',
 'poblado_elcastillo',
 'poblado_eldiamanteno2',
 'poblado_laaguacatala',
 'poblado_lalinde',
 'poblado_losbalsosno1',
 'poblado_losnaranjos',
 'poblado_manila',
 'poblado_sanlucas',
 'poblado_santamariadelosangeles',
 'poblado_villacarlota',
 'hora_1',
 'hora_2',
 'hora_3',
 'hora_6',
 'hora_7',
 'hora_9',
 'hora_10',
 'hora_14',
 'hora_16',
 'hora_19',
 'hora_20',
 'hora_21',
 'hora_23',
 'icon_partly-cloudy-day',
 'icon_partly-cloudy-night',
 'icon_rain',
 'dia_sem_3',
 'dia_sem_4',
 'dia_sem_5',
 'dia_sem_6',
 'humidity_mean',
 'windSpeed_mean']]

test['Accidente'] = y_test

scaler = StandardScaler()
scaler.fit(train)


train_z = pd.DataFrame(scaler.transform(train), columns = train.columns)
validation_z = pd.DataFrame(scaler.transform(validation), columns = train.columns)
test_z = pd.DataFrame(scaler.transform(test), columns = train.columns)

train_z['Accidente'] = y_train
validation_z['Accidente'] = y_val
test_z['Accidente'] = y_test


train_z.to_csv('data/train_z.csv')
validation_z.to_csv('data/validation_z.csv')
test_z.to_csv('data/test_z.csv')

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
barrio_train = train['BARRIO'].values
tw_train = train['TW'].values
train = train.drop(columns = ['BARRIO','TW'])

validation = pd.read_csv('data/validation.csv')
y_val = validation.Accidente
barrio_val = validation['BARRIO'].values
tw_val = validation['TW'].values
validation = validation.drop(columns = ['BARRIO','TW'])

test = pd.read_csv('data/test.csv')
y_test = test.Accidente
barrio_test = test['BARRIO'].values
tw_test = test['TW'].values
test = test.drop(columns = ['BARRIO','TW'])


scaler = StandardScaler()
scaler.fit(train)


train_z = pd.DataFrame(scaler.transform(train), columns = train.columns)
validation_z = pd.DataFrame(scaler.transform(validation), columns = train.columns)
test_z = pd.DataFrame(scaler.transform(test), columns = train.columns)

train_z['Accidente'] = y_train
validation_z['Accidente'] = y_val
test_z['Accidente'] = y_test

train_z['BARRIO'] = barrio_train
validation_z['BARRIO'] = barrio_val
test_z['BARRIO'] = barrio_test

train_z['TW'] = tw_train
validation_z['TW'] = tw_val
test_z['TW'] = tw_test


train_z.to_csv('data/train_z.csv', index = False, sep = ',')
validation_z.to_csv('data/validation_z.csv', index = False, sep = ',')
test_z.to_csv('data/test_z.csv', index = False, sep = ',')

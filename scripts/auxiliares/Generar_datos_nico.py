# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:06:01 2020

@author: Pablo Saldarriaga
"""
import os
import json
import pandas as pd
import datetime as dt
#%%
### Realizamos el cambio de directoroi de trabajo al "Directorio Base"
current_dir = os.getcwd()
base_path = os.path.dirname(current_dir)
base_path = os.path.dirname(base_path)

os.chdir(base_path)
#%%
import scripts.funciones as funciones
#%%

d_ini = dt.datetime(2017,6,1)
d_fin = dt.datetime(2020,1,1)

data = funciones.read_clima_accidentes(d_ini, d_fin)

poblado = ['alejandria','altosdelpoblado',
                'astorga','barriocolombia',
                'castropol','elcastillo',
                'eldiamanteno2','elpoblado',
                'eltesoro','laaguacatala',
                'laflorida','lalinde',
                'laslomasno1','laslomasno2',
                'losbalsosno1','losbalsosno2',
                'losnaranjos','manila',
                'patiobonito','sanlucas',
                'santamariadelosangeles',
                'villacarlota']

data_poblado = data[data['BARRIO'].isin(poblado)].reset_index(drop = True)

#%%
data_poblado.to_csv('data/info_poblado_all.csv', index = False, sep = ',')

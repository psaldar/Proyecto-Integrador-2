# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:44:24 2020

@author: Pablo Saldarriaga
"""
### Ese script consolida la informacion climatica para el poblado en los years
### 2017, 2018 y 2019 en un solo sqlite
import os
import json
import sqlite3
import pandas as pd
#%%
### Realizamos el cambio de directoroi de trabajo al "Directorio Base" que se
### encuentra en el archivo conf.json
current_dir = os.getcwd()

file_name = 'conf.json'
path = os.path.join(current_dir, f'{file_name}')
with open(path, 'r') as f:
    info_conf = json.load(f)
            
base_path = info_conf['base_path']
os.chdir(base_path)
#%%

db2017 = 'data/info_clima2017.sqlite3'
db2018 = 'data/info_clima2018.sqlite3'
db2019 = 'data/info_clima2019.sqlite3'

conn2017 = sqlite3.connect(db2017)
conn2018 = sqlite3.connect(db2018)
conn2019 = sqlite3.connect(db2019)

query_clima = """ SELECT *
                  FROM clima """
                  
query_accidentes = """ SELECT *
                       FROM accidentes """
                       
query_Raccidentes = """ SELECT *
                       FROM raw_accidentes """
                
### Info Clima

clima2017 = pd.read_sql_query(query_clima,conn2017)
clima2018 = pd.read_sql_query(query_clima,conn2018)
clima2019 = pd.read_sql_query(query_clima,conn2019)
clima_all = pd.concat([clima2017,clima2018,clima2019])

### info Raw_accidentes

r_accidentes2017 = pd.read_sql_query(query_Raccidentes,conn2017)
r_accidentes2018 = pd.read_sql_query(query_Raccidentes,conn2018)
r_accidentes2019 = pd.read_sql_query(query_Raccidentes,conn2019)

r_accidentes_all = pd.concat([r_accidentes2017,r_accidentes2018,r_accidentes2019])


### Info accidentes
accidentes2017 = pd.read_sql_query(query_accidentes,conn2017)
accidentes2018 = pd.read_sql_query(query_accidentes,conn2018)
accidentes2019 = pd.read_sql_query(query_accidentes,conn2019)

accidentes_all = pd.concat([accidentes2017,accidentes2018,accidentes2019])

### Guardar toda la info

db_pi = 'data/database_pi2.sqlite3'
conn_pi = sqlite3.connect(db_pi)

clima_all.to_sql('clima',conn_pi, if_exists = 'append', index = False)
r_accidentes_all.to_sql('raw_accidentes',conn_pi, if_exists = 'append', index = False)
accidentes_all.to_sql('accidentes',conn_pi, if_exists = 'append', index = False)

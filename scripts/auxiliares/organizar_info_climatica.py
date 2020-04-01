# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:09:18 2020

@author: Pablo Saldarriaga
"""
import os
import json
import sqlite3
import pandas as pd
import datetime as dt
#%%
current_dir = os.getcwd()


file_name = 'conf.json'
path = os.path.join(current_dir, f'{file_name}')
with open(path, 'r') as f:
    info_conf = json.load(f)
            
base_path = info_conf['base_path']
os.chdir(base_path)
#%%
from scripts.auxiliares.consulta_darksky import darksky 
#%%
file_name = 'darksky_key.json'
with open(f'data/{file_name}', 'r') as f:
    info_key = json.load(f)
            
api_key = info_key['darksky']['key']
darksky_api = darksky(api_key)
#%%

### Realiza la conexion al sqlite donde se guarda la informacion climatica
file_name = 'data/info_clima2018.sqlite3'
conn = sqlite3.connect(file_name)

### Esto se realiza en caso que ya exista informacion consultada en la BD
#query = """SELECT * FROM CLIMA """
#info_db = pd.read_sql_query(query,conn)
#
#barrios = info_db[['TW','BARRIO']].groupby('BARRIO').max().reset_index()
#barrios = barrios[barrios['TW']==barrios['TW'].max()].reset_index(drop = True)

### Define el rango de fechas en el cual se busca la informacion
d_ini = dt.datetime(2018,1,1)
d_fin = dt.datetime(2018,12,31)

### Se obtiene un arreglo cuyas elementos son cada uno de los dias entre el
### rango de fechas dadas, ademas de incluir la informacion de los centroides
### de los barrios en donde se hara la busqueda
dates = pd.date_range(start=d_ini, end=d_fin, freq='1D')
centroides = pd.read_csv('data/centroides_barrios2018.csv', sep = ',')

### elimina los barrios que ya fueron consultados
#centroides = centroides[~centroides['BARRIO'].isin(barrios['BARRIO'])].reset_index(drop = True)
#print(len(centroides))

clima_cols = ['TW','BARRIO','summary','icon','precipIntensity','precipProbability',
              'temperature','apparentTemperature','dewPoint','humidity','windSpeed',
              'windBearing','cloudCover','uvIndex','visibility']

missing = []
longitud = len(centroides)*len(dates)
count = 0
for barrio, lon, lat in centroides[['BARRIO','Lon','Lat']].values:
    for time in dates:
        
        count = count + 1
        
        if count % 100 == 0:
        
            print(f'Porcentaje de informacion recuperada: {100*count/longitud}%')
        
        data, status_code = darksky_api.get_hourly(lat =lat, lon = lon,r_date =time)

        if status_code ==200:

            df = pd.DataFrame(data).rename(columns = {'time':'TW'})
            df['BARRIO'] = barrio
            
            for col in clima_cols:
                if not col in df.columns:
                    df[col] = None
            
            df[clima_cols].to_sql('clima',conn, if_exists = 'append', index = False)
            
        else:
            print('Error')
            missing.append([barrio, lon, lat, time])
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:09:18 2020

@author: Pablo Saldarriaga
"""
import json
import sqlite3
import pandas as pd
import datetime as dt
from consulta_darksky import darksky 
#%%

file_name = 'darksky_key.json'
with open(f'data/{file_name}', 'r') as f:
    info_key = json.load(f)
            
api_key = info_key['darksky']['key']
darksky_api = darksky(api_key)
#%%

### Realiza la conexion al sqlite donde se guarda la informacion climatica
file_name = 'data/clima.sqlite3'
conn = sqlite3.connect(file_name)

### Define el rango de fechas en el cual se busca la informacion
d_ini = dt.datetime(2017,1,1)
d_fin = dt.datetime(2017,12,31)

### Se obtiene un arreglo cuyas elementos son cada uno de los dias entre el
### rango de fechas dadas, ademas de incluir la informacion de los centroides
### de los barrios en donde se hara la busqueda
dates = pd.date_range(start=d_ini, end=d_fin, freq='1D')
centroides = pd.read_csv('data/centroides_barrios.csv', sep = ',')

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
            df[clima_cols].to_sql('clima',conn, if_exists = 'append', index = False)
            
        else:
            print('Error')
            missing.append([barrio, lon, lat, time])



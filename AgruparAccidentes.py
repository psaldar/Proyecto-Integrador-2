# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:45:38 2020

@author: nicol
"""
import pandas as pd
import datetime as dt
#%%
### Leer datos
datos = pd.read_csv('data/Accidentalidad_georreferenciada_2017.csv')

### Eliminamos los registros donde no tenemos informacion del barrio al cual
### corresponde
datos = datos.dropna(subset = ['BARRIO']).reset_index(drop = True)

### Limpieza de los nombres de los barrios
barrios = datos['BARRIO'].values
reemplazos = {' ':'',
              'á':'a',
              'é':'e',
              'í':'i',
              'ó':'o',
              'ú':'u',
              '.':'',
              ',':''}

barrios_limpios = []
for barrio in barrios:
    for reemplazo in reemplazos:
        barrio = str(barrio).replace(reemplazo,reemplazos[reemplazo])
        
    barrios_limpios.append(barrio)

datos['BARRIO'] =  barrios_limpios
datos['FECHA'] = datos['FECHA'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))

### elimino registros con barrios raros
barrios_raros = ['9086','6001','AUC1']
datos = datos[~datos['BARRIO'].isin(barrios_raros)].reset_index(drop = True)

### Coordenada centroide de accidentes para cada barrio
coords_centroids = datos[['BARRIO','X','Y']].groupby(['BARRIO']).mean().reset_index().rename(columns ={'X':'Lon',
                                                                                                       'Y':'Lat'})

coords_centroids.to_csv('data/centroides_barrios.csv', index = False, sep = ',')
### Asignar hora del dia para cada accidente (solo el entero de la hora)
def extraehora(HORA):
    if 'PM' in HORA:
        if '12:' not in HORA:
            hora = int(HORA[:2]) + 12
        else:
            hora = int(HORA[:2])
    else:
        if '12:' not in HORA:
            hora = int(HORA[:2])
        else:
            hora = 0
        
    return hora

datos['hour'] = datos['HORA'].apply(extraehora)
datos['TW'] = datos.apply(lambda x: pd.to_datetime(x['FECHA']) + dt.timedelta(hours = x['hour']), axis = 1)

### Agrupar por dia, hora y por barrio
agrupados = datos.groupby(['MES','DIA','hour','BARRIO']).agg(['count'])

### Miremos un barrio y hora en particular. Por ejemplo, la aguacatala, a las 5pm
agrupados1 = datos[datos['BARRIO']=='LaAguacatala']
agrupados2 = agrupados1[agrupados1['hour']==17].groupby(['MES','DIA']).agg(['count']) 

### Sera posible predecir choques usando las condiciones climaticas? 
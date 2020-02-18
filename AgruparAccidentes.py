# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:45:38 2020

@author: nicol
"""
import pandas as pd
import datetime as dt
import funciones
#%% Importacion de los datos

### Leer datos
datos = pd.read_csv('data/Accidentalidad_georreferenciada_2017.csv')

### Eliminamos los registros donde no tenemos informacion del barrio al cual
### corresponde
datos = datos.dropna(subset = ['BARRIO']).reset_index(drop = True).rename(columns ={'X':'Lon',
                                                                                    'Y':'Lat',
                                                                                    'DIA_NOMBRE':'Dia_sem',
                                                                                    'DIA': 'Dia',
                                                                                    'MES':'Mes'})

#%% Procesamiento de los datos

### Reemplazo para los dias de la semana
semana_limpios = []
semana = datos['Dia_sem'].values
reemplazos = {' ':'',
              'Á':'A',
              'É':'E'}

for dia in semana:
    for reemplazo in reemplazos:
        dia = str(dia).replace(reemplazo,reemplazos[reemplazo])
        
    semana_limpios.append(dia)

datos['Dia_sem'] = semana_limpios

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
datos['Hora'] = datos['HORA'].apply(funciones.extraehora)
datos['TW'] = datos.apply(lambda x: pd.to_datetime(x['FECHA']) + dt.timedelta(hours = x['Hora']), axis = 1)

#%%  elimino registros con barrios raros y obtengo la informacion del
### centroide de los barrios

barrios_raros = ['9086','6001','AUC1']
datos = datos[~datos['BARRIO'].isin(barrios_raros)].reset_index(drop = True)

### Coordenada centroide de accidentes para cada barrio
coords_centroids = datos[['BARRIO','Lon','Lat']].groupby(['BARRIO']).mean().reset_index()#.rename(columns ={'X':'Lon',
                                                                                     #                  'Y':'Lat'})

coords_centroids.to_csv('data/centroides_barrios.csv', index = False, sep = ',')
#%%

datos = datos.drop_duplicates(subset = ['TW','BARRIO'])

cols_ord = ['TW','BARRIO','Lat','Lon','Dia_sem','Mes','Dia','Hora']
datos[cols_ord].to_csv('data/proc_accidentes_2017.csv', index = False, sep = ',')

#%%    
### Agrupar por dia, hora y por barrio
agrupados = datos.groupby(['Mes','Dia','Hora','BARRIO']).agg(['count'])

### Miremos un barrio y hora en particular. Por ejemplo, la aguacatala, a las 5pm
agrupados1 = datos[datos['BARRIO']=='LaAguacatala']
agrupados2 = agrupados1[agrupados1['Hora']==17].groupby(['Mes','Dia']).agg(['count']) 

### Sera posible predecir choques usando las condiciones climaticas? 
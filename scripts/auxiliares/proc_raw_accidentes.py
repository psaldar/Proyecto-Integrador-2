# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:21:10 2020

@author: pasal
"""
import os
import json
import sqlite3
import pandas as pd
import datetime as dt
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
### importamos las funciones que se utilizan para el desarrollo del proyecto
import scripts.funciones as funciones
#%% Importacion de los datos

### Leer datos
datos = pd.read_csv('data/Accidentalidad_georreferenciada_2019.csv')

### Eliminamos los registros donde no tenemos informacion del barrio al cual
### corresponde

### Para 2018 y 2019
datos = datos.dropna(subset = ['BARRIO']).reset_index(drop = True).rename(columns ={'LONGITUD':'Lon',
                                                                                    'LATITUD':'Lat',
                                                                                    'DIA_NOMBRE':'Dia_sem',
                                                                                    'DIA': 'Dia',
                                                                                    'MES':'Mes'})

### Para 2017
# datos = datos.dropna(subset = ['BARRIO']).reset_index(drop = True).rename(columns ={'X':'Lon',
#                                                                                     'Y':'Lat',
#                                                                                     'DIA_NOMBRE':'Dia_sem',
#                                                                                     'DIA': 'Dia',
#                                                                                     'MES':'Mes'})

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
        barrio = str(barrio).lower().replace(reemplazo,reemplazos[reemplazo])
        
    barrios_limpios.append(barrio)

### agrego los nombres de los barrios procesados, al igual que organizar la
### fecha del accidente
datos['BARRIO'] =  barrios_limpios
datos['FECHA'] = datos['FECHA'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
datos['Hora_num'] = datos['HORA'].apply(funciones.extraehora)
datos['TW'] = datos.apply(lambda x: pd.to_datetime(x['FECHA']) + dt.timedelta(hours = x['Hora_num']), axis = 1)
#%%  elimino registros con barrios raros y obtengo la informacion del
### centroide de los barrios

barrios_raros = ['9086','6001','7001','9004','auc1','auc2','0']
datos = datos[~datos['BARRIO'].isin(barrios_raros)].reset_index(drop = True)

datos = datos.dropna(subset=['Lon','Lat']).reset_index(drop = True)

### Coordenada centroide de accidentes para cada barrio
coords_centroids = datos[['BARRIO','Lon','Lat']].groupby(['BARRIO']).mean().reset_index()#.rename(columns ={'X':'Lon',
                                                                                     #                  'Y':'Lat'})

coords_centroids.to_csv('data/centroides_barrios2019.csv', index = False, sep = ',')
#%%
### Organizamos para cada barrio, si ocurrio o no un accidente en la ventana de
### tiempo

datos_b = datos.drop_duplicates(subset = ['TW','BARRIO'])

cols_ord = ['TW','BARRIO','Lat','Lon','Dia_sem','Mes','Dia','Hora_num']
datos_b = datos_b[cols_ord].rename(columns = {'Hora_num':'Hora'})
datos_b.to_csv('data/proc_accidentes_2019.csv', index = False, sep = ',')
#%%
cols_datos = ['Lon', 'Lat', 'OBJECTID', 'RADICADO', 'HORA', 'Dia_sem', 'PERIODO',
               'CLASE', 'DIRECCION', 'DIRECCION_ENC', 'CBML', 'TIPO_GEOCOD',
               'GRAVEDAD', 'BARRIO', 'COMUNA', 'DISENO', 'Mes', 'Dia', 'FECHA',
               'MES_NOMBRE', 'Hora_num', 'TW']

datos = datos[cols_datos]
#%%
### Esto solo se corre si se tiene recopilada la informacion de las variables
### climaticas
db_name = 'database_pi2.sqlite3'
conn = sqlite3.connect(f'data/{db_name}')

datos.to_sql('raw_accidentes',conn, if_exists = 'append', index = False)
datos_b.to_sql('accidentes',conn, if_exists = 'append', index = False)
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:51:12 2020

@author: Pablo Saldarriaga
"""
import funciones
import pandas as pd
import datetime as dt
#%%
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
#%%

datos_group = datos[['FECHA','Hora','Lon']].groupby(['FECHA','Hora']).count().reset_index()
data_dia = pd.pivot_table(datos_group,values = 'Lon', index = 'FECHA', columns ='Hora').fillna(0)

cols = []
for col in data_dia.columns:
    cols.append('Hora_' + str(col))

data_dia.columns = cols
data_dia = data_dia.reset_index().rename(columns = {'index':'Dia'})
data_dia.to_csv('data/accidentes_dia.csv', sep = ',', index = False)
#%%
datos_group = datos[['BARRIO','Hora','Lon']].groupby(['BARRIO','Hora']).count().reset_index()
data_barrio = pd.pivot_table(datos_group,values = 'Lon', index = 'BARRIO', columns ='Hora').fillna(0)

cols = []
for col in data_barrio.columns:
    cols.append('Hora_' + str(col))

data_barrio.columns = cols
data_barrio = data_barrio.reset_index().rename(columns = {'index':'BARRIO'})
data_barrio.to_csv('data/accidentes_barrio.csv', sep = ',', index = False)
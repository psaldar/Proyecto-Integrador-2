# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:40:06 2018

@author: cmejia
"""
import os
current_dir = os.getcwd()


#%%
import json

file_name = 'conf.json'
path = os.path.join(current_dir, f'{file_name}')
with open(path, 'r') as f:
    info_conf = json.load(f)
            
base_path = info_conf['base_path']

os.chdir('C:/Users/Lenovo/Documents/BigBangData/NeuiJuan/Model_V_1.8/iRain')
from pathlib import Path

from get_data import get_lat_lon_file
from datetime import datetime
from dateutil import tz
import time


import pandas as pd
import datetime as dt
import numpy as np
import geopandas as gpd

os.chdir(base_path)
from essentials.transferencia_mysql import save_sql
from shapely.geometry import Polygon, Point
import essentials.funciones_esenciales as funciones


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


file_name = 'tables_mysql.json'
path = os.path.join(base_path, f'essentials/{file_name}')
with open(path, 'r') as f:
    tables_info = json.load(f)
#%%
import logging
 
logging.basicConfig(
level=logging.DEBUG,
handlers=[
# logging.FileHandler("my_log.log"),
logging.StreamHandler(),
],
# filename=Path().resolve() / 'my_log.log',
# format='%(asctime)s - %(name)s - %(levelname)s - %(func)s - %(message)s',
format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger()
#%%

#%% RECTANGULO VALLE DEL CAUCA
def descargar_iRain(d_ini, d_fin,now_date):
      
    path = Path('C:/Users/Lenovo/Documents/BigBangData/NeuiJuan/Model_V_1.8/iRain')
    path2 = Path('C:/Users/Lenovo/Documents/BigBangData/NeuiJuan/data/iRain/temp')
    
    to_zone = tz.gettz('UTC')
    from_zone = tz.gettz('America/Bogota')
    
    ulx = -77.656
    uly = 5.056
    lry=2.930
    lrx = -75.546
    
    ix = pd.date_range(start=d_ini, end=d_fin, freq='H')
    df_ = pd.DataFrame(index=ix)
    data_resource = df_.reset_index()
    
    list_error = []
    results = []
    for date in data_resource['index']:
        aux = date.strftime("%Y%m%d%H")
        utc_col = datetime.strptime(aux,"%Y%m%d%H")
        utc_col = utc_col.replace(tzinfo=from_zone)
        utc = utc_col.astimezone(to_zone)
        startDate = utc.strftime("%Y%m%d%H")
        date_file = date.strftime("%Y-%m-%dT%H:%M")
        try:
            result = get_lat_lon_file(path,startDate, startDate,date_file, ulx,uly,lry,lrx)
        except Exception as e:
            print(e)
            list_error.append(date)
            continue
        result['TW'] = date
        results.append(result)
        time.sleep(3)
    
    try:
        date_file_name = now_date.strftime("%Y-%m-%d %H")
        results = pd.concat(results)
        path_file = path2 / f'Act_{date_file_name}.csv'
        results.to_csv(path_file,index = False, sep = '\t' ,decimal = ',')
    except Exception as e:
        print(e)

    if len(list_error)>0:
        pd.DataFrame(list_error).to_csv(f'Errores_descargando_en_{date_file_name}.csv')

    return None

#%%
def data_iRain(d_ini, d_fin):
           
    file_path = 'data/iRain/temp'
    dir_rain_file = base_path + '/'+ file_path   
    
    files = pd.date_range(start=d_ini, end=d_fin, freq='D').strftime("%Y-%m-%d")
    hora = d_fin.strftime("%H")
    rain = pd.DataFrame()
    logger.info("Leyendo archivos de iRain")
    for f in files:
        rain = rain.append(pd.read_csv(dir_rain_file + '/Act_' + f + ' '+hora + '.csv', sep = "\t")).dropna()   
        
    rain['lon'] = rain.apply(lambda x: float(x['LON'].split(sep = ',')[0] +'.'+ x['LON'].split(sep = ',')[1] ),axis = 1)
    rain['lat'] = rain.apply(lambda x: float(x['LAT'].split(sep = ',')[0] +'.'+ x['LAT'].split(sep = ',')[1] ),axis = 1)
    
    geometry = [Point(xy) for xy in zip(rain.lon.values,rain.lat.values)]
    rain_gdf = gpd.GeoDataFrame(rain, geometry = geometry, crs={'init' :'epsg:4326'}).drop(columns = {'LON','LAT','FECHAHORA'})
    
    return rain_gdf

def construir_pixel(X):    
    c = Polygon([(X['x1'],X['y1']), (X['x2'],X['y2']), (X['x3'],X['y3']),
                 (X['x4'],X['y4']),(X['x5'],X['y5'])])
    return c

def construir_pixel_2(X): 
    minx = min([X['x1'],X['x2'],X['x3'],X['x4'],X['x5']])
    maxx = max([X['x1'],X['x2'],X['x3'],X['x4'],X['x5']])
    miny = min([X['y1'],X['y2'],X['y3'],X['y4'],X['y5']])
    maxy = max([X['y1'],X['y2'],X['y3'],X['y4'],X['y5']])
    
    c = Polygon([(minx,miny), (maxx,miny), (maxx,maxy),(minx,maxy)])
    return c

def pixeles_gdf():
    
    pixeles = pd.read_csv('Model_V_1.8/iRain/iRain_pixels.csv',sep = ';')
    pixeles['Poligono'] = pixeles.apply(construir_pixel_2,axis = 1)
    pixeles_gdf = gpd.GeoDataFrame(pixeles, geometry = pixeles.Poligono, crs={'init' :'epsg:4326'})
    pixeles_gdf = pixeles_gdf[['Pixel','geometry']]
    
    return pixeles_gdf


def iRain_pixeles(d_ini,d_fin):
    rain = data_iRain(d_ini, d_fin)
    rain['geometry'] = rain['geometry']#.buffer(0.001)
    pixeles = pixeles_gdf()
    
    rain_base = rain.drop_duplicates(subset = ['lon','lat'])
    rain_base['geometry_2'] = pixeles['geometry'].copy()

    rain_aux = gpd.sjoin(rain,rain_base[['geometry','geometry_2']], how ='left')
    rain_aux['geometry'] = rain_aux['geometry_2']
    info = rain_aux[['TW','PRECIP','geometry']]
    
    
    return info

def read_Precipitacion(d_ini,d_fin):
    rain = iRain_pixeles(d_ini,d_fin)
    circuitos  = funciones.read_tese()
    
    data = gpd.sjoin(circuitos,rain, how = 'left').fillna(0)
    data = data[['TW','INSTALACION_ORIGEN','PRECIP']]
    data['TW'] = pd.to_datetime(data['TW'])
    data['INSTALACION_ORIGEN'] = (data['INSTALACION_ORIGEN']).astype('int64')
    data_precip = data.groupby(['TW','INSTALACION_ORIGEN']).mean().reset_index()
    
    return data_precip

def organizar_precipitacion(d_ini,d_fin,n_lags):
    precipitacion = read_Precipitacion(d_ini,d_fin)
    
    tw = precipitacion['TW'].drop_duplicates()
    data = precipitacion[~precipitacion['TW'].isin(tw[0:n_lags])]
#    tw =tw[n_lags:]
    data['TW'] = pd.to_datetime(data['TW'])
    #data = data.sort_values(by = ['TW','INSTALACION_ORIGEN'])
    
    cols = ['PRECIP']
       
    for lag in range(0,n_lags):
        logger.info(f"creando lag {lag+1}")
        cols.append('PRECIP-'+str(lag+1))
        aux = precipitacion.copy().rename(columns = {'PRECIP':'PRECIP-'+str(lag+1)})
        aux['TW'] = (pd.to_datetime(aux['TW']) + dt.timedelta(hours=lag + 1))
        data = data.merge(aux, how = 'left', on =['TW','INSTALACION_ORIGEN']).fillna(0)
    
    data['TW'] = data['TW'] + dt.timedelta(hours = 1) #data.drop(columns = 'PRECIP')
    data = data.rename(columns = {'PRECIP':'PRECIP-1','PRECIP-1':'PRECIP-2','PRECIP-2':'PRECIP-3','PRECIP-3':'PRECIP-4'})
    return data

#%%

def iRain_hora(now_date):

    day = now_date
    d_ini = (now_date - dt.timedelta(hours = 5)).strftime('%Y-%m-%d %H:00:00')
    d_fin = (now_date - dt.timedelta(hours = 2)).strftime('%Y-%m-%d %H:00:00')
    
    descargar_iRain(d_ini, d_fin,now_date)
    precipitacion =  organizar_precipitacion(day,day,3)
    return precipitacion

#%%
#d_ini =  dt.datetime(2019,3, 8,13,0,0)
#d_fin =  dt.datetime.now() - dt.timedelta(hours= 1)
#
#ix = pd.date_range(start=d_ini, end=d_fin, freq='H')
##
#for date in ix:
#    res = iRain_hora(date)
#    save_sql(res, 'iRain', tables_info['clima']['index'])

now_date = dt.datetime.now() - dt.timedelta(hours= 1)
res = iRain_hora(now_date)

save_sql(res, 'iRain', tables_info['clima']['index'])




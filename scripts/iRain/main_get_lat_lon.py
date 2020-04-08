# -*- coding: utf-8 -*-
"""
Created on Mon Apr 6 20:40:06 2019

@author: Pablo Saldarriaga
"""
import sys
import os
import json
current_dir = os.getcwd()
#%%
file_name = 'conf.json'
path = os.path.join(current_dir, f'{file_name}')
with open(path, 'r') as f:
    info_conf = json.load(f)
            
base_path = info_conf['base_path']
os.chdir(base_path)
sys.path.insert(0, base_path)
#%%
import time
import sqlite3
import numpy as np
import pandas as pd 
import datetime as dt
from dateutil import tz
from pathlib import Path
from datetime import datetime
from scripts.iRain.get_data import get_lat_lon_file
#%%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
import logging
logger = logging.getLogger()
#%% Rectangulo Medellin

def descargar_iRain(d_ini, d_fin):
    
    folder = 'f1'
    path = Path(base_path + f'/scripts/iRain/{folder}')
    path2 = Path(base_path +'/data/iRain')
    
    db_name = 'irain_info.sqlite3'
    conn = sqlite3.connect(f'data/{db_name}')
    
    to_zone = tz.gettz('UTC')
    from_zone = tz.gettz('America/Bogota')
    
    ulx = -75.733
    lrx = -75.413
    lry = 6.084
    uly = 6.404
    
    ix = pd.date_range(start=d_ini, end=d_fin, freq='H').drop(d_fin)
    df_ = pd.DataFrame(index=ix)
    data_resource = df_.reset_index()
    
    list_error = []
    results = []
    for date in data_resource['index']:
        print(date)
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
        date_file_name = date.strftime("%Y-%m-%d")
        results = pd.concat(results)
        path_file = path2 / f'info_piloto/{date_file_name}.csv'
        #results.to_csv(path_file,index = False, sep = '\t' ,decimal = ',')
        #results['TW'] = pd.to_datetime(result['FECHAHORA'], format = '%Y-%m-%dT%H:%M')
        results.to_sql('irain',conn, if_exists = 'append', index = False)
    except Exception as e:
        print(e)

    if len(list_error)>0:
        pd.DataFrame(list_error).to_csv(f'data/iRain/Errores_descargando_en_{date_file_name}.csv')

    return None
#%%
d_ini = dt.datetime(2017, 1, 1,0,0,0)
d_fin = dt.datetime(2020, 1, 1,0,0,0)

times = pd.date_range(start=d_ini, end=d_fin, freq='D')

for i in range(len(times)-1):
    print(times[i])
    descargar_iRain(times[i], times[i+1])
    
    
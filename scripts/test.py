# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:46:02 2020

@author: Pablo Saldarriaga
"""
import os
import sys
import json
import pandas as pd
import datetime as dt
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
import scripts.funciones as funciones
#%%
import logging
from logging.handlers import RotatingFileHandler

file_name = 'test'
logger = logging.getLogger()
dir_log = os.path.join(base_path, f'logs/{file_name}.log')

### Crea la carpeta de logs
if not os.path.isdir('logs'):
    os.makedirs('logs', exist_ok=True) 

handler = RotatingFileHandler(dir_log, maxBytes=2000000, backupCount=10)
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
                    handlers = [handler])
#%%
def main(d_ini, d_fin):
    
    version = 'ver002'    
    mod_version = funciones.carga_model(base_path, f'models/{version}', version)
    
    if 'model' in mod_version:
        logger.info("Model loaded")
        mod = mod_version['model'].steps[0][1]
        model_sel = mod_version['model'].steps[1][1]
    else:
        logger.error("No model found")
        return None
    
    data = funciones.read_clima_accidentes(d_ini, d_fin)
    data_org = funciones.organizar_data_infoClima(data)
    
    
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
    
    data_org = data_org[data_org['BARRIO'].isin(poblado)]
    
    data_org['poblado'] = data_org['BARRIO']
    data_org= pd.get_dummies(data_org, columns=['poblado'])
    
    X = data_org.drop(columns = ['TW','BARRIO','Accidente','summary'])
    Y = data_org['Accidente']

    preds_ff = mod.predict(X, model_sel)
    preds_ff['Accidente'] = Y
    
    #graph the results
    funciones.graphs_evaluation_test(f'{base_path}/models/{version}',  preds_ff, save = True)
    funciones.precision_recall_graph_test(f'{base_path}/models/{version}',  preds_ff, save = True)
    
    bound = [0.1,0.2,0.3,0.4,0.5,0.6]  
    for b in bound:
        funciones.matrix_confusion_test(f'{base_path}/models/{version}', preds_ff,b, save = True)    
    
    return None

if __name__ == '__main__':
    
    d_ini = dt.datetime(2019,7,1)
    d_fin = dt.datetime(2020,1,1)    
    main(d_ini, d_fin)
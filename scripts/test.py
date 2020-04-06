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
    
    version = 'ver010p2'    
    mod_version = funciones.carga_model(base_path, f'models/{version}', version)
    
    if 'model' in mod_version:
        logger.info("Model loaded")
        mod = mod_version['model'].steps[0][1]
        model_sel = mod_version['model'].steps[1][1]
    else:
        logger.error("No model found")
        return None
    
    data = funciones.read_clima_accidentes(d_ini, d_fin, poblado = True)
    data_org = funciones.organizar_data_infoClima(data)
    
    ### agregamos la informacion relacionada a la cantidad de accidentes ocurridas
    ### en las ultimas X horas
    
    ### En caso que se considere acumulado de fallas, realiza la validacion
    ### si el modelo entrenado tiene la frecuencia utilizada
    try:
        freq1 = mod.freq1
        freq2 = mod.freq2
    except Exception as e:
        logger.info(f'Problemas con las frecuencias de las senales {e}')
        freq1 = '1H'
        freq2 = '1H'
    
    d_ini_acc = d_ini - dt.timedelta(hours = int(freq2.replace('H', '')))
    raw_accidentes = funciones.read_accidentes(d_ini_acc, d_fin)
    
    ### Agrega senal a corto plazo
    data_org = funciones.obtener_accidentes_acumulados(data_org, 
                                                        raw_accidentes, 
                                                        freq = freq1)
    
    ### Agrega senal a largo plazo
    data_org = funciones.obtener_accidentes_acumulados(data_org, 
                                                        raw_accidentes, 
                                                        freq = freq2)
    
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
    
    d_ini = dt.datetime(2019,8,1)
    d_fin = dt.datetime(2020,1,1)    
    main(d_ini, d_fin)
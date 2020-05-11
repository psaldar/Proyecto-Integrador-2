# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:46:02 2020

@author: Pablo Saldarriaga
"""
import os
import sys
import pandas as pd
import datetime as dt
#%%
### Realizamos el cambio de directoroi de trabajo al "Directorio Base"
current_dir = os.getcwd()
base_path = os.path.dirname(current_dir)

os.chdir(base_path)
sys.path.insert(0, base_path)
#%%
import scripts.funciones as funciones
#%%
#Define el archivo en el que se guararan los logs del codigo
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
    
    ### Realizamos la carga del modelo entrenado con el script training.py,
    ### importamos tanto el mejor modelo como el objeto de clase modelo 
    ### utilizado durante el entrenamiento
    version = 'ver012NNPabs'    
    mod_version = funciones.carga_model(base_path, f'models/{version}', version)
    
    if 'model' in mod_version:
        logger.info("Model loaded")
        mod = mod_version['model'].steps[0][1]
        model_sel = mod_version['model'].steps[1][1]
    else:
        logger.error("No model found")
        return None
    
    ### Realizamos la lectura de la informacion climatica en el rango de fechas
    ### especificado, incluye la etiqueta de si ocurre o no un accidente. 
    ### Posteriormente, en la organizacion de la informacion climatica, lo
    ### que se hace es agregar las variables con la informacion distribucional
    ### de las ultimas 5 horas de la info climatica    
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
        freq1 = '1D'
        freq2 = '1D'
    
    ### agregamos la informacion relacionada a la cantidad de accidentes ocurridas
    ### en las ultimas X horas    
    d_ini_acc = d_ini - dt.timedelta(days = int(freq2.replace('D', '')))
    raw_accidentes = funciones.read_accidentes(d_ini_acc, d_fin)
    
    ### Agrega senal a corto plazo
    data_org = funciones.obtener_accidentes_acumulados(data_org, 
                                                        raw_accidentes, 
                                                        freq = freq1)
    
    ### Agrega senal a largo plazo
    data_org = funciones.obtener_accidentes_acumulados(data_org, 
                                                        raw_accidentes, 
                                                        freq = freq2)

    ### Convertimos la bariable de Barrios en variable dummy para ser incluida
    ### en el modelo    
    data_org['poblado'] = data_org['BARRIO']
    data_org= pd.get_dummies(data_org, columns=['poblado'])
    
    ### Filtro del 31 de diciembre
    data_org = data_org[data_org['TW']<dt.datetime(2019,12,31)].reset_index(drop = True)
    ### Relizamos la particion del conjunto de datos en las variables
    ### explicativas (X) y la variable respuesta (Y)    
    X = data_org.drop(columns = ['TW','BARRIO','Accidente','summary'])
    Y = data_org['Accidente']
    
    
    ### El modelo realiza la prediccion con el conjunto de datos, para esto los
    ### parametros de la funcion son el conjunto de datos, y el modelo que 
    ### se quiere utilizar
    preds_ff = mod.predict(X, model_sel, save = True)
    preds_ff['Accidente'] = Y


    ### Guardo congunto de test
    
    X_test_save = pd.read_csv('data/test.csv')
    X_test_save['Accidente'] = Y
    X_test_save['BARRIO'] = data_org['BARRIO'].values
    X_test_save['TW'] = data_org['TW'].values
    
    X_test_save.to_csv('data/test.csv', index = False, sep = ',')
    
    ### Realizamos las gracias de violines, roc y precision-recall en el conjunto
    ### de datos de prueba
    funciones.graphs_evaluation_test(f'{base_path}/models/{version}',  preds_ff, save = True)
    funciones.precision_recall_graph_test(f'{base_path}/models/{version}',  preds_ff, save = True)
    
    bound = [0.1,0.2,0.3,0.4,0.5,0.6]  
    for b in bound:
        funciones.matrix_confusion_test(f'{base_path}/models/{version}', preds_ff,b, save = True)    
    
    return None

if __name__ == '__main__':
    
    ### Definimos el rango de fechas en el cual realizaremos la prediccion de
    ### accidentalidad con el modelo entrenado
    d_ini = dt.datetime(2019,8,1)
    d_fin = dt.datetime(2020,1,1)    
    main(d_ini, d_fin)
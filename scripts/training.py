# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:19:43 2020

@author: Pablo Saldarriaga
"""
import os
import json
import numpy as np
import pandas as pd
import datetime as dt
import multiprocessing
#%%
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
#%%
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#%%
current_dir = os.getcwd()

file_name = 'conf.json'
path = os.path.join(current_dir, f'{file_name}')
with open(path, 'r') as f:
    info_conf = json.load(f)
            
base_path = info_conf['base_path']
os.chdir(base_path)

import sys
sys.path.insert(0, base_path)
#%%
#Define el archivo en el que se guararan los logs del codigo
import logging
from logging.handlers import RotatingFileHandler

file_name = 'model_Training'
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
import scripts.funciones as funciones
from scripts.clase_model.modelo import Modelo
#%%

def main(d_ini, d_fin):
    
    version = 'ver010p2'
    now_date = dt.datetime.now()
    
    cv = 3
    freq1 = '8H'
    freq2 = '168H'
    balance = 'rus'
    score = 'roc_auc'
    prop_deseada_under = 0.4
    n_proc = multiprocessing.cpu_count() -1
    
    descripcion = f""" Entrena modelo para realizar la prediccion de accidentes
                       en los barrios del Poblado. considera solo variables
                       de hora, dia semana y climaticas con sus means. 
                       Adicional, considera un acumulado de accidentes
                       considerando una frecuencia de {freq1}-{freq2}. Entrena en las
                       fechas {d_ini}-{d_fin}. {balance}-{score}-{prop_deseada_under}.
                       Entrena el modelo ganador de la version 10 agregando la senal
                       de fallas"""
                       
    mod = Modelo(now_date, version, base_path, descripcion)
    
    mod.freq1 = freq1
    mod.freq2 = freq2
    
    layers_nn = []

    layer_lim_max = 7
    layer_lim_min = 4
 
    nodes_lim_max = 128
    nodes_lim_min = 6
 
    iter_max = 4
 
    for _ in range(iter_max):
        for size in range(layer_lim_min, layer_lim_max + 1, 2):
            vec = tuple(np.random.randint(nodes_lim_min, nodes_lim_max, size))
            layers_nn.append(vec)
          
    layers_nn = list(set(layers_nn))
    
    # models = {
    #                'logistic':{
    #                            'mod':LogisticRegression(random_state = 42),
    #                            'par':{
    #                                'penalty': ('l1','l2'),
    #                                'solver': ('saga','lbfgs')
                                 
    #                            }
    #                },
    #                 'ridge_log':{
    #                             'mod':RidgeClassifier(random_state = 42),
    #                             'par':{
    #                                 'alpha':[0.2, 0.4, 0.6, 0.8, 1],
    #                                 'solver': ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga')
    #                             }
                         
    #                 },
    #                 'naiveBayes':{
    #                             'mod':GaussianNB(),
    #                             'par':{}
                         
    #                 },
    #                 'bernoulli':{
    #                             'mod':BernoulliNB(),
    #                             'par':{
    #                                 'fit_prior':[True, False],
    #                                 'alpha': [0,0.2,0.4,0.6,0.8,1]
    #                             }
                         
    #                 },
    #                 'qda':{
    #                             'mod':QuadraticDiscriminantAnalysis(),
    #                             'par':{
    #                                 'reg_param':[0,0.3,0.5,0.7,0.9]
    #                             }
                         
    #                 },
    #                 'nn':{
    #                             'mod' : MLPClassifier( solver = 'adam',shuffle = True, random_state= 42),
    #                             'par':{
    #                                 'hidden_layer_sizes' : layers_nn,
    #                                 'activation' : ('logistic', 'relu','tanh','identity'),
    #                                 'learning_rate_init': [0.001,0.01,0.1,0.3,0.5,0.9],
    #                                 'alpha':[0.05, 0.1, 0.5 , 3, 5, 10, 20]
    #                                 }
                         
    #                },
    #                'rforest':{
    #                          'mod': RandomForestClassifier(random_state= 42),
    #                          'par': {'n_estimators':[10,20,30,40,50,60,70,80,90,100,200,300,400,500],
    #                                  'max_depth': [None, 2, 4, 6, 8, 10, 20, 30],
    #                                  'criterion':('gini','entropy'),
    #                                  'bootstrap': [True,False]
    #                                  }
    #                },
    #                'xtree':{
    #                          'mod': ExtraTreesClassifier(random_state = 42),
    #                          'par': {'n_estimators':[10,20,30,40,50,60,70,80,90,100,200,300,400,500],
    #                                  'max_depth':[None, 2, 4, 6, 8, 10, 20, 30],
    #                                  'criterion':('gini','entropy'),
    #                                  'bootstrap': [True,False]}                     
    #                },
    #                'gradient':{
    #                            'mod' : GradientBoostingClassifier(random_state = 42),
    #                            'par' : {'loss' : ('deviance', 'exponential'),
    #                                    'n_estimators': [10,20,30,40,50,60,70,80,90,100,200,300,400,500],
    #                                    'max_depth' : [3, 4, 5, 6, 7, 8, 9],
    #                                    'learning_rate':[0.1,0.3,0.5,0.7,0.9]
    #                                    }
    #                },
    #                'xgboost':{
    #                        'mod':XGBClassifier(random_state = 42),
    #                        'par':{
    #                              'n_estimators':[10,20,30,40,50,60,70,80,90,100,200,300,400,500],
    #                              'max_depth': [ 2, 4, 6, 8, 10, 20, 30]
    #                            }
    #                        }
    #           }
    
    models = {
                  'xgboost':{
                           'mod':XGBClassifier(random_state = 42,n_estimators =100,max_depth=4 ),
                           'par':{ }
                           }
              }
    
    mod.models = models
    
    data = funciones.read_clima_accidentes(d_ini, d_fin, poblado = True)
    data_org = funciones.organizar_data_infoClima(data)
    
    
    ### agregamos la informacion relacionada a la cantidad de accidentes ocurridas
    ### en las ultimas X horas
    
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
    
    
    ### Conservo solo climaticas y variables relevantes
    
    vars_ele = ['precipIntensity',
                'precipProbability', 'temperature', 'apparentTemperature', 'dewPoint',
                'humidity', 'windSpeed', 'cloudCover',  
                'uvIndex', 'visibility', 'poblado_alejandria', f'cumAcc_{freq1}',f'cumAcc_{freq2}', 
                'poblado_altosdelpoblado', 'poblado_astorga', 'poblado_castropol', 
                'poblado_elcastillo', 'poblado_eldiamanteno2', 
                'poblado_laaguacatala', 'poblado_lalinde', 'poblado_losbalsosno1', 
                'poblado_losnaranjos', 'poblado_manila', 'poblado_sanlucas', 
                'poblado_santamariadelosangeles', 'poblado_villacarlota', 'hora_1', 
                'hora_2', 'hora_3', 'hora_6', 'hora_7', 'hora_9', 'hora_10', 
                'hora_14', 'hora_16', 'hora_19', 'hora_20', 'hora_21', 
                'hora_23', 'icon_partly-cloudy-day', 
                'icon_partly-cloudy-night', 'icon_rain', 'dia_sem_3', 'dia_sem_4', 
                'dia_sem_5', 'dia_sem_6', 'humidity_mean', 'windSpeed_mean',]
    X = X[vars_ele]    
    ### Fin
    
    ### Caso sin var climaticas
    # vars_ele = ['hora_0', 'hora_1', 'hora_2', 'hora_3', 'hora_4', 'hora_5',
    #             'hora_6', 'hora_7', 'hora_8', 'hora_9', 'hora_10', 'hora_11', 'hora_12',
    #             'hora_13', 'hora_14', 'hora_15', 'hora_16', 'hora_17', 'hora_18',
    #             'hora_19', 'hora_20', 'hora_21', 'hora_22', 'hora_23','dia_sem_0',
    #             'dia_sem_1', 'dia_sem_2', 'dia_sem_3', 'dia_sem_4', 'dia_sem_5',
    #             'dia_sem_6','poblado_alejandria', 
    #             'poblado_altosdelpoblado', 'poblado_astorga', 'poblado_castropol', 
    #             'poblado_elcastillo', 'poblado_eldiamanteno2', 
    #             'poblado_laaguacatala', 'poblado_lalinde', 'poblado_losbalsosno1', 
    #             'poblado_losnaranjos', 'poblado_manila', 'poblado_sanlucas', 
    #             'poblado_santamariadelosangeles', 'poblado_villacarlota']
    
    # X = X[vars_ele]
    ## Fin
    
    Y = data_org['Accidente']    
    
    X_test, Y_test, models, selected = mod.train(X, 
                                                 Y, 
                                                 cv = cv,
                                                 score = score,
                                                 n_proc = n_proc,
                                                 balance = balance,
                                                 prop_deseada_under = prop_deseada_under)
       
 
    #Realiza la prediccion de las fallas en un conjunto de datos de prueba
    model_sel = models[selected]['bestModel']
    preds_ff = mod.predict(X_test, model_sel)
    preds_ff['Accidente'] = Y_test
    #Realiza graficas de la curva ROC-AUC y diagramas de violin que permitan
    #analizar el comportamiento y deseméño del modelo
    funciones.graphs_evaluation(f'{base_path}/models/{version}', selected, preds_ff, save = True)
    funciones.precision_recall_graph(f'{base_path}/models/{version}', selected, preds_ff, save = True)
    
    #Umbrales
    bound = [0.1,0.2,0.3,0.4,0.5,0.6]
    
    #Obtencion de matrices de confusion para diferentes umbrales de la predicicon
    for b in bound:
        funciones.matrix_confusion(f'{base_path}/models/{version}', 
                                   selected, preds_ff, b,  save=True)
    
    
    # Guarda el modelo elegido y el objeto de clase modelo como parte de un
    #pipeline    
    mod_pipe = Pipeline([('procesador', mod),
                          ('modelo', models[selected]['bestModel'])])
    
    path_best_mod = os.path.join(f'{base_path}/models/{version}', f"{version}.sav")
    
    logger.info(f"Saving model for version {version}")
    joblib.dump(mod_pipe, path_best_mod)    
    
    return None


if __name__ == '__main__':
    
    d_ini = dt.datetime(2017,6,1)
    d_fin = dt.datetime(2019,8,1)    
    main(d_ini, d_fin)

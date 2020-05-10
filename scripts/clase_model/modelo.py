# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:33:08 2020

@author: Pablo Saldarriaga
"""
#%%
import os
import logging

logger = logging.getLogger(__name__)
#%%
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

import scripts.funciones as funciones
#%%

class Modelo:
    
    # define el constructor de la clase, inicializa como campos vacios tanto
    # el diccionario de modelos, como el mejor modelo seleccionado. adicional
    # a esto, verifica si el directorio con el nombre de la version del modelo
    # esta creado, en caso de que no, lo crea
    def __init__(self, now_date, version, base_path, desc):
        logger.info("Inicializa el objeto del Clasificador")
        
        self.now_date = now_date
        self.base_path = base_path        
        self.description = desc        
        self.version = version        
        self.model_name = None
        self.best_model = ''
        self.models = {}        
        self.cols_order = []
        self.models_path = base_path + '/models'                                 
        self.path_version = self.models_path + '/' + version
        self.kwargs = {}
        self.cols_outliers = []
        self.mean = None
        self.cutoff = None
        self.covariance = None
        self.inv_covariance = None
        self.freq1 = '1H'
        self.freq2 = '1H'
        

        if not os.path.isdir(self.models_path):
            logger.info("directory doesn't exist, creating it now")
            os.makedirs(self.models_path, exist_ok=True)  
                    
        if not os.path.isdir(self.path_version):
            logger.info("directory doesn't exist, creating it now")
            os.makedirs(self.path_version, exist_ok=True)  
    
    
    # Esta funcion se encarga de entrenar los modelos en el diccionario de
    # modelos, ademas de elegir el mejor modelo para ser utilizado
    def train(self,X,Y,score = 'roc_auc',cv = 2,n_proc = 1,desc_model_sav = '',
              prop_deseada_under = 0.2,balance = 'rus',**kwargs):
        
        # Realiza una particion de los datos de entrenamiento, se utiliza un
        # 80% de los datos y un 20% para ver el desempeño del conjunto de
        # entrenamiento
        logger.info("Parte un conjunto de entrenamiento y test dentro del entrenamiento")
        X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                            Y,
                                                            stratify = Y,
                                                            test_size = 0.2,
                                                            random_state = 42)

        can_0 = len(Y_train) - Y_train.sum()
        can_1 = Y_train.sum()
        logger.info(f"""Data: (initial) Proporcion 0 es {round((can_0/len(Y_train))*100, 2)}%
                        y de 1 es {round((can_1/len(Y_train))*100, 2)}%""")       
        

        tra_0 = int(len(Y_train) - Y_train.sum())
        tra_1 = int(Y_train.sum())
        
        mul_updown = (tra_0 * prop_deseada_under - tra_1 * (1 - prop_deseada_under)) / (tra_0 * prop_deseada_under)    
        fac_1 = int(tra_0 * (1 - mul_updown))
        
        if balance == 'rus':
        # keeps original sample size for Failures 1
            ratio_u = {0 : fac_1, 1 : tra_1}
            rus = RandomUnderSampler(sampling_strategy = ratio_u, random_state=42)
            
            os_X_tt, os_Y_tt = rus.fit_sample(X_train, Y_train)
        
        logger.info("Undersampling done")
        can_0 = len(os_Y_tt) - sum(os_Y_tt)
        can_1 = sum(os_Y_tt)
        logger.info(f"""Data: (train after under) Proporcion 0 es {round((can_0/len(os_Y_tt))*100, 2)}%
                        y de 1 es {round((can_1/len(os_Y_tt))*100, 2)}%""")
        logger.info(f"Data: (train after under) Datos 0 son {can_0} y Datos 1 son {can_1}")
        
        ### Guardar data de entrenamiento y validacion
        
        X_train_save = os_X_tt.reset_index(drop = True)
        X_train_save['Accidente'] = os_Y_tt.values

        X_val_save = X_test.reset_index(drop = True)
        X_val_save['Accidente'] = Y_test.values
        
        X_train_save.to_csv('data/training.csv', index = False, sep = ',')
        X_val_save.to_csv('data/validation.csv', index = False, sep = ',')
        
        # Entrena los diferentes modelos utilizando grid search, ademas de ir
        # guardando el mejor modelo de cada una de las familias de modelos que
        # se consideran en el diciconario de modelos
        most_date = dt.datetime.now()
        models, selected = funciones.grid(self.base_path, most_date, self.path_version,
                                          os_X_tt, os_Y_tt,X_test,Y_test, self.models, 
                                          score = score, cv = cv, n_proc = n_proc)
        
        # Asigna al objeto el mejor modelo, ademas de establecer el orden en el
        # cual se deben de pasar las variables al modelo entrenado
        self.model_name = f"{selected}_{desc_model_sav}_{most_date.strftime('%Y%m%d_%H%M')}"
        self.best_model = selected
        self.cols_order = list(X.columns)  
        
        return X_test, Y_test, models, selected
    
    def transform():
        pass
    
    # Se define la funcion fit para poder guardar el objeto creado dentro del
    # pipeline
    def fit(self, X, y = None, **kwargs):
        pass
    
    # Esta funcion realiza la prediccion de la clasificacion de los avisos, 
    # toma como entrada los datos organizados retornados por la funcion
    # transform. Revisa que se tenga información para todas las variables
    # con las que fue entrenado el modelo, en caso de que se tengan variables
    # faltantes, se crean y se les asigna un valor de 0
    def predict(self, X,model,**kwargs):
        
        cols_order = self.cols_order
        data_cols = list(X.columns)
        
        # Revisa si hay mas variables en la informacion utilizada para entrenar
        # el modelo que en la informacion de prueba

        logger.info("Hay mas variables en los datos de entrenamiento que en los datos de prueba")
        aux = pd.DataFrame(cols_order, columns = ['field'])
        idx = ~aux['field'].isin(data_cols)
        missing_cols = aux[idx]['field'].values
        
        #En caso de tener variables faltantes, se crean
        for col_name in missing_cols:
            logger.info(f"Falta variable: {col_name}, se agrega y se llena con 0s")
            X[col_name] = 0
        
        # Revisa si se pasaron a la funcion mas variables de las necesarias 
        # para hacer la prediccion

        logger.info("Hay variables extras en los datos de prueba")
        aux = pd.DataFrame(data_cols, columns = ['field'])
        idx = ~aux['field'].isin(cols_order)
        extra_columns = aux[idx]['field'].values
        for col_name in extra_columns:
            logger.info(f"Se tiene la variable extra: {col_name}")
        
        # Toma unicamente las variables que fueron consideradas en el 
        # entrenamiento del modelo
        X_val= X[cols_order]
        
        # En caso de que no se tenga informacion para realizar la prediccion,
        # la funcion retorna vacio
        if X_val.empty:
            logger.error(f"No data to predict")
            return None
        
        # Realiza la prediccion del modelo
        prob = model.predict_proba(X_val)
        
        preds_ff = X.loc[X_val.index]
        preds_ff['Predicted'] = prob[:, 1]     

        return preds_ff
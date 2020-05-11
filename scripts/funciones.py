# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:05:55 2020

@author: Pablo Saldarriaga
"""
import os
import time
import sqlite3
import holidays
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#%%
import logging
logger = logging.getLogger(__name__)
#%%
### Asignar hora del dia para cada accidente (solo el entero de la hora)
def extraehora(HORA):
    if ('p. m.'  in HORA) or ('PM'  in HORA):
        if '12:' not in HORA:
            hora = int(HORA.split(':')[0]) + 12
        else:
            hora = int(HORA.split(':')[0])
    else:
        if '12:' not in HORA:
            hora = int(HORA[:2].split(':')[0])
        else:
            hora = 0
        
    return hora

### Devuelve el nombre del mes segun el numero de mes que se pase como 
### argumento a la funcion
def obtener_mes(x):
    
    if x == 1:
        return 'Enero'
    elif x== 2:
        return 'Febrero'
    elif x== 3:
        return 'Marzo'
    elif x== 4:
        return 'Abril'
    elif x== 5:
        return 'Mayo'
    elif x== 6:
        return 'Junio'
    elif x== 7:
        return 'Julio'
    elif x== 8:
        return 'Agosto'
    elif x== 9:
        return 'Septiembre'
    elif x== 10:
        return 'Octubre'
    elif x== 11:
        return 'Noviembre'
    elif x== 12:
        return 'Diciembre'
    
def precision_recall_auc_score(y_true, proba):
    
    precision, recall, _ = metrics.precision_recall_curve(y_true, proba)
    
    auc = metrics.auc(recall, precision)
    
    return auc

### Esta funcion realiza la conexion y ejecucion de un query en un archivo
### sqlite. Los parametros son la ruta o directorio del archivo y el query
### a ser ejecutado
def read_sqlite(db_path, query):
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query,conn)
    except Exception as e:
        logger.info(f'Error leyendo el archivo {db_path}: {e}')
        return None
    
    return df

### Esta funcion realiza la lectura de los accidentes "crudos" a ser considerados
### cada registro corresponde a un accidente, y tiene informacion de su 
### ubicacion, gravedad, barrio, fecha, hora, etc. Retorna la informacion de 
### los accidentes en un rango de fechas especifico    
def read_accidentes(d_ini, d_fin):
    
    db_name = 'database_pi2.sqlite3'
    db_path = f'data/{db_name}'
        
    query = f""" SELECT * 
                FROM raw_accidentes
                WHERE
                TW >= '{d_ini}' AND
                TW < '{d_fin}' """
    
    accidentes = read_sqlite(db_path, query)
    accidentes['TW'] = pd.to_datetime(accidentes['TW'])
    
    return accidentes

### Esta funcion realiza la lectura de la informacion climatica consultada
### en darksky, a esta informacion (la cual tiene informacion de para cada
### barrio y para cada hora) le agrega la etiqueta 1 o 0 refiriendose asi
### si se tuvo o no un accidente en ese barrio a esa hora. Los parametros de 
### la funcion nos indican el rango de fechas en el cual se va a leer la info
### ademas de filtrar o no la info que solo esta en aquellos barrios del Poblado 
def read_clima_accidentes(d_ini, d_fin, poblado = False):
    
    db_name = 'database_pi2.sqlite3'
    db_path = f'data/{db_name}'    
    
    
    if poblado:
        
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
                
        barrios = "','".join(poblado)
        
        query_clima = f""" SELECT *
                          FROM clima
                          WHERE
                          TW >= '{d_ini}' AND
                          TW < '{d_fin}' AND
                          BARRIO IN ('{barrios}')"""
    
        query_accidentes = f""" SELECT *
                              FROM accidentes
                              WHERE
                              TW >= '{d_ini}' AND
                              TW < '{d_fin}' AND 
                              BARRIO IN ('{barrios}') """
                              
    else:

        query_clima = f""" SELECT *
                          FROM clima
                          WHERE
                          TW >= '{d_ini}' AND
                          TW < '{d_fin}' """
    
        query_accidentes = f""" SELECT *
                              FROM accidentes
                              WHERE
                              TW >= '{d_ini}' AND
                              TW < '{d_fin}' """            

    clima = read_sqlite(db_path, query_clima)
    clima['TW'] = pd.to_datetime(clima['TW'])

    accidentes = read_sqlite(db_path, query_accidentes)
    accidentes['TW'] = pd.to_datetime(accidentes['TW'])    
    accidentes['Accidente'] = 1
    
    data_accidentes = clima.merge(accidentes[['TW','BARRIO','Accidente']], 
                                  how = 'left', on = ['TW','BARRIO'])
    
    data_accidentes['Accidente'] = data_accidentes['Accidente'].fillna(0)
    
    ### Mantenemos los registros que tengan informacion en la variable icon
    data_accidentes = data_accidentes[~data_accidentes['icon'].isna()].reset_index(drop = True)
    
    ### dentro del proceso de preparacion de datos, se determina que no se 
    ### utilizara la variable windbearing en el modelo, ya que esta falta
    ### en una cantidad considerable de registros
    data_accidentes = data_accidentes.drop(columns = ['windBearing'])
    
    return data_accidentes

### Esta funcion agrega a las variables climaticas informacion a cerca del
### promedio de dichas variables para cada barrio en las ultimas 5 horas, asi
### estamos incluyendo informacion sobre la distribucion de la info climatica
def organizar_data_infoClima(data):

    data['hora'] = data['TW'].dt.hour
    data['dia_sem'] = data['TW'].dt.dayofweek
    
    data= pd.get_dummies(data, columns=['hora'])
    data= pd.get_dummies(data, columns=['icon'])
    data= pd.get_dummies(data, columns=['dia_sem'])
    
    ### Festivos
    col_holidays = holidays.CountryHoliday('CO')
    es_hol = []
    for i in data['TW']:
        if i in col_holidays:
            es_hol.append(1)
        else:
            es_hol.append(0)
    data['festivo'] = es_hol
    
    ### Agrega mes y year
    
    data['Mes'] = data['TW'].dt.month
    data['Mes'] = data['Mes'].apply(lambda x: obtener_mes(x))
    
    data['Year'] = data['TW'].dt.year
    
    data= pd.get_dummies(data, columns=['Mes'])
    data= pd.get_dummies(data, columns=['Year'])
    
    
    
    
    
    ### Feature augmentation
    freq = '3H'
    variables = ['temperature','precipIntensity','apparentTemperature','dewPoint',
                 'humidity','windSpeed','cloudCover','visibility']
    
    data_aux = data.copy()
    data_aux.index = data_aux.TW
    data_aux = data_aux.sort_index()
    data_aux = data_aux.drop(columns = 'TW')
    
    data_pivot = data_aux.pivot_table(values=variables, index='TW',columns='BARRIO', aggfunc=sum)
    data_mean = data_pivot.rolling(freq, closed = 'left').mean().stack().reset_index(drop = False)
    
    col_means = [*data_mean.columns[:2]]
    for col in data_mean.columns[2:]:
        col_means.append(col + '_mean')
        
    data_mean.columns = col_means
    
    data = data.merge(data_mean, how = 'left', on = ['TW','BARRIO'])
    data = data.dropna().reset_index(drop = True)    
    
    ### Mean de precipitacion en las proximas 2 horas

    data_aux = data.copy()
    data_aux['TW'] = data_aux['TW'] - dt.timedelta(hours = 2)
    
    data_aux.index = data_aux.TW
    data_aux = data_aux.sort_index()
    data_aux = data_aux.drop(columns = 'TW')
    
    data_pivot = data_aux.pivot_table(values=variables, index='TW',columns='BARRIO', aggfunc=sum)
    data_precip = data_pivot.rolling('2H', closed = 'right').mean().stack().reset_index(drop = False)
    
    col_precip = [*data_precip.columns[:2]]
    for col in data_precip.columns[2:]:
        col_precip.append(col + '_mean_forward')
    
    data_precip.columns = col_precip
    
    data = data.merge(data_precip, how = 'left', on = ['TW','BARRIO'])
    data = data.dropna().reset_index(drop = True)
    
    return data

### Esta funcion organiza una variable extra que contiene informacion a cerca
### de la cantidad de accidentes que han ocurrido en las ultimas X horas, estas
### horas estan representadas en el parametro frecuencia como XH
def obtener_accidentes_acumulados(data, raw_accidentes, freq = '4D'):
    
    raw_accidentes['Accidente'] = 1
    spa_cum = raw_accidentes.pivot_table(values='Accidente', 
                                         index='TW',
                                         columns='BARRIO', 
                                         aggfunc=sum)
    spa_cum = spa_cum.resample('1D').agg('sum')
    
    Acc_sum = spa_cum.rolling(freq, closed = 'left').sum().stack().reset_index(drop = False)    
    Acc_sum.columns = ['TW_aux', 'BARRIO', f'cumAcc_{freq}']
    
    data['TW_aux'] =pd.to_datetime(data['TW'].apply(lambda x: x.strftime("%Y-%m-%d 00:00:00")))
    
    data = data.merge(Acc_sum, how = 'left', on =['TW_aux','BARRIO']).fillna(0)
    
    data = data.drop(columns = 'TW_aux')
    return data

#En la funcion grid, se considera un diccionario de modelos para ser entrenado,
#el conjunto de datos para entrenar, y devuelve el mejor modelo entrenado
#dado el score determinado.
def grid(base_path, now_date, path_file, os_X_tt, os_Y_tt,X_test,Y_test, models, 
         score = 'roc_auc', cv = 3, n_proc = 2, random = False, n_iter = 20):    
    
    for name in models:
       logger.info('*'*80)
       logger.info("Model: " + name)
       t_beg = time.time()
       
       ### Crea el Pipeline de forma tal que primero se realice la estandarizacion
       ### del conjunto de datos y posteriormente la evaluacion del modelo de ML
       pipeline = Pipeline([('scaler', StandardScaler()), (name,  models[name]['mod'])])
       parameters = {}          
       
       ### organiza los parametros para ser pasados a la busqueda Grid o Random
       for par in models[name]['par']:
             aux = name + '__' +  par
             parameters[aux] = models[name]['par'][par]
       
        ### Se realiza el entrenamiento ya sea utilizando la estrategia de
        ### Random Search o de Grid Searcj
       if random:
           
           aux = RandomizedSearchCV(pipeline, parameters, n_jobs = n_proc,\
                              scoring = ['roc_auc','balanced_accuracy','f1'], 
                              verbose=2, cv = cv,refit = score, n_iter = n_iter)
           
       else:
           
           aux = GridSearchCV(pipeline, parameters, n_jobs = n_proc,\
                              scoring = ['roc_auc','balanced_accuracy','f1'], 
                              verbose=2, cv = cv,refit = score)
           
       aux.fit(os_X_tt, os_Y_tt)
       
       ### Evalua el mejor modelo obtenido en el conjunto de validacion
       preds = aux.predict(X_test)
       proba = aux.predict_proba(X_test)
       
       
       models[name]['bestModel'] = aux.best_estimator_
       models[name]['mae'] = metrics.roc_auc_score(Y_test, proba[:,1])#aux.best_score_
       
       ### realizamos el calculo de las metricas en el conjunto de validacion
       bAccuracy = metrics.balanced_accuracy_score(Y_test,preds) 
       fScore = metrics.f1_score(Y_test,preds) 
       precision = metrics.precision_score(Y_test,preds)
       recall = metrics.recall_score(Y_test,preds)
       prec_rec_auc = precision_recall_auc_score(Y_test, proba[:,1])
       
       selection_time = time.time() - t_beg
       
       models[name]['selection_time'] = selection_time
       
       sample_f_path = os.path.join(base_path, path_file, f'{name}_{now_date.strftime("%Y%m%d_%H%M")}.sav')
       
       ### Guardamos el mejor modelo obtenido para cada una de las familias de
       ### modelos considerados
       logger.info(f"Saving model at {sample_f_path}")    
       joblib.dump(models[name]['bestModel'], sample_f_path)
       
       ### Escribe en el archivo de log las metricas obtenidas en el conjunto
       ### de validacion
       logger.info(f"El tiempo de seleccion fue: {selection_time:0.3f} s")
       logger.info(f"El {score} de la familia {name} es: {prec_rec_auc:0.3f}")
       logger.info(f"El precision-recall AUC de la familia {name} es: {models[name]['mae']:0.3f}")
       logger.info(f"El b_accuracy de la familia {name} es: {bAccuracy:0.3f}")
       logger.info(f"El fscore de la familia {name} es: {fScore:0.3f}")
       logger.info(f"La precision de la familia {name} es: {precision:0.3f}")
       logger.info(f"El recall de la familia {name} es: {recall:0.3f}")
       logger.info('*'*80)
   
    ### Obtiene el mejor modelo en base al ROC-AUC
    mod_name = None
    best_mae = -np.inf
    for name in models:
       if models[name]['mae'] > best_mae:
           mod_name = name
           best_mae = models[name]['mae']

    logger.info(f"best {score} was: {mod_name} with an error of: {best_mae}")
    
    return models, mod_name

### Las siguientes funciones, crean los graficos de la curva ROC y precision-recall
### tanto para el conjunto de entrenamiento como el conjunto de prueba. Estas
### funciones giardan los graficos en PDF
def graphs_evaluation(base_path, selected, ev_data, save = False):
    
    cols_plot = ['Accidente', 'Predicted']
    pl_data = ev_data[cols_plot]
    
    quant_fail = pl_data.groupby('Accidente').agg({'Accidente' : 'count'})
    quant_fail['ind']= quant_fail.index.values
    quant_fail['leg'] = quant_fail.apply(lambda x: str(int(x['ind'])) + ' : ' + str(x['Accidente']), axis = 1)
    quant_fail = quant_fail['leg'].values
    
    fig, ax = plt.subplots(1, 2)    
    plt.suptitle(f"Resultados con modelo {selected} - {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
    sns.set_style("whitegrid")
    sns.violinplot(x = 'Accidente', y = 'Predicted', data = pl_data, bw = 0.25, ax = ax[0],
                   palette = 'Pastel1', showmeans = True, showmedians=True,
                   showextrema=True)    
    ax[0].set_title("Violin plot")
    ax[0].set_ylim([-0.10, 1.05])
    ax[0].set_xlabel("Accidente efectivo")
    ax[0].set_ylabel("Probabilidad Accidente")
    ax[0].legend(quant_fail, fontsize=12)
    
    fpr, tpr, _ = metrics.roc_curve(pl_data['Accidente'], pl_data['Predicted'], drop_intermediate=False)
    roc_auc = metrics.auc(fpr, tpr)
            
    ax[1].plot(fpr, tpr, color='red', label=f'auc %0.5f' % roc_auc)    
    ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[1].legend(loc="lower right")
    
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')    
    ax[1].set_title('Receiver operating characteristic (ROC)')
    
    if save:
        logger.info("Saving figure testing data set evaluation")
        f_name = f"{dt.datetime.now().strftime('%Y%m%d_%H%M')}_trainOutput.pdf"
        fig_f_path = os.path.join(base_path, f_name)
        fig.savefig(fig_f_path, orientation='landscape')
        plt.close(fig)
    
    return None

def graphs_evaluation_test(base_path, ev_data, save = False):
    
    cols_plot = ['Accidente', 'Predicted']
    pl_data = ev_data[cols_plot]
    
    quant_fail = pl_data.groupby('Accidente').agg({'Accidente' : 'count'})
    quant_fail['ind']= quant_fail.index.values
    quant_fail['leg'] = quant_fail.apply(lambda x: str(int(x['ind'])) + ' : ' + str(x['Accidente']), axis = 1)
    quant_fail = quant_fail['leg'].values
    
    fig, ax = plt.subplots(1, 2)    
    plt.suptitle(f"Resultados Generado {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
    sns.set_style("whitegrid")
    sns.violinplot(x = 'Accidente', y = 'Predicted', data = pl_data, bw = 0.25, ax = ax[0],
                   palette = 'Pastel1', showmeans = True, showmedians=True,
                   showextrema=True)    
    ax[0].set_title("Violin plot")
    ax[0].set_ylim([-0.10, 1.05])
    ax[0].set_xlabel("Accidente efectivo")
    ax[0].set_ylabel("Probabilidad Accidente")
    ax[0].legend(quant_fail, fontsize=12)
    
    fpr, tpr, _ = metrics.roc_curve(pl_data['Accidente'], pl_data['Predicted'], drop_intermediate=False)
    roc_auc = metrics.auc(fpr, tpr)
            
    ax[1].plot(fpr, tpr, color='red', label=f'auc %0.5f' % roc_auc)    
    ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[1].legend(loc="lower right")
    
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')    
    ax[1].set_title('Receiver operating characteristic (ROC)')
    
    if save:
        logger.info("Saving figure testing data set evaluation")
        f_name = f"{dt.datetime.now().strftime('%Y%m%d_%H%M')}_testOutput.pdf"
        fig_f_path = os.path.join(base_path, f_name)
        fig.savefig(fig_f_path, orientation='landscape')
        plt.close(fig)
    
    return None

def precision_recall_graph(base_path, selected, ev_data, save = False):
    
    cols_plot = ['Accidente', 'Predicted']
    pl_data = ev_data[cols_plot]
    
    quant_fail = pl_data.groupby('Accidente').agg({'Accidente' : 'count'})
    quant_fail['ind']= quant_fail.index.values
    quant_fail['leg'] = quant_fail.apply(lambda x: str(int(x['ind'])) + ' : ' + str(x['Accidente']), axis = 1)
    quant_fail = quant_fail['leg'].values
    
    fig, ax = plt.subplots(1, 2)    
    plt.suptitle(f"Resultados con modelo {selected} - {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
    sns.set_style("whitegrid")
    sns.violinplot(x = 'Accidente', y = 'Predicted', data = pl_data, bw = 0.25, ax = ax[0],
                   palette = 'Pastel1', showmeans = True, showmedians=True,
                   showextrema=True)    
    ax[0].set_title("Violin plot")
    ax[0].set_ylim([-0.10, 1.05])
    ax[0].set_xlabel("Accidente efectivo")
    ax[0].set_ylabel("Probabilidad Accidente")
    ax[0].legend(quant_fail, fontsize=12)
    
    precision, recall, _ = metrics.precision_recall_curve(pl_data['Accidente'], pl_data['Predicted'])
    
    fscore = metrics.f1_score(pl_data['Accidente'], (pl_data['Predicted'] >= 0.5).astype(int) )
    
    random = pl_data['Accidente'].sum()/len(pl_data['Accidente'])
    ax[1].plot(recall, precision, color='red', label=f'F1score %0.5f' % fscore)    
    ax[1].plot([0, 1], [random, random], color='navy', linestyle='--')
    ax[1].legend(loc="upper right")
    
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')    
    ax[1].set_title('Precision-Recall Curve')
    
    if save:
        logger.info("Saving figure testing data set evaluation")
        f_name = f"PrecRec_{dt.datetime.now().strftime('%Y%m%d_%H%M')}_trainOutput.pdf"
        fig_f_path = os.path.join(base_path, f_name)
        fig.savefig(fig_f_path, orientation='landscape')
        plt.close(fig)
    
    return None

def precision_recall_graph_test(base_path, ev_data, save = False):
    
    cols_plot = ['Accidente', 'Predicted']
    pl_data = ev_data[cols_plot]
    
    quant_fail = pl_data.groupby('Accidente').agg({'Accidente' : 'count'})
    quant_fail['ind']= quant_fail.index.values
    quant_fail['leg'] = quant_fail.apply(lambda x: str(int(x['ind'])) + ' : ' + str(x['Accidente']), axis = 1)
    quant_fail = quant_fail['leg'].values
    
    fig, ax = plt.subplots(1, 2)    
    plt.suptitle(f"Resultados Generado - {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
    sns.set_style("whitegrid")
    sns.violinplot(x = 'Accidente', y = 'Predicted', data = pl_data, bw = 0.25, ax = ax[0],
                   palette = 'Pastel1', showmeans = True, showmedians=True,
                   showextrema=True)    
    ax[0].set_title("Violin plot")
    ax[0].set_ylim([-0.10, 1.05])
    ax[0].set_xlabel("Accidente efectivo")
    ax[0].set_ylabel("Probabilidad Accidente")
    ax[0].legend(quant_fail, fontsize=12)
    
    precision, recall, _ = metrics.precision_recall_curve(pl_data['Accidente'], pl_data['Predicted'])
    
    fscore = metrics.f1_score(pl_data['Accidente'], (pl_data['Predicted'] >= 0.5).astype(int) )
    
    random = pl_data['Accidente'].sum()/len(pl_data['Accidente'])
    ax[1].plot(recall, precision, color='red', label=f'F1score %0.5f' % fscore)    
    ax[1].plot([0, 1], [random, random], color='navy', linestyle='--')
    ax[1].legend(loc="upper right")
    
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')    
    ax[1].set_title('Precision-Recall Curve')
    
    if save:
        logger.info("Saving figure testing data set evaluation")
        f_name = f"PrecRec_{dt.datetime.now().strftime('%Y%m%d_%H%M')}_testOutput.pdf"
        fig_f_path = os.path.join(base_path, f_name)
        fig.savefig(fig_f_path, orientation='landscape')
        plt.close(fig)
    
    return None

### Las siguientes dos funciones crean la matriz de confusion tanto para el
### conjunto de validacion como de prueba, ademas de guardar las matrices de
### confusion en un archivo PDF
def matrix_confusion(base_path, selected, ev_data, bound,save = False):
    
    cols_plot = ['Accidente', 'Predicted']
    pl_data = ev_data[cols_plot]
    pl_data['Predicted'] = (pl_data['Predicted'] > bound).astype(int) 
    lon_g = len(pl_data['Accidente'])
    
    mat = (confusion_matrix(pl_data['Accidente'].values, pl_data['Predicted']).ravel()/lon_g) * 100
    
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(np.array(mat).reshape((2,2)), cbar=False,annot=True, xticklabels=['Pred 0', 'Pred 1'], annot_kws={'fontsize' : 15}, yticklabels=['Real 0', 'Real 1'])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
#    return mat
    if save:
        logger.info("Saving figure confusion matrix test1")
        f_name = f"{str(bound)}_CM_training.pdf"
        fig_f_path = os.path.join(base_path, f_name)
        fig.savefig(fig_f_path, orientation='landscape')
        plt.close(fig)
    
    return None

def matrix_confusion_test(base_path, ev_data, bound,save = False):
    
    cols_plot = ['Accidente', 'Predicted']
    pl_data = ev_data[cols_plot]
    pl_data['Predicted'] = (pl_data['Predicted'] > bound).astype(int)    
    lon_g = len(pl_data['Accidente'])
    
    mat = (confusion_matrix(pl_data['Accidente'].values, pl_data['Predicted']).ravel()/lon_g) * 100
    
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(np.array(mat).reshape((2,2)), cbar=False,annot=True, xticklabels=['Pred 0', 'Pred 1'], annot_kws={'fontsize' : 15}, yticklabels=['Real 0', 'Real 1'])
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
#    return mat
    if save:
        logger.info("Saving figure confusion matrix test1")
        f_name = f"{str(bound)}_CM_test.pdf"
        fig_f_path = os.path.join(base_path, f_name)
        fig.savefig(fig_f_path, orientation='landscape')
        plt.close(fig)
    
    return None

### Esta funcion carga tanto el archivo guardado que incluye tanto el objeto
### de clases modelo como el modelo entrenado
def carga_model(base_path, path_savs, sav_name):
    ext = 'sav'
    f_name = f"{sav_name}.{ext}"
    
    f_path = os.path.join(base_path, path_savs, f_name)
    
    samples = {'path' : f_path}
    
    try:
        if not os.path.isfile(f_path):
            logger.error(f"File {f_path} does not exist")            
            raise Exception('non-existing file')
            
        loaded_model = joblib.load(f_path)
        samples['model'] = loaded_model
            
    except Exception as e:
        logger.error(f"Falla en base de datos {f_path}: " + str(e))    
       
    return samples

### Rsta funcion carga en memoria un modelo .sav entrenado, es solo el modelo
### no incluye el objeto de clases modelo
def carga_model_ind(base_path, path_savs, sav_name):
    ext = 'sav'
    f_name = f"{sav_name}.{ext}"
    
    f_path = os.path.join(base_path, path_savs, f_name)
    
    try:
        if not os.path.isfile(f_path):
            logger.error(f"File {f_path} does not exist")            
            raise Exception('non-existing file')
            
        loaded_model = joblib.load(f_path)
        
            
    except Exception as e:
        logger.error(f"Falla en base de datos {f_path}: " + str(e))    
       
    return loaded_model
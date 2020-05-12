# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:23:23 2020

@author: nicol
"""

import os
### Realizamos el cambio de directoroi de trabajo al "Directorio Base"
current_dir = os.getcwd()
base_path = os.path.dirname(current_dir)
base_path = os.path.dirname(base_path)

os.chdir(base_path)

import numpy as np
import random
import pandas as pd
import datetime as dt
from random import sample 
import os
import sys

random.seed(42)



### Lee datos de train (para hallar barrios con mas accidentes en train)
data_train_u = pd.read_csv('data/train_z.csv') 
data_val = pd.read_csv('data/validation_z.csv') 
data_train = pd.concat([data_train_u, data_val])


### Lee datos de test
data_test_completa = pd.read_csv('data/test_z.csv')
data_test = data_test_completa.drop(['BARRIO', 'Accidente', 'TW'], axis=1)



### Modelo usado
sys.path.insert(0, os.getcwd())
import scripts.funciones as funciones
version = 'verFinal'    
mod_version = funciones.carga_model('.', f'models/{version}', version)
mod = mod_version['model'].steps[0][1]

classifier = mod_version['model'].steps[1][1][1]





### Calcular predicciones en test
predicciones = classifier.predict(data_test)
probabilidades = classifier.predict_proba(data_test)[:,1]



#################### Cargar predicciones lme
#dataf_de_r = pd.read_excel(r"C:\Users\nicol\Documents\Mis_documentos\MaestriaCienciaDatos\Semestre2\PI_2\Lme\Prediccion_test_r.xlsx")
#predicciones2 = [] 
#probabilidades = dataf_de_r['Prediccion'].values
#for i in probabilidades2:
#    if i>0.5:
#        predicciones2.append(1)
#    else:
#        predicciones2.append(0)




###### A partir de aqui, comparar entre asignar agentes con el modelo 
#### o de otras formas (aleatoria)

#### Consolidar data_test
data_test_full = data_test_completa.copy()
data_test_full['predic_modelo'] = probabilidades

data_test_full['TW'] = pd.to_datetime(data_test_full['TW'])
data_test_full['hour'] = data_test_full['TW'].dt.hour

#### Ordenar por fecha
data_test_full = data_test_full.sort_values(by=['TW','BARRIO']).reset_index(drop=True)
data_test_full = data_test_full[['TW','Accidente', 'predic_modelo','BARRIO']]

#### Numero de turnos totales diarios que se cubren
### Se asume que cada agente cubre x turnos diarios
num_agentes = 5
turnos_diarios_agente = 4
agentes = num_agentes * turnos_diarios_agente

#### Barrios EL Poblado
numb = len(pd.unique(data_test_full['BARRIO']))


### Los 'agentes' barrios con mas choques en train
accis = data_train.groupby('BARRIO').sum()['Accidente']
most_accis = np.argsort(accis.values)[::-1]
barrs = accis.index[most_accis[:1]]
most_a = []
for i in range(int(len(data_test_full)/numb)): #### Todas las horas, el mismo barrio
    most_a.append(most_accis[0]+i*numb)

### Accidentes reales
acci_reales = data_test_full['Accidente']






######### Ciclo para empezar a comparar
numb = numb*24 #dias en vez de horas
dias_tota = len(data_test_full)/numb  
aci1 = []
aci1b = []
aci2 = []
aci3 = []
dic_toto = {}
for i in range(24):
    dic_toto[i] = 1
for hor in range(int(dias_tota)):

    ### Comparar matches con los 3 casos
    
    ### Caso 1: prediccion modelo
    #### Las probabilidades de accidentalidad en esta franja
    acci_prob = data_test_full['predic_modelo'][hor*numb:(hor+1)*numb].values
    acci_p = np.argsort(acci_prob)[::-1][:agentes]
    predigo1 = np.zeros(numb)
    predigo1[acci_p]=1
    aci1.extend(list(predigo1))

    ## Caso 1b: prediccion modelo (restringido max por hora)
    ### Las probabilidades de accidentalidad en esta franja
    max_por_hora = num_agentes
    acci_pre = data_test_full[hor*numb:(hor+1)*numb].reset_index(drop=True).sort_values(by='predic_modelo', ascending=False)
    new_accip = []
    dic_cumul = {}
    for di in np.argsort(acci_prob)[::-1]:
        esta_o = acci_pre.loc[di]
        if esta_o[0].hour in [8,9,10,11,12,13,14,15,16,17,18,19,20]:  ### solo prediga entre esas horas
            if esta_o[0].hour not in dic_cumul:
                new_accip.append(di)
                dic_cumul[esta_o[0].hour] = 1
                dic_toto[esta_o[0].hour] = dic_toto[esta_o[0].hour] + 1
            else:
                if dic_cumul[esta_o[0].hour]<max_por_hora:
                    new_accip.append(di)
                    dic_cumul[esta_o[0].hour] = dic_cumul[esta_o[0].hour] + 1
                    dic_toto[esta_o[0].hour] = dic_toto[esta_o[0].hour] + 1
        if len(new_accip)==agentes:
            break
    predigo1b = np.zeros(numb)
    predigo1b[new_accip]=1
    aci1b.extend(list(predigo1b))
    
    
    ### Caso 2: aleatorio
    predigo2 = np.zeros(numb)
    accialeat = sample(range(numb),agentes)    
    predigo2[accialeat]=1
    aci2.extend(list(predigo2))



### Caso 3: Barrios con mas accidentes, horas mañana
numbar = len(pd.unique(data_test_full['BARRIO']))
horas_tota = len(data_test_full)/numbar
horas_pico = [8,9,10,11]
aci3 = []
for hor in range(int(horas_tota)):
    predigo3 = np.zeros(numbar)
    if hor%24 in horas_pico:
        predigo3[most_accis[:num_agentes]]=1
    aci3.extend(list(predigo3))
    




### Caso 4: Barrios con mas accidentes, horas tarde
numbar = len(pd.unique(data_test_full['BARRIO']))
horas_tota = len(data_test_full)/numbar
horas_pico = [14,15,16,17]
aci4 = []
for hor in range(int(horas_tota)):
    predigo4 = np.zeros(numbar)
    if hor%24 in horas_pico:
        predigo4[most_accis[:num_agentes]]=1
    aci4.extend(list(predigo4))


#    ### Caso 5: Barrios con mas accidentes, horas pico
#    numbar = len(pd.unique(data_test_full['BARRIO']))
#    horas_tota = len(data_test_full)/numbar
#    horas_pico = [10,11,12,13,14,16,17,18]
#    aci4 = []
#    for hor in range(int(horas_tota)):
#        predigo4 = np.zeros(numbar)
#        if hor%24 in horas_pico:
#            predigo4[most_accis[:num_agentes]]=1
#        aci4.extend(list(predigo4))
#        
#        
#    
#    ### Caso 6: Barrios con mas accidentes, horas pico
#    numbar = len(pd.unique(data_test_full['BARRIO']))
#    horas_tota = len(data_test_full)/numbar
#    horas_pico = [10,11,12,13,14,16,17,18]
#    aci4 = []
#    for hor in range(int(horas_tota)):
#        predigo4 = np.zeros(numbar)
#        if hor%24 in horas_pico:
#            predigo4[most_accis[:num_agentes]]=1
#        aci4.extend(list(predigo4))
    


### Modelo (sin maximo por hora)
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
sensibilidad = recall_score(acci_reales.values, aci1)
precision =precision_score(acci_reales.values, aci1)
b_accuracy = accuracy_score(acci_reales.values, aci1)
print('\nAsignacion modelo (sin max por hora): ')
print(f'La sensibilidad es {sensibilidad}')
print(f'La precision es {precision}')
print(f'El  accuracy es {b_accuracy}')
print(f'Se cubrieron {round(precision*agentes*dias_tota)} accidentes.')


### Modelo (maximo por hora)
sensibilidad = recall_score(acci_reales.values, aci1b)
precision =precision_score(acci_reales.values, aci1b)
b_accuracy = accuracy_score(acci_reales.values, aci1b)
print('\nAsignacion modelo (con max por hora): ')
print(f'La sensibilidad es {sensibilidad}')
print(f'La precision es {precision}')
print(f'El  accuracy es {b_accuracy}')
print(f'Se cubrieron {round(precision*agentes*dias_tota)} accidentes.')


### Aleatorio
sensibilidad = recall_score(acci_reales.values, aci2)
precision =precision_score(acci_reales.values, aci2)
b_accuracy = accuracy_score(acci_reales.values, aci2)
print('\nAsignacion aleatoria: ')
print(f'La sensibilidad es {sensibilidad}')
print(f'La precision es {precision}')
print(f'El  accuracy es {b_accuracy}')
print(f'Se cubrieron {round(precision*agentes*dias_tota)} accidentes.')



### Barrio con mas accidentes
sensibilidad = recall_score(acci_reales.values, aci3)
precision =precision_score(acci_reales.values, aci3)
b_accuracy = accuracy_score(acci_reales.values, aci3)
print('\nBarrio con mas accidentes (horas mañana): ')
print(f'La sensibilidad es {sensibilidad}')
print(f'La precision es {precision}')
print(f'El  accuracy es {b_accuracy}')
print(f'Se cubrieron {round(precision*agentes*dias_tota)} accidentes.')

### Horas pico, Barrios con mas accidentes
sensibilidad = recall_score(acci_reales.values, aci4)
precision =precision_score(acci_reales.values, aci4)
b_accuracy = accuracy_score(acci_reales.values, aci4)
print('\nBarrios con mas accidentes (horas tarde): ')
print(f'La sensibilidad es {sensibilidad}')
print(f'La precision es {precision}')
print(f'El  accuracy es {b_accuracy}')
print(f'Se cubrieron {round(precision*agentes*dias_tota)} accidentes.')


print('\n\nAccidentes totales: ' + str(round((precision*agentes*dias_tota)/sensibilidad)))
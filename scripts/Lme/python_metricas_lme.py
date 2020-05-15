#### Script para calcular metricas de modelo LME

import pandas as pd


### Leer datos de validation
li = pd.read_csv("predicciones_val_r.txt", sep=';')


### Consolidar prediccion con probabilidades
li_clase = []
for i in li['predictio_v']:
    if i>0.5:
        li_clase.append(1)
    else:
        li_clase.append(0)



### Realizamos el cambio de directorio de trabajo al "Directorio Base"
import os
import sys
current_dir = os.getcwd()
base_path = os.path.dirname(current_dir)
base_path = os.path.dirname(base_path)
sys.path.insert(0, os.getcwd())
os.chdir(base_path)


### Calcular metricas
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
import scripts.funciones as funciones
sensibilidad = recall_score(li['val_dat.Accidente'], li_clase)
precision =precision_score(li['val_dat.Accidente'], li_clase)
fscore = f1_score(li['val_dat.Accidente'], li_clase)
rocauc = roc_auc_score(li['val_dat.Accidente'], li['predictio_v'])
prauc = funciones.precision_recall_auc_score(li['val_dat.Accidente'], li['predictio_v'])
b_accuracy = balanced_accuracy_score(li['val_dat.Accidente'], li_clase)
print('\nMetricas de LME en conjunto de validation: ')
print('\nSensibilidad: '+str(sensibilidad))
print('Precision: '+str(precision))
print('F1 score: '+str(fscore))
print('ROC AUC: '+str(rocauc))
print('PR AUC: '+str(prauc))
print('Balanced accuracy: '+str(b_accuracy))
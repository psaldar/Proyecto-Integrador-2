#### Script para calcular metricas de modelo LME

import pandas as pd


### Leer datos de validation
li = pd.read_csv("validation_predict_lme.csv", sep=';')



### Consolidar prediccion con probabilidades
li_clase = []
for i in li['Prediccion_R']:
    if i>0.5:
        li_clase.append(1)
    else:
        li_clase.append(0)



### Calcular metricas
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
sensibilidad = recall_score(li['Accidente'], li_clase)
precision =precision_score(li['Accidente'], li_clase)
fscore = f1_score(li['Accidente'], li_clase)
rocauc = roc_auc_score(li['Accidente'], li['Prediccion_R'])
b_accuracy = balanced_accuracy_score(li['Accidente'], li_clase)
print('\nMetricas de LME en conjunto de validation: ')
print('\nSensibilidad: '+str(sensibilidad))
print('Precision: '+str(precision))
print('F1 score: '+str(fscore))
print('ROC AUC: '+str(rocauc))
print('Balanced accuracy: '+str(b_accuracy))
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:05:55 2020

@author: Pablo Saldarriaga
"""

### Asignar hora del dia para cada accidente (solo el entero de la hora)
def extraehora(HORA):
    if 'PM' in HORA:
        if '12:' not in HORA:
            hora = int(HORA[:2]) + 12
        else:
            hora = int(HORA[:2])
    else:
        if '12:' not in HORA:
            hora = int(HORA[:2])
        else:
            hora = 0
        
    return hora

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
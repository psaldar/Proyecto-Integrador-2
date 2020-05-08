# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:25:00 2020

@author: nicol
"""

##### Este archivo contiene todos los scripts utilitarios usados 
##### para el algoritmo genetico

import random
import matplotlib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

random.seed(42)
np.random.seed(42)


### Elegir solo estas features
def reduce_features(solution, features):
    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features



### Accuracy
def classification_accuracy(labels, predictions):
    #metric = balanced_accuracy_score(labels, predictions)
    metric = roc_auc_score(labels, predictions)
    return metric


### Calcular evaluaciones
def cal_pop_fitness(pop, features, labels, train_indices, test_indices, classifier, ya_vistas):
    accuracies = np.zeros(pop.shape[0])
    idx = 0

    for curr_solution in pop:
        reduced_features = reduce_features(curr_solution, features)
        train_data = reduced_features[train_indices, :]
        test_data = reduced_features[test_indices, :]

        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        
        
        ### Si ya se habia visto antes esta solucion
        if str(curr_solution) in ya_vistas:
            accuracies[idx] = ya_vistas[str(curr_solution)]
        
        ### de lo contrario, estimar valor de funcion objetivo
        else:
            classifier.fit(X=train_data, y=train_labels)
    
            predictions = classifier.predict(test_data)
            probabilities = classifier.predict_proba(test_data)
            accuracies[idx] = classification_accuracy(test_labels, probabilities[:,1])
            
            ya_vistas[str(curr_solution)] = accuracies[idx]
        
            
        idx = idx + 1
    return accuracies, ya_vistas


#### Seleccionar los padres, por torneo entre 3
def select_mating_pool(pop, fitness, num_parents):
    # Seleccionar los mejores padres, usando torneo entre 3    
    parents = np.empty((num_parents, pop.shape[1]))
    
    ### Padres no elegidos
    no_eleg = list(range(len(fitness)))
    

    ### Dejar siempre el mejor padre
    for parent_num in range(1):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
        
        ### Quitar el padre de los permitidos
        no_eleg.remove(max_fitness_idx)        
        
    
    ### Elegir el resto padres por torneo
    for parent_num in range(1,num_parents):
        parents_torneo = random.sample(no_eleg, 3)
        
        ### elegir el mejor de estos
        mejor_padre = None
        mejor_val = 0

        for pa in parents_torneo:
            if fitness[pa]>mejor_val:
                mejor_val = fitness[pa]
                mejor_padre = pa
        
        parents[parent_num, :] = pop[mejor_padre, :]
        fitness[mejor_padre] = -99999999999     
        
        ### Quitar el padre de los permitidos
        no_eleg.remove(mejor_padre)
        

     

    return parents


### Hacer el cruce
### Por ahora, es un simple one point crossover
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    
    # Punto donde se hace el cruce.
    # Elegir aleatorio el punto de corte, pero al menos 10 genes de cada padre
    crossover_point = np.random.randint(10, int(offspring_size[1])-10)

    for k in range(offspring_size[0]):
        # Indice del primer padre
        parent1_idx = k%parents.shape[0]
        # Indice del segundo padre
        parent2_idx = (k+1)%parents.shape[0]
        # La primera mitad de los genes es del primer padre
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # La segunda mitad de los genes es del segundo padre
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


#### Esto varia genes del hijo, de forma aleatoria (cambia un 1 por un 0)
def mutation(offspring_crossover, num_mutations, num_genes):
    for i in range(num_genes):
        mutation_idx = np.random.randint(low=0, high=offspring_crossover.shape[1], size=num_mutations)
        # La mutacion cambia genes de forma aleatoria (1 por 0 o viceversa)
        for idx in range(offspring_crossover.shape[0]):
            offspring_crossover[idx, mutation_idx] = 1 - offspring_crossover[idx, mutation_idx]
    return offspring_crossover
	
	
def genetic_algorithm(X,num_generations, sol_per_pop, porc_mutation, porc_genes,
                      num_feature_elements,data_inputs, data_outputs, 
                      train_indices, test_indices, classifier, num_mutations,num_genes):   
    
    ### Fijar semillas
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Defining the population shape.
    pop_shape = (sol_per_pop, num_feature_elements)

    # Crear poblacion inicial
    new_population = np.random.randint(low=0, high=2, size=pop_shape)
    #print(new_population.shape)

    num_parents_mating = int(sol_per_pop/2) # Cuantos padres se cruzaran 




    ### Le agrego las mejores variables que habiamos obtenido usando
    ### feature importances en la tarea 3, para que tenga una buena solucion
    ### al menos en el conjunto de las iniciales
    cols = ['apparentTemperature', 'temperature', 'humidity_mean',
           'temperature_mean', 'apparentTemperature_mean', 'humidity',
           'dewPoint_mean', 'dewPoint', 'windSpeed_mean', 'windSpeed',
           'cloudCover_mean', 'uvIndex', 'icon_partly-cloudy-night',
           'poblado_LaAguacatala', 'visibility_mean', 'cloudCover', 'dia_sem_6',
           'precipIntensity_mean', 'visibility', 'poblado_ElCastillo',
           'icon_partly-cloudy-day', 'poblado_VillaCarlota', 'poblado_Astorga',
           'precipIntensity', 'poblado_AltosdelPoblado', 'precipProbability',
           'hora_7', 'poblado_LosBalsosNo1', 'poblado_ElDiamanteNo2',
           'poblado_Manila', 'poblado_SantaMariadeLosÁngeles', 'poblado_Lalinde',
           'hora_3', 'hora_0', 'hora_1', 'dia_sem_4', 'dia_sem_5', 'hora_2',
           'poblado_ElPoblado', 'poblado_SanLucas', 'poblado_LasLomasNo2',
           'dia_sem_1', 'dia_sem_2', 'poblado_BarrioColombia',
           'poblado_LosBalsosNo2', 'hora_19', 'hora_4', 'dia_sem_3', 'dia_sem_0',
           'hora_17', 'hora_6', 'icon_cloudy']

    sol_original = []
    for i in X.columns:
        if i in cols:
            sol_original.append(1)
        else:
            sol_original.append(0)

    ### Reemplazo una solucion con esta
    new_population[0] = np.array(sol_original)


    ### Para ahorrar tiempos de computo, guardar las soluciones ya vistas
    ya_vistas = {}

    ####### Iterar por generaciones
    best_outputs = []
    for generation in range(num_generations):
        #print("Generation : ", generation)
        # Medir la funcion objetivo de la poblacion
        fitness, ya_vistas = cal_pop_fitness(new_population, data_inputs, data_outputs, train_indices, test_indices, classifier, ya_vistas)
        best_outputs.append(np.max(fitness))


        ### Ver como va variando
        #print(np.sort(fitness)[::-1][:5])

        # Mejor resultado actualmente
        #print("Best result : ", best_outputs[-1])

        # Seleccion: simplemenete se eligen los mejores num_parents_mating para ser padres
        parents = select_mating_pool(new_population, fitness, num_parents_mating)

        # Se hace el cruce: aqui se usa cruce por un punto
        # El hijo tendra un bloque de genes del padre 1, y otro bloque del padre 2
        offspring_crossover = crossover(parents, offspring_size=(pop_shape[0]-parents.shape[0], num_feature_elements))

        # Se varian aleatoriamente algunos genes de algunos hijos
        offspring_mutation = mutation(offspring_crossover, num_mutations=num_mutations, num_genes = num_genes)

        # Combino a los mejores padres y sus hijos, para tener la poblacion de la proxima generacion
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        
    #### Plotear los resultados obtenidos
    fitness, ya_vistas = cal_pop_fitness(new_population, data_inputs, data_outputs, train_indices, test_indices, classifier, ya_vistas)
    best_match_idx = np.where(fitness == np.max(fitness))[0]
    best_match_idx = best_match_idx[0]

    best_solution = new_population[best_match_idx, :]
    best_solution_indices = np.where(best_solution == 1)[0]
    best_solution_num_elements = best_solution_indices.shape[0]
    best_solution_fitness = fitness[best_match_idx]

    print("Mejor solucion : ", best_solution)
    print("Indices seleccionados : ", best_solution_indices)
    print("Numero de variables seleccionadas : ", best_solution_num_elements)
    print("ROC-AUC de la mejor solucion de todo el algoritmo : ", best_solution_fitness)

    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(best_outputs)
    matplotlib.pyplot.xlabel("Generacion")
    matplotlib.pyplot.ylabel("Mejor ROC-AUC")
    matplotlib.pyplot.show()
    
    return best_solution, best_solution_indices, best_solution_num_elements, best_solution_fitness
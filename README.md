Predicción de Accidentes Vehiculares en el Poblado
=======================================================================================


## Descripción

Este repositorio contiene la información requerida para reproducir el trabajo de "Predicción de Accidentes Vehiculares en el Poblado"

### "Predicción de Accidentes Vehiculares en el Poblado"

Pablo A. Saldarriaga<sup>1</sup>, Nicolás Prieto<sup>1</sup>

<sup>1</sup> Maestría en Ciencia de Datos, Universidad EAFIT, Medellín, Colombia






# Flujo para la ejecución del código

Para replicar el trabajo que se realizó, explicamos a continuación el flujo/orden en el cual se deben de ejecutar los códigos de este repositorio.

### Preparación de Datos

Inicialmente, descargamos la información de accidentalidad de Medellín de la página Open Data Medellín, consideramos entonces la información en los años 2017, 2018 y 2019.

- Accidentes 2017: https://geomedellin-m-medellin.opendata.arcgis.com/datasets/accidentalidad-georreferenciada-2017
- Accidentes 2018: https://geomedellin-m-medellin.opendata.arcgis.com/datasets/accidentalidad-georreferenciada-2018
- Accidentes 2019: https://geomedellin-m-medellin.opendata.arcgis.com/datasets/accidentalidad-georreferenciada-2019

Estos 3 archivos descargados se guardarían en la carpeta "data".

Cada uno de los archivos descargados serán la entrada para el archivo scripts/auxiliares/proc_raw_accidentes.py. Este script toma cada uno de los archivos descargados y realiza una limpieza en los nombres de los barrios y la asignación de la "Ventana de Tiempo (TW)", la cual indica a qué franja horaria pertenece cada accidente. Además, elimina aquellos registros que se consideran "raros" por los nombres de los barrios (ya que se encontraron inconsistencias en el cambio del nombre del barrio). Se menciona también que el script deja solo máximo un accidente por cada franja de fecha, hora y barrio, ya que en nuestro caso modelamos solamento si ocurrió o no ocurrió al menos un accidente en dicha franja. Adicionalmente, este script crea un archivo csv con los centroides de cada uno de los barrios (Lat. y Lon), ya que en estos puntos es donde se consultará la información climática del barrio. Finalmente, el script guarda en una base de datos SQLite la información recopilada antes y despues de obtener registros únicos de los accidentes.

Una vez se ha organizado la información de los accidentes, procedemos a obtener la información climática para cada uno de los barrios utilizando el archivo scripts/auxiliares/organizar_info_climatica.py. Este script lee el archivo csv que contiene la información de cada barrio con su respectivo centroide generado con el código anterior. Por lo que, para una franja de tiempo en específico, este archivo realizará para cada fecha y hora y para cada barrio, la consulta de las variables climáticas al API de Dark Sky (tener en cuenta que para poder realizar la consulta a Dark Sky, se debe realizar la suscripción al API para obtener una llave de acceso. La llave de acceso generada en este proyecto no se incluye, ya que el uso de esta API puede generar costos asociados a la persona registrada). A medida que va realizando la consulta al API, guarda la información climática en la base de datos SQLite creada en el script anterior en una tabla nueva.

De esta forma, en el SQLite 'database_pi2.sqlite3' se encuentra toda la información del proyecto. Esta base de datos puede ser descargada del siguiente link: https://drive.google.com/open?id=1DHZ4r8gIvHAqHSjvSs3khUOA2vckeeGu (este es un link al archivo almacenado en una carpeta de Google Drive de uno de los autores del trabajo). Este se debe guardar en la carpeta "data".


### Para tener en cuenta:

Ya que se acaba de almacenar la información en una base de datos SQLite, y teniendo en cuenta que a lo largo del proyecto se estarán utilizando diversas funciones en varios códigos, es importante mencionar que en el archivo scripts/funciones.py se encuentran todas las funciones y métodos relevantes del proyecto. Entre esas, se incluye la función que lee directamente la información tanto de los accidentes "crudos" (tal cual vienen de open data Medellín, pero con cierta limpieza en sus variables), además de la información de las condiciones climáticas para cada fecha y hora y para cada barrio incluyendo si el registro representa un accidente o no. Otras funciones que se encontrarán acá son el entrenamiento de un modelo en base a un Grid o Random Search, la generación de gráficas de evaluación de modelos, además de funciones para cargar los modelos entrenados y almacenados en archivos con extensión .sav. Se menciona que estos modelos guardados incluyen por dentro no solo el modelo sino también la herramienta para estandarizar los datos a manera de un pipeline.


## Modelamiento

Nota: Se aclara que en los scripts de esta sección, se parten los datos en conjuntos de entrenamiento, validación y test para las distintas etapas. El período entre Agosto 1 y Dic 30 de 2019 es período de test, mientras que el período entre junio de 2017 y julio de 2019 contiene entrenamiento y validación (80% de entrenmamiento y 20% de validación). Se aclara que cuando se explique el resampling, se remuestrea solamente el conjunto de entrenamiento.

- Teniendo en cuenta que el problema que estamos estudiando presenta un gran desbalance de clases, lo que hacemos es la evaluación de diferentes métodos de remuestreo (tanto undersampling como oversampling, además de considerar versiones híbridas), adicional a las proporciones deseadas entre los elementos de las clases 1 y 0. Para ello ejecutamos el archivo scripts/Resampling/PruebasResampling.py. Este archivo creará una carpeta donde se podrán encontrar un archivo de logs llamado "resampling.log", a medida que este script va evaluando tanto los diferentes métodos de resampling como las diferentes proporciones, escribirá en el archivo log los resultados obtenidos. Es importante mencionar que acá se está utilizando una evaluación del mejor método de resampling con una técnica de envoltura, ya que realizaremos la selección del método en base al desempeño de un modelo entrenado. Con este script se determina entonces la técnica y la proporción de resampling que mejores resultados provea en el conjunto de validación. En los siguientes scripts, nosotros los adaptamos de tal forma que usen la técnica y proporción de resampling que ganó cuando nosotros hicimos la ejecución (si se desea cambiar dicho resampling, tendría que cambiarse en los demás scripts que le siguen a este).

- Luego procedemos a realizar la selección de variables que incluiremos en nuestro modelo de predicción. Para esto se utiliza el Notebook relacionado al archivo notebooks/Feature Selection.ipynb. Este Notebook realiza la carga de todos los datos y aplica primero una eliminación de variables utilizando criterios basados en la correlación y correlación parcial, luego realiza la selección de variables utilizando (1) Regresión Lasso, (2) Backward Selection y (3) Algoritmo Genético. El Notebook realiza el entrenamiento de un modelo que evalúa en el conjunto de validación para obtener aquel que provee un mejor desempeño, finalmente se guarda un archivo JSON con la información resultante. Este archivo JSON se llama analisis_var_relevantes.json (se encuentra en la carpeta models/verFinal), el cual incluye la cantidad de variables seleccionada por cada método, el valor del ROC del modelo entrenado en cada método en el conjunto de validación y posteriormente la lista de las variables seleccionadas.

- Después de ejecutado el Notebook, vamos a realizar la selección final de las variables mirando el notebook relacionado al archivo notebooks/Seleccion final de variables.ipynb. Este Notebook toma como entrada las variables elegidas por los 3 métodos anteriores (es decir, el archivo analisis_var_relevantes.json), así procedemos a considerar las variables obtenidas por los 3 métodos, pero también consideraremos la unión de los conjuntos de las variables encontradas, la intersección de los conjuntos de las variables encontradas, y finalmente una votación de las variables con de los 3 conjuntos (Esta votación consta de que, si una variable está incluida en al menos 2 de los 3 conjuntos, va a ser considerada). Posteriormente, consideramos 3 arquitecturas diferentes de clasificadores, así con cada arquitectura, entrenamos un modelo con las variables de todos los conjuntos posibles ya mencionados, y encontramos las métricas promedio de cada uno de los conjuntos de variables, y el que haya tenido mejores métricas en promedio ese será el conjunto de variables seleccionado. Este Notebook, al igual que el anterior, guarda los resultados en un archivo json con las mismas características del anterior. Este se llama vars_relevantes_final.json (se encuentra en la carpeta models/verFinal).

Ya definido tanto la estrategia de resampling a utilizar como las variables relevantes a considerar, procedemos a realizar el entrenamiento del modelo. Para esto, es importante tener claro lo que hacen el siguiente conjunto de archivos:

- El archivo scripts/clase_model/modely.py define un objeto de clase "Modelo" el cual se encargará de tener atributos como: Orden de las columnas en el dataframe necesarios para realizar la predicción del modelo, la versión del modelo entrenado, nombre del mejor modelo seleccionado, fecha de entrenamiento, e información acerca de parámetros relevantes para la creación de variables (por ejemplo, las frecuencias a utilizar para calcular el acumulado histórico de accidentes), diccionario de modelos para considerar en el entrenamiento (este diccionario de modelos contiene tanto el modelo como las diferentes combinaciones de hiperparámetros a ser consideradas durante el entrenamiento). Además de tener estos atributos, este objeto tiene una función de "train", la cual toma un conjunto de datos X e Y, realiza la partición en conjunto de entrenamiento y validación, luego al conjunto de entrenamiento aplica la técnica de resampling encontrada y posteriormente realiza el entrenamiento de los modelos utilizando una búsqueda Grid. En esta búsqueda se elige el mejor modelo en base al mejor desempeño obtenido por los modelos en el conjunto de validación. Así, cuando se crea este objeto, creará una carpeta con el nombre de la versión. Ahí se guardará el mejor modelo después de cada Grid Search. Este objeto también tiene una función para realizar las predicciones, por lo cual tiene la función "predict", esta función se encarga de verificar que el Dataframe pasado para realizar la predicción contenga todas las variables y conserva solo las variables seleccionadas para realizar la predicción de accidentes, en caso de que se tengan variables faltantes, en la función se crearán y se llenarán con un valor de 0.

- El archivo scripts/training.py realiza una instanciación del objeto de clase "Modelo" mencionado anteriormente, luego define un diccionario de modelos con todos los modelos que se desean evaluar al igual que todos los valores de los hiperpárametros a considerar de cada familia de modelo. Este archivo define de igual manera cual será la métrica sobre la cual se realizará la selección del mejor modelo, el número de validaciones cruzadas a utilizar, la cantidad de procesadores o trabajos en paralelo que se desean, al igual que el método de resampling junto con la proporción a utilizar. Seguido, tenemos la lectura de los datos en un rango de fechas especifico (rango de fechas para entrenamiento), y este script lee el archivo vars_relevantes_final.json para dejar en el conjunto de datos únicamente las variables relevantes, Luego llama la función train de la instancia de la clase Modelo para así obtener el mejor modelo en base a la métrica seleccionada. Luego este script crea en PDFs las matrices de confusión en el conjunto de validación a diferentes umbrales, diagramas de violín para ver la distribución de las probabilidades dadas las clases, y las curvas ROC y Precision-Recall. Finalmente, guarda el mejor modelo seleccionado (según su ROC AUC) junto con el objeto de clase Modelo para así ser utilizado (recordar entonces que este objeto tiene las características de las variables utilizadas en el entrenamiento y la función de predict que organiza y garantiza que la información esté completa para realizar predicciones con el modelo seleccionado). Es relevante mencionar que este archivo tiene su propio archivo de logs (model_Training.log), en este archivo se puede monitorear en qué pasos se encuentra el código, además de que queda registrado el valor de las métricas en el conjunto de validación de cada uno de los modelos después de terminar la búsqueda Grid.

- Una vez termina de ejecutar el archivo training.py y realizar un análisis de los desempeños de los modelos encontrados, procedemos a revisar el desempeño de un último clasificador de ensamble. Este clasificador es un Voting Classifier en base a los mejores 3 modelos obtenidos en training.py, así que miramos el archivo notebooks/Voting Classifier.ipynb, el cual realiza la carga tanto de los datos, como de los mejores 3 modelos obtenidos (guardados en archivos .sav en el proceso de la búsqueda Grid). Al momento de crear el modelo de Voting Classifier, utilizamos el parámetro refit=False para que no reentrene (pues ya tenemos los 3 modelos entrenados), posteriormente procedemos a ver el desempeño del modelo ensamblado en el conjunto de validación, y finalmente guardamos el modelo junto con el objeto de clases Modelo obtenido en el script de training.py.

- Se menciona también que, además de los modelos mencionados anteriormente, se evalúa también el desempeño en el conjunto de validación de un modelo logístico de efectos mixtos. Este modelo se desarrolló en R (a diferencia de todo el resto de los scripts que están en Python 3). En la carpeta scripts/Lme se puede encontrar el script de R con el que se desarrolló este modelo, junto con sus salidas y un script de python para evaluar las métricas. Se aclara que finalmente este modelo no fue el modelo elegido, por lo que se deja esta carpeta solo por fines de investigación.


- Finalmente realizaremos la predicción de accidentalidad en base a un modelo entrenado y un objeto de clases Modelo. Por lo tanto, acá consideramos el archivo scripts/test.py, éste archivo lee un archivo .sav que contenga (1) Un objeto de clase Modelo y (2) Un modelo entrenado, carga la información en una ventana de tiempo (periodo de prueba), y luego con el objeto de clase Modelo, realiza la predicción (recordar que en la función predict del objeto filtra las variables seleccionadas, las pone en el orden requerido y realiza la predicción). Una vez se tenga la predicción, el script creará en PDFs las matrices de confusión para diferentes umbrales, al igual que los diagramas de violín y las curvas ROC y Precision-Recall que permitirán ver el desempeño del modelo en el conjunto de prueba.


- Para tener en cuenta: el script de training.py al realizar el entrenamiento, crea 2 archivos csv, data/train.csv y data/validation.csv, luego el archivo de test.py al realizar las predicciones crea el archivo data/test.csv, estos archivos son la entrada para el script scripts/auxiliares/Estandarizacion_datos.py, el cual crea los archivos data/train_z.py, data/validation_z.py y data/test_z.py los cuales son usados para el modelo de efectos mixtos.

Como los archivos generados son pesados, no se encuentran dentro del repositorio, pero pueden ser descargado utilizando los siguientes links (corresponden al Google Drive de uno de los autores)

- train.csv: https://drive.google.com/open?id=1SRll7AINmiJPAH0q0bsmImwxBOz06aBA
- validation.csv: https://drive.google.com/open?id=1QlSk61MghHjMsZg1ERlBdY2iANBKBjrX
- test.csv: https://drive.google.com/open?id=1-mcs6gvWICxYRWXF1G5XzS1eezPHz3LZ

- train_z.csv: https://drive.google.com/open?id=19OM8h5eZUlT3zzyjtusHEHYmUNsbmeSz
- validation_z.csv: https://drive.google.com/open?id=1sc5aMiAh0Yk9pzizDhkxVNcmxt1UBJGB
- test_z.csv: https://drive.google.com/open?id=1j4wPaNlQDvBBrgQgy6U8hgdQKWrFA4sv

Estos archivos anteriores deben ser guardados en la carpeta "data".



Además de estos archivos de texto, los modelos guardados en .sav que se utilizan se pueden descargar del siguiente link:

- Modelos guardados en archivos .sav: https://drive.google.com/open?id=1sXLnTMLlXXrWUVWxj9-rzD1Vj4iNiHfk

El archivo del modelo debe ser guardado en la carpeta "models/verFinal" para que lo encuentren los scripts.



## Caso de aplicación

Teniendo ya determinado el modelo final luego de todo el estudio junto con sus variables, pasamos a utilizar el modelo en un posible caso de aplicación real, que en este caso es usar el modelo para hacer la asignación de los turnos para que patrullen los agentes de tránsito (decirles en que fecha y hora y en que barrios se sugiere que patrullen, buscando que estén en los lugares y tiempos con mayores probabilidades de accidentalidad). Se desarrolló entonces un notebook que se encuentra en Notebooks/AsignarTurnos.ipynb, y en el cual se evalúa el desempeño del modelo al utilizarlo para asignar turnos de agentes de tránsito en la franja de prueba. En el script se compara la cantidad de accidentes de tránsito que hubiera cubierto el modelo, junto con la cantidad de accidentes de tránsito cubiertos si se hubiera hecho la asignación con otros tipos de criterios. La cantidad de agentes de tránsito disponibles para asignar y la cantidad de horas diarias son algunos parámetros que podrían modificarse en este notebook.



### Nota: 
En el trabajo final luego de realizar las pruebas y ejecutar el flujo tal como fue descrito, encontramos que nuestro mejor modelo para el problema que estamos estudiando (según las métricas) fue el clasificador ensamblado (Voting Classifier) y por eso usamos ese en los scripts scripts/test.py y Notebooks/AsignarTurnos.ipynb. Este archivo se llama "verFinal_voting.sav" en la carpeta de Google Drive. Sin embargo, se podría usar cualquier otro modelo.sav que resulta de scripts/training.py para evaluar el desempeño del modelo en el conjunto de prueba y para mirar la asignación de turnos.

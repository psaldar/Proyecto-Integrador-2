Predicción de Accidentes Vehiculares en el Poblado
=======================================================================================


## Description

Este repositorio contiene la informacionrequerida para reproducir el trabajo de "Predicción de Accidentes Vehiculares en el Poblado"

### "Predicción de Accidentes Vehiculares en el Poblado"

Pablo A. Saldarriga<sup>1</sup>, Nicolás Prieto<sup>1</sup>

<sup>1</sup> Maestría en Ciencia de Datos, Universidad EAFIT, Medellin, Colombia


### Problema 

El análisis de accidentes vehiculares es un caso de estudio de gran importancia en cualquier ciudad. Poder hacer inferencia acerca de las características de las zonas donde ocurren accidentes, realizar predicciones acerca de los sitios de la ciudad donde es más probable que ocurra un accidente en una franja determinada de tiempo y analizar tendencias sobre la ocurrencia de estos eventos, es una información que resultaría de gran utilidad para las autoridades de una ciudad. Con esta información, podrían brindar una solución más eficiente y estar más preparados cuando se presenten accidentes.

Resulta entonces de gran importancia realizar un estudio para poder entender mejor las dinámicas de accidentalidad de la ciudad de Medellín (por ejemplo, detectar barrios o días con una dinámica particular de accidentalidad o encontrar agrupaciones de accidentes que probablemente se relacionen a una misma calle, intersección o glorieta). Luego de los análisis anteriores, será entonces de gran utilidad poder contar con un modelo que sea capaz de predecir la ocurrencia de accidentes en distintas zonas de la ciudad, para de esta forma poder conocer los barrios con mayor probabilidad de accidente en cada período de tiempo y poder tomar acciones ante ello.

### Disponibilidad de los datos

Open Data Medellín (https://geomedellin-m-medellin.opendata.arcgis.com/) es una fuente de información de datos libres que provee la alcaldía de Medellín de forma que los ciudadanos puedan conocer y verificar la gestión de los diferentes entes en la ciudad, de igual forma, la información allí presentada ha sido un proceso de consolidación de diferentes procesos de gestión en los entes no gubernamentales de forma que se fomente el uso de tecnologías de información.

En particular de esta fuente de datos, se encuentra disponible información georeferenciada de los diferentes accidentes de tránsito ocurridos en la ciudad de Medellín entre los años 2014 y 2019. Esta base de datos de accidentes tiene información relacionada a:

- Barrio donde ocurrió el accidente
- Ubicación geográfica del accidente (longitud y latitud)
- Comuna donde ocurrió el accidente
- Fecha y hora del accidente
- Dirección del lugar donde ocurrió el accidente
- La gravedad del accidente

Adicional a la información de accidentalidad, se cuenta con datos de las variables climáticas asociados a los barrios de la ciudad, dicha información climática fue obtenida con un servicio de API de la compañía Dark Sky (proveedor de información de variables climáticas, información histórica y pronósticos (https://darksky.net/poweredby/), así para cada barrio y para cada hora, contamos con información de las variables:

- Temperatura
- Temperatura aparente
- Intensidad de precipitación
- Punto de rocío
- Humedad
- Velocidad del viento
- Porcentaje de cobertura de nubes
- Visibilidad 
- Descripción del tipo de día (nublado, soleado, etc.)

### Objetivos

#### Objetivo General

Realizar un estudio general de la accidentalidad en la ciudad de Medellín, combinando técnicas de análisis descriptivo, predictivo y prescriptivo, para así poder entender la dinámica de accidentalidad en la región y predecir en qué barrios y a qué hora ocurrirán accidentes, y de esta forma poder sugerir donde ubicar los agentes de tránsito o abogados de empresas de aseguradoras, para que puedan estar presentes en las zonas con la mayor probabilidad de accidentalidad en cada hora.

#### Objetivos Específicos

- Realizar un análisis descriptivo de la información de accidentalidad de la ciudad de Medellín considerando enfoques de datos funcionales y técnicas de análisis no supervisado.

- Implementar y evaluar diferentes técnicas de aprendizaje automático y métodos estadísticos avanzados para la obtención de un modelo que presente un buen desempeño al momento de predecir la accidentalidad para los barrios de la comuna 14 - ''El Poblado'' de la ciudad de Medellín.
    
- Utilizar técnicas de visualización adecuadas para la presentación de resultados de los distintos análisis de este trabajo, además de mostrar las ventajas que se tendrían al usar el modelo de predicción de accidentes para la asignación de agentes de tránsito o abogados de aseguradoras.


# Flujo para la ejecución del código

Para replicar el trabajo que se realizó, explicamos a continuación el flujo/orden en el cual se deben de ejecutar los códigos en este repositorio.

### Preparación de Datos

Inicialmente, descargamos la información de accidentalidad de Medellín de la página Open Data Medellín de la accidentalidad en los años 2017, 2018 y 2019.

- Accidentes 2017: https://geomedellin-m-medellin.opendata.arcgis.com/datasets/accidentalidad-georreferenciada-2017
- Accidentes 2018: https://geomedellin-m-medellin.opendata.arcgis.com/datasets/accidentalidad-georreferenciada-2018
- Accidentes 2019: https://geomedellin-m-medellin.opendata.arcgis.com/datasets/accidentalidad-georreferenciada-2019

Cada uno de los archivos descargados, serán la entrada para el archivo scripts/auxiliares/proc_raw_accidentes.py. Este script toma cada uno de los archivos descargados y realiza una limpieza en los nombres de los barrios, asignación de la "Ventana de Tiempo (TW)", la cual indica a qué franja horaria pertenece el accidente. Además elimina aquellos registros que se consideran "raros" por los nombres de los accidentes (ya que se encontraron inconsistencias en el cambio del nombre del barrio). Adicionalmente, este script crea un archivo csv con los centroides de cada uno de los barrios (Lat y Lon), de forma tal que en estos puntos es que se consultará la información climática del barrio. Finalmente, el script guarda en una base de datos sqlite la información recopilada.

Una vez se ha organizado la información de los accidentes, procedemos a obtener la información climática para cada uno de los barrios utilizando el archivo scripts/auxiliares/organizar_info_climatica.py. Este script lee el archivo csv que contiene la información de cada barrio con su respectivo centroide generado con el código anterior. Por lo que para una franja de tiempo en específico, este archivo realizará para cada hora y para cada barrio, la consulta de las variables climáticas al API de Darksky (tener en cuenta que para poder realizar la consulta a DarkSky, se debe realizar la suscripción al API para obtener una llave de acceso del API. La llave de acceso generada en este proyecto no se incluye, ya que el uso de esta API puede generar costos asociados a la persona registrada). A medida que va realizando la consulta al API, guarda la información climática en la base de datos sqlite creada en el script anterior en una tabla nueva.

De esta forma, en el sqlite 'database_pi2.sqlite3' se encuentra toda la información del proyecto. Esta base de datos puede ser descargada del siguiente link: https://drive.google.com/open?id=1DHZ4r8gIvHAqHSjvSs3khUOA2vckeeGu (este es un link al archivo almacenado en una carpeta de Google Drive de uno de los autores del trabajo)

### Para tener en cuenta:

Ya que se acaba de almacenar la información en una base de datos sqlite, y teniendo en cuenta que a lo largo del proyecto se estarán utilizando diversas funciones en varios códigos, es importante mencionar que en el archivo scripts/funciones.py se encuentran todas las funciones relevantes del proyecto. Entre esas, se incluye la función que lee directamente la información tanto de los accidentes "crudos" (tal cual vienen de open data medellín, pero con cierta limpieza en sus variables), además de la información de las condiciones climáticas para cada hora y para cada barrio incluyendo si el registro representa un accidente o no. Otras funciones que se encontrarán acá son el entrenamiento de un modelo en base a un Grid o Random Search, la generación de gráficas de evaluación de modelos, además de funciones para cargar los modelos entrenados y almacenados en archivos conextensión .sav

## Modelamiento

- Teniendo en cuenta que el problema que estamos estudiando presenta un gran desbalanceo de clases, lo que hacemos es la evaluación de diferentes métodos de remuestreo (tanto undersampling como oversampling, además de considerar versiones híbridas), además de las proporciones deseadas entre los elementos de las clases 1 y 0. Para ello ejecutamos el archivo scripts/resampling/PruebasResampling.py. Este archivo creará una carpeta donde se podrán encontrar un archivo de logs llamado "resampling.log", a medida que este script va evaluando tando los diferentes métodos de resampling como las diferentes proporciones, escribirá en el archivo log los resultados obtenidos. Es importante mencionar que acá se está utilizando una evaluación del mejor método de resampling con una técnica de envoltura, ya que realizaremos la selección del método en base al desempeño de un modelo entrenado. Aquel modelo que nos de un mejor desempeño, utilizaremos la técnica y proporción de resampling como la mejor.

Una vez se realice en análisis de la mejor técnica de resampling, esta será la técnica usada en los pasos siguientes.

- Luego procedemos a realizar la selección de variables que incluiremos en nuestro modelo de predicción. Para esto se utiliza el Notebook relacionado al archivo notebooks/Feature Selection.ipynb. Este Notebook realiza la carga de todos los datos, realiza la partición en entrenamiento y validación, y posteriormente eliminación de variables utilizando criterios mirando la correlación y correlación parcial, luego realiza la selección de variables utilizando (1) Regresión Lasso, (2) Backward Selection y (3) Algoritmo Genético. El Notebook realiza el entrenamiento de un modelo que evalua en el conjunto de validación para obtener aquel que provee un mejor desempeño, finalmente se guarda un archivo JSON con la información resultante. Este archivo JSON se llama analisis_var_relevantes.json, el cual incluye la cantidad de variables seleccionada por cada método, el valor del ROC del modelo entrenado en cada método en el conjunto de validación y posteriormente la lista de las variables seleccionadas.

- Después de ejecutado el Notebook, vamos a realizar la selección final de las variables mirando el notebook relacionado al archivo notebooks/Seleccion final de variables.ipynb. Este Notebook toma como entrada las variables elegidas por los 3 métodos anteriores (es decir, el archivo analisis_var_relevantes.json), así procedemos a considerar las variables obtenidas por los 3 métodos, pero también consideraremos la union de los conjuntos de las variables encontradas, la intersección de los conjuntos de las variables encontradas, y finalmente una votación de las variables con de los 3 conjuntos (Esta votación consta de que, si una variable esta incluida en al menos 2 de los 3 conjuntos, va a ser considerada). Posteriormente, consideramos 3 arquitecturas diferentes de clasificadores, asi con cada arquitectura, entrenamos un modelo con las variables de todos los conjuntos posibles ya mencionados, y encontramos las métricas promedio de cada uno de los conjuntos de variables, así en base al que haya tenido mejores métricas en promedio, ese será el conjunto de variables seleccionado. Este Notebook al igual que el anterior, guarda los resultados en un archivo json con las mismas caracteristicas del anterior. Este se llama  vars_relevantes_final.json.

Ya definido tanto la estrategia de resampling a utilzar como las variables relevantes a considerar, procedemos a realizar el entrenamiento del modelo. Para esto, es importante tener claro lo que hacen el siguiente conjunto de archivos:

- El archivo scripts/clase_model/modely.py define un objeto de clase "Modelo" el cual se encargará de tener atributos como: Orden de las columnas en el dataframe necesarios para realizar la predicción del modelo, la versión del modelo entrenado, nombre del mejor modelo seleccionado, fecha de entrenamiento, e información acerca de parametros relevantes para la creación de variables (por ejemplo, las frecuencias a utilizar para calcular el acumulado historico de fallas), diccionario de modelos para considerar en el entrenamiento (este diccionario de modelos contiene tanto el modelo como las diferentes combinaciones de hiperparametros a ser consideradas durante el entrenamiento). Además de tener estos atributos, este objeto tiene cuna función de "train", la cual toma un conjunto de datos X e Y, realiza la partición en conjunto de entrenamiento y validación, luego al conjunto de entrenamiento aplica la técnica de resampling encontrada y posteriormente realiza el entrenamiento de los modelos utilizando una búsqueda Grid. En esta búsqueda se elije el mejor modelo en base al mejor desempeño obtenido por los modelos en el conjunto de validación. Así, cuando se crea este objeto, creará una carpeta con el nombre de la versión. Ahí se guardará el mejor modelo despues de cada Grid search. Este objeto también tiene una función para realizar las predicciones, por lo cual tiene la funcion "predict", esta función se encarga de verificar que el DataFrame pasado para realizar la predicción contenga todas las variables y selecciona solo esas variables para realizar la predicción de accidentes, en caso que se tengan variables faltantes, ahí en la función se crearán y se llenarán con un valor de 0.

- El archivo scripts/training.py realiza una instanciación del objeto de clase "Modelo" creado anteriormente, luego define un diccionario de modelos con todos los modelos que se desean evaluar al igual que todos los valores de los hiperparametros a considerar. Este archivo define de igual manera cual será la métrica sobre la cual se realizará la selección del mejor modelo, el número de validaciones cruzadas a utilizar, la cantidad de procesadores o trabajos en paralelo que se desean, al igual que el método de resampling junto con la proporción a utilizar. Seguido, tenemos la lectura de los datos en un rango de fechas especifico (rango de fechas para entrenamiento), y este script lee el archivo vars_relevantes_final.json para dejar en el conjunto de datos unicamente las variables relevantes, Luego llama la función train de la instancia de la clase Modelo para así obtener el mejor modelo en base a la métrica seleccionada. Luego este script crea en PDFs las matrices de confusión en el conjunto de validación a diferentes umbrales, diagramas de violín para ver la distribución de las probabilidades dadas las clases, y las curvas ROC y Precision-Recall. Finalmente, guarda el mejor modelo seleccionado junto con el objeto de clase Modelo para así ser utilizado (recordar entonces que este objeto tiene las caracteristicas de las variables utilizadas en el entrenamiento y la función de predict que organiza y garantiza que la información esté completa para realizar predicciones con el modelo seleccionado). Es relevante mencionar que este archivo tiene su propio archivo de logs (model_Training.log), en este archivo se puede monitoriear en qué pasos se encuentra el código, además de que queda registrado el valor de las métricas en el conjunto de validación de cada uno de los modelos después de terminar el la búsqueda Grid.

- Una vez termina de ejecutar el archivo training.py y realizar un análisi de los desempeños de los modelos encontrados, procedemos a revisar el desempeño de un último clasificador de ensamble. Este clasificador es un Voting Classifier en base a los mejores 3 modelos obtenidos en trainin.py, así que miramos el archivo notebooks/Voting Classifier.ipynb, el cual realiza la carga tanto de los datos, como de los mejores 3 modelos obtenidos (guardados en archivos .sav en el proceso de la búsqueda Grid). Al momento de crear el modelo de Voting Classifier, utilizamos el parametro refit=False para que no reentrene (pues ya tenemos los 3 modelos entrenados), posteriormente procedemos a ver el desempeño del modelo ensamblado en el conjunto de validación, y finalmente guardamos el modelo junto con el objeto de clases Modelo obtenido en el script de training.py (Después de esto está la evaluación del conjunto de prueba, pero en este momento aún no se puede realizar. 

- Finalmente realizaremos la predicción de accidentalidad en base a un modelo entrenado y un objeto de clases Modelo. Por lo tanto, acá consideramos el archivo scripts/test.py este archivo lee un archivo .sav que contenga (1)Un objeto de clase Modelo y (2)Un modelo entrenado, carga la información en una ventana de tiempo (periodo de prueba), y luego con el objeto de clase Modelo, realiza la predicción (recordar que el la función predict del objeto filtra las variables seleccionadas, las pone en el orden requerido y realiza la predicción). Una vez se tenga la predicción, el script creará en PDFs las matrices de confusión para diferentes umbrales, al igual que los diagramas de violín y las curvas ROC y Precisipon-Recall que permitirán ver el desempeño del modelo en el conjunto de entrenamiento.

## Caso de aplicación

NICOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
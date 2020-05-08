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
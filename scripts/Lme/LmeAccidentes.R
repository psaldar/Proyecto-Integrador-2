##### Mixed effects logistic regression para accidentes

require(lme4)
library(ModelMetrics)


### Leer datos (el working directory debe ser la carpeta donde esta este script)
train_dat = read.csv("../../data/train.csv", encoding = "UTF-8")
val_dat = read.csv("../../data/validation.csv", encoding = "UTF-8")
test_dat = read.csv("../../data/test.csv", encoding = "UTF-8")


### Linear mixed effects model
m <- glmer(Accidente ~ poblado_altosdelpoblado+poblado_astorga+poblado_castropol+poblado_elcastillo+poblado_eldiamanteno2+poblado_laaguacatala+poblado_lalinde+poblado_losbalsosno1+poblado_losnaranjos+poblado_manila+poblado_sanlucas+poblado_santamariadelosangeles+poblado_villacarlota+dewPoint+humidity+windSpeed+cloudCover+temperature+uvIndex+visibility+hora_1+hora_2+hora_3+hora_6+hora_7+hora_9+hora_10+hora_14+hora_16+hora_19+hora_20+hora_21+hora_23+icon_partly.cloudy.day+icon_partly.cloudy.night+icon_rain+dia_sem_3+dia_sem_4+dia_sem_5+dia_sem_6+humidity_mean+windSpeed_mean+
             (0+visibility| BARRIO)+(0+temperature| BARRIO)+(0+dia_sem_4| BARRIO)+(0+dia_sem_3| BARRIO)+(dia_sem_6 | BARRIO), data = train_dat, family = binomial(link="logit"), control = glmerControl(optimizer = "bobyqa")
           )
summary(m)


### Predicciones validacion
predictio_v = predict(m, newdata=val_dat,type="response")

### Exportar predicciones validacion
valpr = data.frame(predictio_v)
write.table(valpr, "predicciones_val_r.txt", sep=";")


### Predicciones test
predictio_t = predict(m, newdata=test_dat,type="response")

### Exportar predicciones test
testpr = data.frame(predictio_t)
write.table(testpr, "predicciones_test_r.txt", sep=";")
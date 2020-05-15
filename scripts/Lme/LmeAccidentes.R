##### Mixed effects logistic regression para accidentes

require(lme4)
library(ModelMetrics)


### Leer datos (el working directory debe ser la carpeta donde esta este script)
train_dat = read.csv("../../data/train_z.csv", encoding = "UTF-8")
val_dat = read.csv("../../data/validation_z.csv", encoding = "UTF-8")
test_dat = read.csv("../../data/test_z.csv", encoding = "UTF-8")


### Linear mixed effects model
m <- glmer(Accidente ~ precipIntensity + precipProbability + uvIndex + visibility + hora_0 + hora_1 + hora_2 + hora_3 + hora_4 + hora_5 + hora_7 + hora_11 + hora_13 + hora_15 + hora_16 + hora_17 + hora_18 + hora_19 + hora_20 + hora_22 + hora_23 + icon_clear.day + icon_cloudy + icon_fog + dia_sem_4 + dia_sem_5 + dia_sem_6 + festivo + Mes_Abril + Mes_Agosto + Mes_Enero + Mes_Febrero + Mes_Julio + Mes_Mayo + Mes_Septiembre + Year_2017 + Year_2018 + Year_2019 + cloudCover_mean + precipIntensity_mean + visibility_mean + windSpeed_mean + cloudCover_mean_forward + dewPoint_mean_forward + precipIntensity_mean_forward + temperature_mean_forward + cumAcc_30D + poblado_altosdelpoblado + poblado_astorga + poblado_barriocolombia + poblado_castropol + poblado_elcastillo + poblado_eldiamanteno2 + poblado_elpoblado + poblado_laaguacatala + poblado_laflorida + poblado_lalinde + poblado_laslomasno1 + poblado_laslomasno2 + poblado_losbalsosno1 + poblado_losbalsosno2 + poblado_losnaranjos + poblado_manila + poblado_patiobonito + poblado_sanlucas + poblado_santamariadelosangeles + poblado_villacarlota+
             (1 | BARRIO), data = train_dat, family = binomial(link="logit"), control = glmerControl(optimizer = "bobyqa")
           )
summary(m)


### Predicciones validacion
predictio_v = predict(m, newdata=val_dat,type="response")

### Exportar predicciones validacion
valpr = data.frame(predictio_v, val_dat$Accidente)
write.table(valpr, "predicciones_val_r.txt", sep=";")

### Roc auc val
library(pROC)
roc_obj <- roc(val_dat$Accidente, predictio_v)
auc(roc_obj)


### Predicciones test
predictio_t = predict(m, newdata=test_dat,type="response")

### Exportar predicciones test
testpr = data.frame(predictio_t)
write.table(testpr, "predicciones_test_r.txt", sep=";")
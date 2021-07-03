library(tidyverse)
library(tidymodels)
library(randomForest)
library(readr)
library(e1071)
library(rpart)
library(caret)
library(adabag)
library(ROCR)
library(gplots)
library(tree)
library(neuralnet)
library(rpart.plot)
library(kableExtra)
library(discrim)
library(leaps)
library(xgboost)

setwd("D:/monash/etc3250ML/project")
# Read data
fires_tr_full <- read.csv("student_training_release.csv") %>%
  mutate(cause = factor(cause))
fires_to_predict <- read.csv("student_predict_x_release.csv")

str(fires_tr_full)
summary(fires_tr_full)
fires_tr_full<- na.omit(fires_tr_full)
#rf subset selection
regfit.p1 <- regsubsets(cause~rf+arf7+arf14+arf28+arf60+arf90+arf180+arf360+arf720
                            ,fires_tr_full)
reg.summary<-summary(regfit.p1)
coef(regfit.p1,1)
#se subset selection
regfit.p2 <- regsubsets(cause~se+ase7+ase14+ase28+ase60+ase90+ase180+ase360+ase720
                          ,fires_tr_full)
reg.summary<-summary(regfit.p2)
coef(regfit.p2,1)
#maxt subset selection
regfit.p3 <- regsubsets(cause~maxt+amaxt7+amaxt14+amaxt28+amaxt60+amaxt90+
                             amaxt180+amaxt360+amaxt720
                           ,fires_tr_full)
reg.summary<-summary(regfit.p3)
coef(regfit.p3,1)
#mint subset selection
regfit.p4 <- regsubsets(cause~mint+amint7+amint14+amint28+amint60+amint90+
                             amint180+amint360+amint720
                           ,fires_tr_full)
reg.summary<-summary(regfit.p4)
coef(regfit.p4,1)
#ws subset selection
regfit.p5 <- regsubsets(cause~ws+aws_m0+aws_m1+aws_m3+aws_m6+aws_m12+aws_m24
                           ,fires_tr_full)
reg.summary<-summary(regfit.p5)
coef(regfit.p5,1)
#distance subset selection
regfit.p6 <- regsubsets(cause~dist_cfa+dist_camp+dist_road
                           ,fires_tr_full)
reg.summary<-summary(regfit.p6)
coef(regfit.p6,2)
#other subset selection
regfit.p7 <- regsubsets(cause~lon+lat+month+day+COVER+HEIGHT+FOREST
                           ,fires_tr_full)
reg.summary<-summary(regfit.p7)
coef(regfit.p7,4)

#subset selection
fires_tr_full_sele<-fires_tr_full%>%
  select(lon,lat,month,HEIGHT,dist_cfa ,dist_camp,aws_m0
         ,amint28,amaxt360,ase7,arf360,cause)


regfit.full<- regsubsets(cause~.,fires_tr_full_sele)

reg.summary<-summary(regfit.full)
#
models <- tibble(nvars=1:8,rsq=reg.summary$rsq, 
                 rss=reg.summary$rss, 
                 adjr2=reg.summary$adjr2, 
                 cp=reg.summary$cp, 
                 bic=reg.summary$bic)
p1 <- ggplot(models, aes(x=nvars, y=rsq)) + geom_line()
p2 <- ggplot(models, aes(x=nvars, y=rss)) + geom_line()
p3 <- ggplot(models, aes(x=nvars, y=adjr2)) + geom_line()
p4 <- ggplot(models, aes(x=nvars, y=cp)) + geom_line()
p5 <- ggplot(models, aes(x=nvars, y=bic)) + geom_line()
p1
#Fit forward stepwise selection
regfit.fwd <- regsubsets(cause~.,fires_tr_full_sele, method="forward")
summary(regfit.fwd)

d <- tibble(nvars=1:12,  
            rss=regfit.fwd$rss)
ggplot(d, aes(x=nvars, y=rss)) + geom_line()
#Full model
coef(regfit.full, 6)
#Forward selection
coef(regfit.fwd, 6)


set.seed(3000)
split <- initial_split(fires_tr_full, 3/4, strata = "cause")
fires_tr <- training(split)
fires_ts <- testing(split)

# Fit a basic model
#LDA
lda_mod <- discrim_linear() %>% 
  set_engine("MASS") %>% 
  translate()

crabs_lda_fit <- 
  lda_mod %>% 
  fit(cause~lon+month+dist_cfa+dist_camp+aws_m0+ase7, data=fires_tr)

fires_to_predict_p1 <- fires_to_predict %>%
  mutate(cause_p1 = predict(crabs_lda_fit, fires_to_predict)$.pred_class) %>%
  select(id, cause_p1) %>%
  dplyr::rename(Id = id, Category = cause_p1)
write_csv(fires_to_predict_p1, file="model1.csv")

#random forest model
fires_rf <- rand_forest() %>%
  set_engine("randomForest",
             importance=TRUE, proximity=TRUE) %>%
  set_mode("classification") %>%
  fit(cause~lon+month+dist_cfa+dist_camp+aws_m0+ase7, data=fires_tr)
options(digits=2)
fires_rf$fit$importance

fires_to_predict_p2 <- fires_to_predict %>%
  mutate(cause_p2 = predict(fires_rf, fires_to_predict)$.pred_class) %>%
  select(id, cause_p2) %>%
  dplyr::rename(Id = id, Category = cause_p2)
write_csv(fires_to_predict_p2, file="model2.csv")

#boost tree model

mymodel <- boost_tree(mtry = tune(),
                      trees = tune(),
                      tree_depth = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")
mycv <- vfold_cv(fires_tr_full, v = 5, strata = "cause")

myrecipe <- recipe(fires_tr_full, formula = cause ~ 
                     lon+month+dist_cfa+dist_camp+aws_m0+ase7) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_knn(all_predictors(), neighbors = tune()) %>%
  step_dummy(all_nominal_predictors())

myworkflow <- workflow() %>%
  add_recipe(myrecipe) %>%
  add_model(mymodel)

mygrid <- expand.grid(mtry = c(5, 7, 9),
                      neighbors = c(3, 5),
                      trees = seq(200, 1000, 200),
                      tree_depth = c(5, 7, 9))

mycontrol <- control_grid(verbose = TRUE)

grid_result <- myworkflow %>%
  tune_grid(resamples = mycv, grid = mygrid, control = mycontrol)

autoplot(grid_result)

mybestpara <- select_best(grid_result, metric = "accuracy")

myworkflow <- myworkflow %>%
  finalize_workflow(mybestpara)

myworkflow <- myworkflow %>%
  fit(data = mydat)


mypred <- myworkflow %>%
  predict(fires_to_predict)

data.frame(Category = mypred$.pred_class) %>%
  mutate(Id = 1:n()) %>%
  select(Id, Category) %>%
  write_csv("model3.csv")



rm(list = ls())
library(randomForest) #for random forests
library(caret) # for CV folds and data splitting
library(gbm)
library(plyr)
library(ROCR) # for diagnostics and ROC plots/stats
library(pROC) # same as ROCR
library(stepPlr) 
library(dplyr)


mt <- read.csv("MTURK_NO_RACE.csv")

# Broward data
br <- read.csv("BROWARD_CLEAN.csv")
rownames(br) <- as.character(br$id)

# Charge id codes and names
charge <- read.csv("CHARGE_ID.csv")
charge$mturk_charge_name <- as.character(charge$mturk_charge_name)


# incorporate charge names
br$mturk_charge_name <- NA
for (i in 1:nrow(br)){
  br$mturk_charge_name[i] <- charge$mturk_charge_name[charge$charge_id == 
                                                        br$charge_id[i]]
}
br$mturk_charge_name <- factor(br$mturk_charge_name)


test.units <- sort(mt[,1])

#subset training units
Lframe.tune <- br[!(br$id %in% test.units),
                  c("two_year_recid","sex","age","juv_fel_count",
                    "juv_misd_count","priors_count","charge_degree..misd.fel.", "mturk_charge_name")]

#Pre-process mturk_charge_name with Empirical Bayes Beta-Bernoulli
mu <- mean(Lframe.tune$two_year_recid)
s2 <- var(Lframe.tune$two_year_recid)/nrow(Lframe.tune)
alpha <- (((1-mu)/(s2))-(1/mu))*mu^2
beta <- alpha*((1/mu) - 1)

gmeans <- data.frame(Lframe.tune %>% group_by(mturk_charge_name) 
                     %>% summarise(gmean = mean(two_year_recid), n = n()))
gmeans$EBmean <- ((alpha + beta)/(alpha + beta + gmeans$n)) * mu +
  (1 - (alpha + beta)/(alpha + beta + gmeans$n)) * gmeans$gmean

Lcharges <- as.character(unique(gmeans$mturk_charge_name))

Lframe.tune$charge_scalar <- NA

for (i in 1:nrow(Lframe.tune)){
  Lframe.tune$charge_scalar[i] <- gmeans$EBmean[gmeans$mturk_charge_name == Lframe.tune$mturk_charge_name[i]]
}

Lframe.tune <- Lframe.tune[,-which(names(Lframe.tune) == "mturk_charge_name")]

Lframe.tune$two_year_recid <- factor(
  Lframe.tune$two_year_recid,
  levels=c(0,1),
  labels=c("No", "Offence"))

positives = subset(Lframe.tune, two_year_recid == "Offence")
negatives = subset(Lframe.tune, two_year_recid == "No")

cf_metric_fun <- function(preds, actuals){
  
  predicts_g <- ifelse(preds > 0.5, "Offence", "No")
  
  tp = as.numeric(sum(tst$two_year_recid == "Offence" & predicts_g == "Offence"))
  fp = as.numeric(sum(tst$two_year_recid == "No" & predicts_g == "Offence"))
  fn = as.numeric(sum(tst$two_year_recid == "Offence" & predicts_g == "No"))
  tn = as.numeric(sum(tst$two_year_recid == "No" & predicts_g == "No"))
  
  n = tp + fp + fn + tn
  p = c((tp + fn)/n, (tn + fp)/n)
  q = c((tp + fp)/n, (tn + fn)/n)
  
  prevalence = (tp + fn)/n
  
  accuracy = (tp + tn) / (tp + fp + tn + fn)
  tpr = tp / (tp + fn)
  tnr = tn / (tn + fp)
  ppv = tp / (tp + fp) 
  npv = tn / (tn + fn)
  baccuracy = 1/2 *(tp/(tp + fn) + tn/(tn + fp))
  informedness = tp/(tp + fn) + tn/(tn + fp) - 1
  F1_score = tp/(tp + 1/2 * (fp + fn))
  gmean = sqrt(tpr * tnr)
  mcc <- (tp * tn - fp * fn)/sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  Fowlkes_Mallows_index = sqrt(ppv * tpr)
  positive_likelihood_ratio = tpr/(1 - tnr)
  negative_likelihood_ratio = (1 - tpr)/tnr
  markedness = ppv + npv - 1
  diagnostic_odds_ratio = positive_likelihood_ratio/negative_likelihood_ratio
  jaccardindex = tp/(tp+fp+fn)
  expAccuracy = sum(p * q)
  cohens_kappa =  (accuracy - expAccuracy) / (1 - expAccuracy)
  f_beta_score_0.5 = (1 + 0.5^2)*((ppv*tpr)/((1+0.5^2)*ppv+tpr))
  f_beta_score_2 = (1 + 2^2)*((ppv*tpr)/((1+2^2)*ppv+tpr))
  
  auc = auc(response = actuals, predictor = preds)
  
  return(c(prevalence, tp, fn, tn, fp, tpr, tnr, ppv, npv, accuracy, baccuracy, informedness, F1_score, mcc, gmean, Fowlkes_Mallows_index, markedness, diagnostic_odds_ratio, jaccardindex, cohens_kappa, f_beta_score_0.5, f_beta_score_2, auc))
  
}

# To reproduce results exactly, we have to set seed before any random sampling steps

# model with original data, run 0
columns = c("Model", "Prevalence", "True_positives", "False_negatives", "True_negatives", "False_positives", "TPR", "TNR", "PPV", "NPV", "Accuracy", "BA", "BI", "F1_score", "MCC", "Gmean", "Fowlkes_Mallows_index", "Markedness", "Diagnostic_odds_ratio", "Jaccardindex", "Cohens_kappa", "F_beta_score_0.5", "F_beta_score_2", "AUC", "Run") 
metricsdf = data.frame(matrix(nrow = 0, ncol = length(columns))) 
colnames(metricsdf) = columns

set.seed(20230516)
# test-train split
trn_idx = sample(nrow(Lframe.tune), size = round(0.8 * nrow(Lframe.tune)))
tst = Lframe.tune[-trn_idx, ]
trn = Lframe.tune[trn_idx, ]

set.seed(20230516)
tc<-trainControl(method="cv", 
                 number=10,
                 summaryFunction=twoClassSummary, 
                 classProb=T)

model.glm <- train(as.factor(two_year_recid)~.,
                   metric="ROC", method="glm", family="binomial",
                   trControl=tc, data=trn)

model.rf<-train(as.factor(two_year_recid)~., 
                metric="ROC", method="rf", 
                importance=T, # Variable importance measures retained
                proximity=F, ntree=700, # number of trees grown
                trControl=tc, data=trn)

model.lda = train(as.factor(two_year_recid) ~ ., method = 'lda',trControl=tc, data = trn)

set.seed(20230516)
model.gbm <-train(as.factor(two_year_recid)~.,
                  method="gbm", 
                  trControl=tc, data=trn)

model.knn <-train(as.factor(two_year_recid)~.,
                  method="knn", 
                  trControl=tc, data=trn)
# random guess
n = nrow(tst)
randomguess = rep(0, n) 
np = nrow(tst[tst$two_year_recid == "Offence", ])
set.seed(20230516)
ridx = sample(n, np)
set.seed(20230516)
randomguess[ridx] = replicate(np, sample(seq(0.51, 0.99, 0.01),1))
set.seed(20230516)
randomguess[-ridx] = replicate((n - np), sample(seq(0.01, 0.49, 0.01),1))

metrics = matrix(rep(0, 138), 6, 23)

predictprob1 <- predict(model.glm, newdata = tst, type="prob")
predictprob2 <- predict(model.rf, newdata = tst, type="prob")
predictprob3 <- predict(model.knn, newdata = tst, type="prob")
predictprob4 <- predict(model.lda, newdata = tst, type="prob")
predictprob5 <- predict(model.gbm, newdata = tst, type="prob")

metrics[1, ] = cf_metric_fun(preds = predictprob1[,2], actuals = tst$two_year_recid)
metrics[2, ] = cf_metric_fun(preds = predictprob2[,2], actuals = tst$two_year_recid)
metrics[3, ] = cf_metric_fun(preds = predictprob3[,2], actuals = tst$two_year_recid)
metrics[4, ] = cf_metric_fun(preds = predictprob4[,2], actuals = tst$two_year_recid)
metrics[5, ] = cf_metric_fun(preds = predictprob5[,2], actuals = tst$two_year_recid)
metrics[6,] = cf_metric_fun(preds = randomguess, actuals = tst$two_year_recid)

crimemetrics = data.frame(cbind(c("Logistic Regression", "Random Forest", "KNN", "LDA", "GBM", "randomguess"), metrics, rep(0, 6)))

colnames(crimemetrics) <- columns

metricsdf = rbind(metricsdf, crimemetrics)

# phase 1 decreasing prevalence level
# this takes around 4 hours
for (i in seq(1, 76)){
  set.seed(20230516)
  pidx = sample(2775, size = 30 * i)
  nidx = sample(3439, size = 30 * i)
  newpositives = positives[-pidx, ]
  addnegatives = negatives[nidx, ]
  newdata = rbind(newpositives, negatives, addnegatives)
  
  set.seed(20230516)
  # test-train split
  trn_idx = sample(nrow(newdata), size = 0.8 * nrow(newdata))
  tst = newdata[-trn_idx, ]
  trn = newdata[trn_idx, ]
  
  set.seed(20230516)
  tc<-trainControl(method="cv", 
                   number=10,
                   summaryFunction=twoClassSummary, 
                   classProb=T)
  
  model.glm <- train(as.factor(two_year_recid)~.,
                     metric="ROC", method="glm", family="binomial",
                     trControl=tc, data=trn)
  
  model.rf<-train(as.factor(two_year_recid)~.,
                  metric="ROC", method="rf",
                  importance=T,
                  proximity=F, ntree=700,
                  trControl=tc, data=trn)
  
  model.lda = train(as.factor(two_year_recid) ~ ., method = 'lda',trControl=tc, data = trn)
  
  set.seed(20230516)
  model.gbm <-train(as.factor(two_year_recid)~.,
                    method="gbm", 
                    trControl=tc, data=trn)
  
  model.knn <-train(as.factor(two_year_recid)~.,
                    method="knn",
                    trControl=tc, data=trn)
  
  # random guess
  n = nrow(tst)
  randomguess = rep(0, n) 
  np = nrow(tst[tst$two_year_recid == "Offence", ])
  set.seed(20230516)
  ridx = sample(n, np)
  set.seed(20230516)
  randomguess[ridx] = replicate(np, sample(seq(0.51, 0.99, 0.01),1))
  set.seed(20230516)
  randomguess[-ridx] = replicate((n - np), sample(seq(0.01, 0.49, 0.01),1))
  
  metrics = matrix(rep(0, 138), 6, 23)
  
  predictprob1 <- predict(model.glm, newdata = tst, type="prob")
  predictprob2 <- predict(model.rf, newdata = tst, type="prob")
  predictprob3 <- predict(model.knn, newdata = tst, type="prob")
  predictprob4 <- predict(model.lda, newdata = tst, type="prob")
  predictprob5 <- predict(model.gbm, newdata = tst, type="prob")
  
  metrics[1,] = cf_metric_fun(preds = predictprob1[,2], actuals = tst$two_year_recid)
  metrics[2,] = cf_metric_fun(preds = predictprob2[,2], actuals = tst$two_year_recid)
  metrics[3,] = cf_metric_fun(preds = predictprob3[,2], actuals = tst$two_year_recid)
  metrics[4,] = cf_metric_fun(preds = predictprob4[,2], actuals = tst$two_year_recid)
  metrics[5,] = cf_metric_fun(preds = predictprob5[,2], actuals = tst$two_year_recid)
  metrics[6,] = cf_metric_fun(preds = randomguess, actuals = tst$two_year_recid)
  
  crimemetrics = data.frame(cbind(c("Logistic Regression", "Random Forest", "KNN", "LDA", "GBM", "randomguess"), metrics, rep(i, 6)))
  
  colnames(crimemetrics) <- columns
  
  metricsdf = rbind(metricsdf, crimemetrics)
  
  print(paste('This is run: ', i))
}

# phase 2 increasing prevalence level
# this takes about 4 hours 
for (i in seq(1, 79)){
  set.seed(20230516)
  pidx = sample(2775, size = 30 * i)
  nidx = sample(3439, size = 30 * i)
  addpositives = positives[pidx, ]
  newnegatives = negatives[-nidx, ]
  newdata = rbind(positives, addpositives, newnegatives)
  
  set.seed(20230516)
  # test-train split
  trn_idx = sample(nrow(newdata), size = 0.8 * nrow(newdata))
  tst = newdata[-trn_idx, ]
  trn = newdata[trn_idx, ]
  
  set.seed(20230516)
  tc<-trainControl(method="cv", 
                   number=10,
                   summaryFunction=twoClassSummary, 
                   classProb=T)
  
  model.glm <- train(as.factor(two_year_recid)~.,
                     metric="ROC", method="glm", family="binomial",
                     trControl=tc, data=trn)
  
  model.rf<-train(as.factor(two_year_recid)~.,
                  metric="ROC", method="rf",
                  importance=T,
                  proximity=F, ntree=700,
                  trControl=tc, data=trn)
  
  model.lda = train(as.factor(two_year_recid) ~ ., method = 'lda',trControl=tc, data = trn)
  
  set.seed(20230516)
  model.gbm <-train(as.factor(two_year_recid)~.,
                    method="gbm", 
                    trControl=tc, data=trn)
  
  model.knn <-train(as.factor(two_year_recid)~.,
                    method="knn",
                    trControl=tc, data=trn)
  
  # random guess
  n = nrow(tst)
  randomguess = rep(0, n) 
  np = nrow(tst[tst$two_year_recid == "Offence", ])
  set.seed(20230516)
  ridx = sample(n, np)
  set.seed(20230516)
  randomguess[ridx] = replicate(np, sample(seq(0.51, 0.99, 0.01),1))
  set.seed(20230516)
  randomguess[-ridx] = replicate((n - np), sample(seq(0.01, 0.49, 0.01),1))
  
  metrics = matrix(rep(0, 138), 6, 23)
  
  predictprob1 <- predict(model.glm, newdata = tst, type="prob")
  predictprob2 <- predict(model.rf, newdata = tst, type="prob")
  predictprob3 <- predict(model.knn, newdata = tst, type="prob")
  predictprob4 <- predict(model.lda, newdata = tst, type="prob")
  predictprob5 <- predict(model.gbm, newdata = tst, type="prob")
  
  metrics[1,] = cf_metric_fun(preds = predictprob1[,2], actuals = tst$two_year_recid)
  metrics[2,] = cf_metric_fun(preds = predictprob2[,2], actuals = tst$two_year_recid)
  metrics[3,] = cf_metric_fun(preds = predictprob3[,2], actuals = tst$two_year_recid)
  metrics[4,] = cf_metric_fun(preds = predictprob4[,2], actuals = tst$two_year_recid)
  metrics[5,] = cf_metric_fun(preds = predictprob5[,2], actuals = tst$two_year_recid)
  metrics[6,] = cf_metric_fun(preds = randomguess, actuals = tst$two_year_recid)
  
  crimemetrics = data.frame(cbind(c("Logistic Regression", "Random Forest", "KNN", "LDA", "GBM", "randomguess"), metrics, rep(i, 6)))
  
  colnames(crimemetrics) <- columns
  
  metricsdf = rbind(metricsdf, crimemetrics)
  
  print(paste('This is run: ', i))
}

write.csv(metricsdf, "finalall.csv", row.names = FALSE)


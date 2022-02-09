df <- churn_clean

library(naivebayes)
library(pROC)
library(OneR)
library(caret)
library(writexl)

df1 <- df[, c("Bandwidth_GB_Year",  "MonthlyCharge", "Contract", "Churn")]

set.seed(1)
row <- nrow(df1)
n <- runif(row)
train <- df1[n < 0.75, ]
test <- df1[n >= 0.75, ]

df2 <- df1
df2$Bandwidth_GB_Year <- bin(df2$Bandwidth_GB_Year, nbins = 4, labels = 
                               c("Lowest 25%", "1QR 25%", "3QR 25%", "Highest 25%"), 
                             method = "content")
df2$MonthlyCharge <- bin(df2$MonthlyCharge, nbins = 4, labels = 
                           c("Lowest 25%", "1QR 25%", "3QR 25%", "Highest 25%"), 
                         method = "content")  #include source from rdocument

df2$Churn <- as.factor(df2$Churn)
df2$Contract <- as.factor(df2$Contract)

set.seed(1)
row2 <- nrow(df2)
n2 <- runif(row2)
train2 <- df2[n < 0.75, ]
test2 <- df2[n >= 0.75, ]

#bandwidth, non-binned
nb_band <- naive_bayes(Churn ~ Bandwidth_GB_Year, data = train)
summary(nb_band)
nb_band
pred_band <- predict(nb_band, test[1], type = "prob")
pred_band1 <- predict(nb_band, test[1])
roc_band <- roc(response = test$Churn, predictor = pred_band[,2])
plot(roc_band, col = "blue", main = "Bandwidth")
(roc_band[["auc"]])

table <- table(test$Churn, pred_band1)
(cm <- confusionMatrix(table))

#bandwidth, binned
nb_band2 <- naive_bayes(Churn ~ Bandwidth_GB_Year, data = train2)
summary(nb_band2)
nb_band2
pred_band2 <- predict(nb_band2, test2[1], type = "prob")
roc_band2 <- roc(response = test2$Churn, predictor = pred_band2[,2])
plot(roc_band2, col = "blue", main = "Bandwidth")
auc(roc_band2)

#monthly charge 
nb_mon <- naive_bayes(Churn ~ MonthlyCharge, data = train)
pred_mon <- predict(nb_mon, test[2], type = "prob")
pred_mon1 <- predict(nb_mon, test[2])
roc_mon <- roc(response = test$Churn, predictor = pred_mon[,2])
plot(roc_mon, col = "blue", main = "Monthly Charge")
(roc_mon[["auc"]])

table_mon <- table(test$Churn, pred_mon1)
(cm_mon <- confusionMatrix(table_mon))

#contract 
nb_con <- naive_bayes(Churn ~ Contract, data = train)
pred_con <- predict(nb_con, test[3], type = "prob")
pred_con1 <- predict(nb_con, test[3])
roc_con <- roc(response = test$Churn, predictor = pred_con[,2])
plot(roc_con, col = "blue", main = "Contract Type")
(roc_con[["auc"]])

table_con <- table(test$Churn, pred_con1)
(cm_con <- confusionMatrix(table_con))

#ALL THREE 
nb_all <- naive_bayes(Churn ~ ., data = train)
pred_all <- predict(nb_all, test, type = "prob")
pred_all1 <- predict(nb_all, test)
roc_all <- roc(response = test$Churn, predictor = pred_all[,2])
plot(roc_all, col = "blue", main = "All Three")
(roc_all[["auc"]])

table_all <- table(test$Churn, pred_all1)
(cm_all <- confusionMatrix(table_all))

##Pairs to see if any better than all three##

#Band + Month 
nb_bm <- naive_bayes(Churn ~ Bandwidth_GB_Year + MonthlyCharge, data = train)
pred_bm <- predict(nb_bm, test, type = "prob")
pred_bm1 <- predict(nb_bm, test)
roc_bm <- roc(response = test$Churn, predictor = pred_bm[,2])
plot(roc_bm, col = "blue", main = "Bandwidth & Monthly Charge")
(roc_bm[["auc"]])

table_bm <- table(test$Churn, pred_bm1)
(cm_bm <- confusionMatrix(table_bm))

#Band + Con 
nb_bc <- naive_bayes(Churn ~ Bandwidth_GB_Year + Contract, data = train)
pred_bc <- predict(nb_bc, test, type = "prob")
pred_bc1 <- predict(nb_bc, test)
roc_bc <- roc(response = test$Churn, predictor = pred_bc[,2])
plot(roc_bc, col = "blue", main = "Bandwidth & Contract Type")
(roc_bc[["auc"]])

table_bc <- table(test$Churn, pred_bc1)
(cm_bc <- confusionMatrix(table_bc))

#Month + Con 
nb_mc <- naive_bayes(Churn ~ MonthlyCharge + Contract, data = train)
pred_mc <- predict(nb_mc, test, type = "prob")
pred_mc1 <- predict(nb_mc, test)
roc_mc <- roc(response = test$Churn, predictor = pred_mc[,2])
plot(roc_mc, col = "blue", main = "Monthly Charge & Contract Type")
(roc_mc[["auc"]])

table_mc <- table(test$Churn, pred_mc1)
(cm_mc <- confusionMatrix(table_mc))

write_xlsx(train, "Train_NB.xlsx")
write_xlsx(test, "Test_NB.xlsx")
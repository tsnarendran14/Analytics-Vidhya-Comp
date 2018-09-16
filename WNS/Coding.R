library(tidyverse)
library(dummies)

train <- read_csv("D:/AV/WNS/train_LZdllcl.csv")
test <- read_csv("D:/AV/WNS/test_2umaH9m.csv")
submission <- read_csv("D:/AV/WNS/sample_submission_M0L0uXE.csv")

train_target <- train$is_promoted
train <- train[, -c(which(names(train) == "is_promoted"))]

train$type <- "train"
test$type <- "test"

data <- bind_rows(train, test)

data$education_encoded <- ifelse(data$education == "Bachelor's", 1, 
                                   ifelse(data$education == "Master's & above", 2,
                                   ifelse(data$education == "Below Secondary", 0, NA)))

data <- cbind(data, dummy(data$department, sep = "_"))
data <- cbind(data, dummy(data$region, sep = "_"))
#data <- cbind(data, dummy(data$education, sep = "_"))
#data <- cbind(data, dummy(data$gender, sep = "_"))
data$gender_imputed <- if_else(data$gender == "m", 1, 0)
data <- cbind(data, dummy(data$recruitment_channel, sep = "_"))


# Correlation Check
df_cor <- (as.data.frame(cor(data), rownames = dimnames(cor(data))[[1]]))

# Age and length_of_service highly correlated


# Dropping Columns
data <- data[,-c(which(names(data) == "department"), which(names(data) == "region"),
                 which(names(data) == "education"), which(names(data) == "gender"),
                 which(names(data) == "recruitment_channel"), which(names(data) == "data_Sales & Marketing"),
                 which(names(data) == "data_region_1"), which(names(data) == "data_other"),
                 which(names(data) == "length_of_service"))]

colSums(is.na(data))
#View(data_frame(colSums(is.na(data))))


##### Imputation #####
library(Amelia)

imputed_Data <-  amelia(data, m=5, idvars = c("employee_id", "type"),parallel = "multicore", noms = "education_encoded")


#View(imputed_Data$imputations[[1]])

imputed_Data_1 <- imputed_Data$imputations[[1]]
imputed_Data_1 <- cbind(imputed_Data_1, dummy(imputed_Data_1$education_encoded, sep = "_"))
imputed_Data_1 <- imputed_Data_1[,-c(which(names(imputed_Data_1) == "imputed_Data_1_0"))]
imputed_Data_1 <- imputed_Data_1[,-c(which(names(imputed_Data_1) == "employee_id"))]
imputed_Data_1 <- imputed_Data_1[,-c(which(names(imputed_Data_1) == "education_encoded"))]

imputed_Data_1$no_of_trainings <- (imputed_Data_1$no_of_trainings - min(imputed_Data_1$no_of_trainings)) / (max(imputed_Data_1$no_of_trainings) - min(imputed_Data_1$no_of_trainings))
imputed_Data_1$age <- (imputed_Data_1$age - min(imputed_Data_1$age)) / (max(imputed_Data_1$age) - min(imputed_Data_1$age))
imputed_Data_1$previous_year_rating <- (imputed_Data_1$previous_year_rating - min(imputed_Data_1$previous_year_rating)) / (max(imputed_Data_1$previous_year_rating) - min(imputed_Data_1$previous_year_rating))
imputed_Data_1$avg_training_score <- (imputed_Data_1$avg_training_score - min(imputed_Data_1$avg_training_score)) / (max(imputed_Data_1$avg_training_score) - min(imputed_Data_1$avg_training_score))


###### Modelling #################
library(caTools)

train <- imputed_Data_1[imputed_Data_1$type == "train", ]
test <- imputed_Data_1[imputed_Data_1$type == "test", ]

train <- train[, -c(which(names(train) == "type"))]
test <- test[, -c(which(names(test) == "type"))]

train_target <- as_data_frame(train_target)
names(train_target)[1] <- "is_promoted"

train <- bind_cols(train, train_target)

# Logistic Regression

model <- glm (is_promoted ~ .-employee_id, data = train, family = binomial)
summary(model)

prediction <- predict(model, type = 'response', newdata = test)

submission$is_promoted <- prediction

colSums(is.na(submission))

write_csv(submission, "Submission_logistic_v1.csv")

###################### Random Forest ##########

train <- train[, -c(which(names(train)=="employee_id"))]
test <- test[, -c(which(names(test)=="employee_id"))]


train_target <- as_data_frame(train_target)
names(train_target)[1] <- "is_promoted"

train <- bind_cols(train, train_target)


train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)

X <- c(1:52)
y <- 53

mdl_rf <- h2o.randomForest(X, y, training_frame = train_h2o, ntrees = 1000, max_depth = 30)

prediction_rf <- h2o.predict(mdl_rf, newdata = test_h2o)

prediction_rf <- as.vector(prediction_rf)

submission$is_promoted <- prediction_rf

submission$is_promoted <- if_else(submission$is_promoted > 0.5, 1, 0)

write_csv(submission, "Submission_RF_v10.csv")

############### GBM

train <- train[, -c(which(names(train)=="type"))]
test <- test[, -c(which(names(test)=="type"))]



train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)

X <- c(1:51)
y <- 52

mdl_gbm <- h2o.gbm(X, y, training_frame = train_h2o, ntrees = 1000, max_depth = 20, learn_rate = 0.01)

prediction_gbm <- h2o.predict(mdl_gbm, newdata = test_h2o)

prediction_gbm <- as.vector(prediction_gbm)

submission$is_promoted <- prediction_gbm

submission$is_promoted <- if_else(submission$is_promoted >= 0.5, 1, 0)

write_csv(submission, "Submission_GBM_v3.csv")

############### SVM V2 ################################
library(e1071)

train <- train[, -c(which(names(train)=="employee_id"))]
test <- test[, -c(which(names(test)=="employee_id"))]


mdl_svm <- svm(is_promoted ~ ., train)

prediction_svm <- predict(mdl_svm, test)

submission$is_promoted <- prediction_svm

submission$is_promoted <- if_else(submission$is_promoted >= 0.5, 1, 0)

write_csv(submission, "Submission_svm_v1.csv")


############### PCA

str(train)

prcomp(train)

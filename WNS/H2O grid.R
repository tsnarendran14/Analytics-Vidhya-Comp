library(tidyverse)
library(dummies)

train <- read_csv("C:/AV/WNS/train_LZdllcl.csv")
test <- read_csv("C:/AV/WNS/test_2umaH9m.csv")
submission <- read_csv("C:/AV/WNS/sample_submission_M0L0uXE.csv")

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

imputed_Data_1 <- imputed_Data$imputations[[3]]
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

############## GBM Grid search 
library(h2o)

h2o.init(max_mem_size = "12g")

train$is_promoted <- as.factor(train$is_promoted)

train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)

X <- c(1:52)
y <- 53

ss <- h2o.splitFrame(train_h2o, seed = 1)
train_h2o <- ss[[1]]
valid_h2o <- ss[[2]]

train_h2o[,y] <- as.factor(train_h2o[,y])
valid_h2o[,y] <- as.factor(valid_h2o[,y])

gbm_params1 <- list(min_rows = c(1, 2),
                    max_depth = c(3, 5, 9,15,20),
                    col_sample_rate_per_tree = c(0.2, 0.5, 1.0))

gbm_grid1 <- h2o.grid("gbm", x = X, y = y,
                      grid_id = "gbm_grid1",
                      training_frame = train_h2o,
                      validation_frame = valid_h2o,
                      ntrees = 1000,
                      seed = 1,
                      stopping_metric = "misclassification",
                      distribution = "bernoulli",
                      hyper_params = gbm_params1
                      )

gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1",
                             sort_by = "f1",
                             decreasing = TRUE)
print(gbm_gridperf1)


best_gbm1 <- h2o.getModel(gbm_gridperf1@model_ids[[1]])

best_gbm_perf1 <- h2o.performance(model = best_gbm1,
                                  newdata = test_h2o)


##### auto ml
train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)



X <- c(1:52)
y <- 53

train_h2o[,y] <- as.factor(train_h2o[,y])

aml <- h2o.automl(x = X, y = y,
                  training_frame = train_h2o,
                  max_runtime_secs = 7200
                  )

lb <- aml@leaderboard
lb

aml

pred <- h2o.predict(aml, test_h2o) 

submission$is_promoted <- as.vector(pred$predict)

write_csv(submission, "Submission_Auto_ML_V4.csv")

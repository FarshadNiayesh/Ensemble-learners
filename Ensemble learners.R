library(caret)
library(RWeka)
library(e1071)
library(randomForest)
library(adabag)
library(plyr)
library(nnet)
library(LiblineaR)
library(LogicReg)
library(rpart)
library(fastAdaboost)

set.seed(1234)
train_data <- read.csv('/Users/farshad/Downloads/lab3-train.csv')
test_data <- read.csv('/Users/farshad/Downloads/lab3-test.csv')
train_data$Class <- as.factor(train_data$Class)
test_data$Class <- as.factor(test_data$Class)
####################### TASK 1 #######################
RF_grid <- expand.grid(mtry = (1:5))
model_RF <- train(Class~., data = train_data, method = 'rf', tuneGrid = RF_grid)
prediction_RF <- predict(model_RF, test_data)
confusionMatrix(prediction_RF, test_data$Class)

model_AB <- boosting(Class~., data = train_data, maxdepth = 1, mfinal = 10)
prediction_AB <- predict.boosting(model_AB, test_data)
confusionMatrix(prediction_AB$class, test_data$Class)

####################### TASK 2 #######################
NN_grid <- expand.grid(size = c(32,64,128) , decay = c(1e-3,1e-4,1e-5))
model_NN <- train(Class~., data = train_data, method = 'nnet', tuneGrid = NN_grid)
prediction_NN <- predict(model_NN, test_data)
confusionMatrix(prediction_NN, test_data$Class)

#using square root of the training instances as the optimal number for K
KNN_grid <- expand.grid(k = (1:21))
model_KNN <- train(Class~., data = train_data, method = 'knn', tuneGrid = KNN_grid)
prediction_KNN <- predict(model_KNN, test_data)
confusionMatrix(prediction_KNN, test_data$Class)

model_LR <- train(Class~., data = train_data, method = 'glm',family = 'binomial')
prediction_LR <- predict(model_LR, test_data)
confusionMatrix(prediction_LR, test_data$Class)

NB_grid <- expand.grid(fL=c(0:5), usekernel = FALSE, adjust=1)
model_NB <- train(Class~., data = train_data, method = 'nb', tuneGrid = NB_grid)
prediction_NB <- predict(model_NB, test_data)
confusionMatrix(prediction_NB, test_data$Class)

DT_grid <- expand.grid(C=(1:3)*0.1, M = (1:5))
model_DT <- train(Class~., data = train_data, method = 'J48', tuneGrid = DT_grid)
prediction_DT <- predict(model_DT, test_data)
confusionMatrix(prediction_DT, test_data$Class)

####################### UNWEIGHTED ENSEMBLE LEARNING ####################### 
ensemble.df <- data.frame(nn = as.numeric(prediction_NN), 
                          knn = as.numeric(prediction_KNN), 
                          LR = as.numeric(prediction_LR),
                          nb = as.numeric(prediction_NB),
                          dt = as.numeric(prediction_DT))
ensemble.df$sum <- rowSums(ensemble.df)
ensemble.prediction <- ifelse(ensemble.df$sum <= 7,0,1)
confusionMatrix(ensemble.prediction, test_data$Class)

####################### WEIGHTED ENSEMBLE LEARNING ####################### 
ensemble2.df <- data.frame(nn = as.numeric(prediction_NN), 
                          knn = as.numeric(prediction_KNN), 
                          LR = as.numeric(prediction_LR),
                          nb = as.numeric(prediction_NB),
                          dt = as.numeric(prediction_DT))
#using the training accuracy as the weights
weights <- c(0.7553781, 0.7205489, 0.7474009, 0.7328376, 0.7455507)

ensemble2.df <- ensemble2.df * weights
ensemble2.df$weighed.average <- rowMeans(ensemble2.df)

upperbound <- mean(weights * 2)
lowerbound <- mean(weights * 1)
midpoint <- (lowerbound + upperbound)/2

ensemble2.prediction <- ifelse(ensemble2.df$weighed.average <= midpoint,0,1)
confusionMatrix(ensemble2.prediction, test_data$Class)


####################### UNWEIGHTED ENSEMBLE LEARNING ####################### 
ensemble3.df <- data.frame(rf = as.numeric(prediction_RF),
                          ab = as.numeric(as.factor(prediction_AB$class)),
                          nn = as.numeric(prediction_NN), 
                          knn = as.numeric(prediction_KNN), 
                          LR = as.numeric(prediction_LR),
                          nb = as.numeric(prediction_NB),
                          dt = as.numeric(prediction_DT))
ensemble3.df$sum <- rowSums(ensemble3.df)
ensemble3.prediction <- ifelse(ensemble3.df$sum <= 7,0,1)
confusionMatrix(ensemble3.prediction, test_data$Class)


ensemble4.df <- data.frame(rf = as.numeric(prediction_RF),
                           ab = as.numeric(as.factor(prediction_AB$class)),
                           nn = as.numeric(prediction_NN), 
                           knn = as.numeric(prediction_KNN), 
                           LR = as.numeric(prediction_LR),
                           nb = as.numeric(prediction_NB),
                           dt = as.numeric(prediction_DT))
#using the training accuracy as the weights
weights2 <- c(0.7341828, 0.7245373, 0.7553781, 0.7205489, 0.7474009, 0.7328376, 0.7455507)

ensemble4.df <- ensemble4.df * weights
ensemble4.df$weighed.average <- rowMeans(ensemble4.df)

upperbound2 <- mean(weights2 * 2)
lowerbound2 <- mean(weights2 * 1)
midpoint2 <- (lowerbound2 + upperbound2)/2

ensemble4.prediction <- ifelse(ensemble4.df$weighed.average <= midpoint2,0,1)
confusionMatrix(ensemble4.prediction, test_data$Class)


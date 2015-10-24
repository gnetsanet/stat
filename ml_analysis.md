---
title: "Machine Learning - Practical"
author: "Netsanet Gebremedhin"
date: "October 21, 2015"
output: html_document
keep_md: true
---

##  Introduction

## Data

+ Source: [source](http://groupware.les.inf.puc-rio.br/har)
+ Training data: [Train](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
+ Test data: [Test](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

### Data acquisition and clean up

Download training and testing datasets.

```{r eval=FALSE, cache=TRUE}
trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(trainurl, method='curl', destfile='trainingfile')

testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(testurl, method='curl', destfile='testfile')
```
No of observations and features in the training and test data

```{r cache=TRUE, echo=TRUE, eval=TRUE}
train<-read.csv('trainingfile' ,na.strings=c("","NA", "#DIV/0!"))
dim(train)
test<-read.csv('testfile' ,na.strings=c("","NA", "#DIV/0!"))
dim(test)
```
Discard features with all missing values

There are 60 such columns 
```{r echo=TRUE, eval=TRUE}
dim(train[,colSums(is.na(train))==0])
which(colSums(is.na(train))==0)
train<-train[,which(colSums(is.na(train))==0)]

dim(test[,colSums(is.na(test))==0])
which(colSums(is.na(test))==0)
test<-test[,which(colSums(is.na(test))==0)]
```

Remove the following features/columns that are irrelevant and hence non-predictive

```{r echo=TRUE, eval=TRUE}
colnames(train[,1:7])
train<-train[,-c(1:7)]
dim(train)
test<-test[,-c(1:7)]
dim(test)
```
## Preprocessing

### Further partitioning the training data into training and validation sets
Partition the data using random subsampling without replacement

```{r warning=FALSE,eval=TRUE}
library(caret)
set.seed(1413)
trainset <- createDataPartition(y=train$classe, p=0.75, list=FALSE)
trainingSet <- train[trainset,]
validationSet <- train[-trainset,]
```

```{r echo=TRUE, eval=TRUE}
dim(trainingSet)
dim(validationSet)
```
Visualizing the distribution of the outcome variable

```{r echo=TRUE, eval=TRUE}
plot(trainingSet$classe, col="coral2", main="Frequency of Unilateral Dumbbell Biceps Curl Fashion", xlab="Unilateral Dumbbell Biceps Curl Fashions", ylab="Frequency")
```
### Prediction using Random Forest
Train model on the training set and check the performance of the model by making predictions on the validation set.

```{r, eval=TRUE}
library(randomForest)
model <- randomForest(classe ~. , data=trainingSet, method="class")
model_prediction1 <- predict(model, trainingSet, type = "class")
```

Not surprising the model performs well on the dataset it was trained on and accurately predicts the outcome class

```{r echo=TRUE, eval=TRUE}
confusionMatrix(model_prediction1, trainingSet$classe)
```

## Model Evaluation

Evaluate model's performance on the held out dataset

```{r echo=TRUE, eval=TRUE}
model_prediction2 <- predict(model, validationSet, type = "class")
confusionMatrix(model_prediction2, validationSet$classe)
```

Model performs well on the cross-validation dataset with 99.53% accuracy and out-of-sample error rate of 0.47%

## Prediction on the test dataset

Make class prediction on  unlabelled dataset

```{r echo=TRUE, eval=TRUE}
model_prediction2 <- predict(model, test, type = "class")
model_prediction2
```

## Submission

Prep files for submission. Write the class predicted for each observation in the test dataset into a file.

```{r cache=TRUE}
write_class_predictions_to_file = function(x) {
     for (k in 1:length(x)) {
         file_name = paste0("problem_id_",k,".txt")
         write.table(x[k],file=file_name, quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

write_class_predictions_to_file(model_prediction2)
```
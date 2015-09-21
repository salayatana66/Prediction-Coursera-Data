# Prediction
Andrea Schioppa  
We build a Machine Learning algorithm to predict
how well participants in a weight lifting experiment did. The data is available from the website (http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). We assume that the data is downloaded in the current directory in the file 'pml-training.csv'.

#Cleaning the Data

We load required libraries.

```r
library(ggplot2);
library(caret);
```

We load the training data.


```r
data.train <- read.csv('pml-training.csv', header=TRUE);
```

Remove first column which is just the row number. Some predictors consist of a large
fraction of `NA`s and so we remove them.


```r
data.train <- data.train[,-1];

na.predictors <- c();

for(i in 1:length(names(data.train))) {
    if(mean(is.na(data.train[,i]))>0.90)
        na.predictors <- c(na.predictors,i);
}

data.train <- data.train[,-na.predictors];
```

The predictor `cvtd_timestamp`, is a kind of time stamp which has been read as a character,
and we convert it to a numeric format using the `POSIXct` standard.


```r
data.train$cvtd_timestamp <- strptime(data.train$cvtd_timestamp, format='%d/%m/%Y %H:%M');
data.train$cvtd_timestamp <- as.numeric(as.POSIXct(data.train$cvtd_timestamp));
```

There are also other predictors which contain many missing values. They were not
eliminated before because they show as factors; we now remove them:


```r
na.predictors <- c();

for(i in 2:(length(names(data.train))-1)) {
    if(class(data.train[,i])=='factor') {
    if(mean(as.character(data.train[,i])=="")>0.90) {
        na.predictors <- c(na.predictors,i);
    }
   }
}

data.train <- data.train[,-na.predictors];
```

We finally convert to `numeric` predictors that were read as `integers`.


```r
for(i in 1:length(names(data.train))) {
    if(class(data.train[,i])=='integer') {
        data.train[,i] <- as.numeric(data.train[,i]);
    }
}
```

# Further Choice of Predictors

The cleaned data frame `data.train` contains now 59 columns, 58 of which are predictors, and the last
one is the variable `classe` which has to be predicted. We now split the data set into a train and test
part. Note that as we are going to fit a Random Forest, this is not necessary, but we do this to
illustrate the model


```r
set.seed(125);
library(caret);

split.indx <- createDataPartition(data.train$classe, p = 0.8, list = FALSE);
data.test <- data.train[-split.indx, ]
data.train <- data.train[split.indx, ]
```

The variables `cvtd_timestamp` and `raw_timestamp_part_1` are almost perfectly correlated.


```r
cor(data.train$cvtd_timestamp, data.train$raw_timestamp_part_1)
```

```
## [1] 1
```


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
timestamps <- data.frame(cvtd = data.train$cvtd_timestamp, rw1 = data.train$raw_timestamp_part_1);
preproc.tsmp <- preProcess(timestamps, method = c('center', 'scale'));
timestamps <- predict(preproc.tsmp, timestamps)
with(timestamps, plot(cvtd, rw1));
```

![](project_files/figure-html/unnamed-chunk-1-1.png) 

Also a plot shows that `raw_timestamp_part_2` is rather homogeneous among the different classes.


```r
tmsp.plot <- ggplot(data.train, aes(x=raw_timestamp_part_2, fill=classe, y = ..count..));
tmsp.plot <- tmsp.plot
tmsp.plot + geom_histogram() + facet_wrap(~classe);
```

![](project_files/figure-html/unnamed-chunk-2-1.png) 

Overall, the timestamps are variables which to do not seem to have predictive value, so we
remove them.


```r
toremove <- c('raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp');
data.train <- data.train[,-which(names(data.train) %in% toremove)];
data.test <- data.test[,-which(names(data.test) %in% toremove)];
```

# Model Choice and Training

We now fit a Random Forest. We have also tried
other models using `method=lda`, `method=gbm`, `method=svmLinear`, but the Random Forest seems to give the best results. 
Note that, as we use a Random Forest, there is no need for explicitly invoking cross-validation
via setting `TrControl` in this model. 
As it takes some hours to train this model, cross-validation would also make the code slower.
As Fitting the model is time-consuming, we save the model to an external file.

```r
mod.rf <- train(classe ~ ., method = 'rf', prox = TRUE, data = data.train)
save(mod.rf, file='model.RData')
```

We summarize the model.


```r
load('model.RData')
mod.rf
```

```
## Random Forest 
## 
## 15699 samples
##    55 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 15699, 15699, 15699, 15699, 15699, 15699, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9924559  0.9904552  0.0015192881  0.0019218990
##   30    0.9967060  0.9958328  0.0006322107  0.0007999285
##   59    0.9944491  0.9929773  0.0015469643  0.0019587288
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 30.
```

The following plot shows how the model used accuracy to select the optimal number of trees.


```r
plot(mod.rf, main='Accuracy vs. Number or trees')
```

![](project_files/figure-html/unnamed-chunk-4-1.png) 

We compute the confusion Matrix to show Accuracy and Kappa on the test set to compare with the one predicted from the model.


```r
library(caret)
pred.test <- predict(mod.rf, newdata = data.test)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
tab <- confusionMatrix(data.test$classe, pred.test)
tab
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    4  754    1    0    0
##          C    0    1  683    0    0
##          D    0    0    0  643    0
##          E    0    2    0    0  719
## 
## Overall Statistics
##                                          
##                Accuracy : 0.998          
##                  95% CI : (0.996, 0.9991)
##     No Information Rate : 0.2855         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9974         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9960   0.9985   1.0000   1.0000
## Specificity            1.0000   0.9984   0.9997   1.0000   0.9994
## Pos Pred Value         1.0000   0.9934   0.9985   1.0000   0.9972
## Neg Pred Value         0.9986   0.9991   0.9997   1.0000   1.0000
## Prevalence             0.2855   0.1930   0.1744   0.1639   0.1833
## Detection Rate         0.2845   0.1922   0.1741   0.1639   0.1833
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9982   0.9972   0.9991   1.0000   0.9997
```

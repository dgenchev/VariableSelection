# Variable reduction with Boruta
# https://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/
# 2016-11-06

# Load package
install.packages("Boruta")
library("Boruta")

# Set dir and read data
setwd("../Data/Loan_Prediction")
traindata <- read.csv("train.csv", header = T, stringsAsFactors = F)
str(traindata)
names(traindata) <- gsub("_", "", names(traindata))

# Convert categorical into factor
convert <- c(2:6, 11:13)
traindata[,convert] <- data.frame(apply(traindata[convert], 2, as.factor))

# Use Boruta
set.seed(123)
boruta.train <- Boruta(LoanStatus~.-LoanID, data = traindata, doTrace = 2)
print(boruta.train)

# Plot results
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i) boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels), at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

# Decide on tentative attributes 
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

# Get a list of the confirmed attributes
getSelectedAttributes(final.boruta, withTentative = F)

# Create a dataframe of the final Boruta output
boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)


## Compare Boruta to Recursive Feature Elimination (RFE) via caret
library(caret)
library(randomForest)
set.seed(123)
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
rfe.train <- rfe(traindata[,2:12], traindata[,13], sizes=1:12, rfeControl=control)
rfe.train
plot(rfe.train, type=c("g", "o"), cex = 1.0, col = 1:11)
predictors(rfe.train)
# RFE selects only CreditHistory as important among the 11 features
# Boruta returns much better results
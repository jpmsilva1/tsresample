# Rscript fit_others.R <dir> <pattern> <learner: rpart|rf|arima> <par1> <par2>
suppressPackageStartupMessages({library(rpart); library(randomForest); library(forecast)})
a <- commandArgs(trailingOnly=TRUE); dir <- a[1]; pat <- a[2]; learner <- a[3]
p1 <- if (length(a) >= 4) as.numeric(a[4]) else NA; p2 <- if (length(a) >= 5) as.numeric(a[5]) else NA
files <- list.files(dir, pattern=paste0("^", pat, ".*_train\\.csv$"))
set.seed(1234)
for (f in files) {
  name <- sub("_train\\.csv$", "", f)
  tr <- read.csv(file.path(dir, f)); te <- read.csv(file.path(dir, paste0(name, "_test.csv")))
  extra <- ""
  if (learner == "rpart") {
    m <- rpart(V10 ~ ., tr, control=rpart.control(minsplit=p1, cp=p2)); p <- predict(m, te)
    extra <- sprintf("nodes=%d", sum(m$frame$var == "<leaf>"))
  } else if (learner == "rf") {
    m <- randomForest(V10 ~ ., tr, mtry=p1, ntree=p2); p <- predict(m, te)
  } else if (learner == "arima") {
    trainY <- tr$V10; trues <- te$V10
    m <- auto.arima(trainY); data <- c(trainY, trues)
    p <- fitted(Arima(data, model=m))[(length(trainY)+1):length(data)]
    extra <- paste0("order=", paste(arimaorder(m), collapse=","), " drift=", as.integer("drift" %in% names(coef(m))), " mean=", as.integer("intercept" %in% names(coef(m)) || "mean" %in% names(coef(m))))
  }
  write.csv(data.frame(pred=as.vector(p)), file.path(dir, paste0(name, "_", learner, "_pred.csv")), row.names=FALSE)
  cat(name, learner, extra, "\n")
}

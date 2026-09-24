# Runs the ORIGINAL workflows of Exps.R (functions sourced unchanged) on one data set with the
# parameters of Table 7 of the article, and writes the per-iteration scores of every workflow.
#   Rscript run_exps.R <dataset> <nreps> <workflow filter or 'all'> <outdir>
args <- commandArgs(trailingOnly = TRUE)
i <- as.integer(args[1]); nreps <- as.integer(args[2]); filt <- args[3]; outdir <- args[4]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages({library(lubridate); library(xts); library(performanceEstimation); library(uba); library(UBL); library(DMwR)})
here <- normalizePath(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])))
# original R code and data: in the same repository (R_Code/ next to R_replication/) or in the
# sibling clone ../TSResampStrat_JDSA2017 when the port is used as a separate repository
orig <- if (dir.exists(file.path(here, "..", "R_Code"))) file.path(here, "..") else file.path(here, "..", "..", "TSResampStrat_JDSA2017")
rcode <- file.path(orig, "R_Code", "Exps.R")
# lines 2..2475 of Exps.R = library() calls + every function definition (create.data, eval.stats,
# the 52 mc.* workflows, the 9 resampling functions); line 1 is setwd(""), 2479+ is the driver
eval(parse(text = readLines(rcode)[2:2475]))
load(file.path(orig, "Data", "data_NM_PB_LT_DSAA2016.Rdata"))

ds <- create.data(data[[i]], 10)
if (any(is.na(ds))) { cat("knnImputation on", sum(!complete.cases(ds)), "rows\n"); ds <- knnImputation(ds) }
form <- as.formula(V10 ~ .)

# Table 7 of the article: svm cost/gamma, mars nk/degree/thresh, rf mtry/ntree, rpart minsplit/cp
T7 <- list(
  c(300,0.01,17,1,0.001,5,1500,10,0.01),  c(300,0.01,17,2,0.001,7,750,10,0.001),  c(300,0.01,17,1,0.001,7,500,10,0.001),
  c(150,0.01,10,1,0.001,7,750,10,0.1),    c(300,0.001,10,2,0.001,7,750,20,0.001), c(300,0.01,17,2,0.001,5,500,10,0.001),
  c(300,0.01,10,1,0.001,7,750,30,0.001),  c(300,0.01,17,2,0.001,7,750,30,0.001),  c(10,0.01,10,2,0.001,5,750,30,0.001),
  c(300,0.01,17,2,0.001,7,500,10,0.001),  c(10,0.01,17,1,0.001,7,500,20,0.001),   c(300,0.01,17,1,0.001,7,750,10,0.001),
  c(150,0.01,17,2,0.001,7,750,10,0.001),  c(150,0.01,17,2,0.001,7,1500,10,0.001), c(300,0.01,17,2,0.001,5,1500,10,0.001),
  c(300,0.01,17,2,0.001,7,750,10,0.001),  c(300,0.01,17,2,0.001,7,500,10,0.001),  c(300,0.01,17,2,0.001,5,500,10,0.001),
  c(150,0.01,17,1,0.01,5,500,10,0.001),   c(300,0.01,17,2,0.001,7,500,10,0.001),  c(150,0.001,17,2,0.001,7,500,10,0.001),
  c(150,0.001,10,2,0.001,7,500,10,0.001), c(10,0.001,10,1,0.001,5,500,10,0.001),  c(150,0.01,17,1,0.001,7,750,10,0.001))
p <- T7[[i]]
cost <- p[1]; gamma <- p[2]; nk <- p[3]; degree <- p[4]; thresh <- p[5]; mtry <- p[6]; ntree <- p[7]; minsplit <- p[8]; cp <- p[9]
sz <- if (i %in% c(21, 22)) c(.1, .05) else if (i %in% c(23, 24)) c(.2, .1) else c(.5, .25)

strat <- c("", "_UNDERB", "_UNDERT", "_UNDERTPhi", "_OVERB", "_OVERT", "_OVERTPhi", "_SMOTEB", "_SMOTET", "_SMOTETPhi")
wfs <- list()
for (s in strat) wfs[[paste0("mc.lm", s)]] <- Workflow(paste0("mc.lm", s))
for (s in strat) wfs[[paste0("mc.svm", s)]] <- Workflow(paste0("mc.svm", s), cost = cost, gamma = gamma)
for (s in strat) wfs[[paste0("mc.mars", s)]] <- Workflow(paste0("mc.mars", s), nk = nk, degree = degree, thresh = thresh)
for (s in strat) wfs[[paste0("mc.rf", s)]] <- Workflow(paste0("mc.rf", s), mtry = mtry, ntree = ntree)
for (s in strat) wfs[[paste0("mc.rpart", s)]] <- Workflow(paste0("mc.rpart", s), minsplit = minsplit, cp = cp)
wfs[["mc.arima"]] <- Workflow("mc.arima"); wfs[["mc.BDES"]] <- Workflow("mc.BDES")
if (filt != "all") {
  keep <- unlist(lapply(strsplit(filt, ",")[[1]], function(f) grep(f, names(wfs), fixed = TRUE, value = TRUE)))
  wfs <- wfs[unique(keep)]
}
cat(sprintf("DS%d: %d rows, %d workflows, nReps=%d, sizes %s/%s, params cost=%s gamma=%s nk=%s degree=%s thresh=%s mtry=%s ntree=%s minsplit=%s cp=%s\n",
            i, nrow(ds), length(wfs), nreps, sz[1], sz[2], cost, gamma, nk, degree, thresh, mtry, ntree, minsplit, cp))
et <- EstimationTask("totTime", method = MonteCarlo(nReps = nreps, szTrain = sz[1], szTest = sz[2]))
summary <- data.frame()
for (nm in names(wfs)) {
  out <- file.path(outdir, sprintf("ds%d__%s.csv", i, nm))
  if (file.exists(out)) { cat("[skip]", nm, "\n"); next }
  t0 <- proc.time()[3]
  res <- tryCatch(suppressWarnings(performanceEstimation(PredTask(form, ds), wfs[[nm]], et)), error = function(e) e)
  if (inherits(res, "error")) {
    cat(sprintf("[FAIL] %s: %s\n", nm, conditionMessage(res)))
    writeLines(conditionMessage(res), sub("\\.csv$", ".err", out)); next
  }
  sc <- t(sapply(1:nreps, function(it) getIterationsInfo(res, workflow = 1, task = 1, it = it)$evaluation))
  colnames(sc) <- c("prec", "rec", "F1"); sc <- as.data.frame(sc); sc$it <- 1:nreps
  sc$start <- sapply(1:nreps, function(it) min(getIterationsInfo(res, workflow = 1, task = 1, it = it)$train) + nrow(ds) * 0)  # first train row
  write.csv(sc, out, row.names = FALSE)
  cat(sprintf("[done] %-18s prec=%.3f rec=%.3f F1=%.3f  (%.0fs)\n", nm, mean(sc$prec), mean(sc$rec), mean(sc$F1), proc.time()[3] - t0))
}

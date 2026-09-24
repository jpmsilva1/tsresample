# Table 6 of the article with the ORIGINAL R code: SVM workflows of OptParmsSearch.R (un / ov
# percentages) with the parameters of Tables 7 and 8, data sets 4, 10, 12, 50 repetitions.
#   Rscript run_table6.R <dataset> <nreps> <outdir>
args <- commandArgs(trailingOnly = TRUE)
i <- as.integer(args[1]); nreps <- as.integer(args[2]); outdir <- args[3]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
suppressPackageStartupMessages({library(lubridate); library(xts); library(performanceEstimation); library(uba); library(UBL); library(DMwR); library(e1071)})
here <- normalizePath(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])))
# original R code and data: in the same repository (R_Code/ next to R_replication/) or in the
# sibling clone ../TSResampStrat_JDSA2017 when the port is used as a separate repository
orig <- if (dir.exists(file.path(here, "..", "R_Code"))) file.path(here, "..") else file.path(here, "..", "..", "TSResampStrat_JDSA2017")
rdir <- file.path(orig, "R_Code")
eval(parse(text = readLines(file.path(rdir, "Exps.R"))[2:2475]))            # resamplers, eval.stats, create.data
eval(parse(text = readLines(file.path(rdir, "OptParmsSearch.R"))[62:217]))   # mc.svm* with un/ov (override)
load(file.path(orig, "Data", "data_NM_PB_LT_DSAA2016.Rdata"))
ds <- create.data(data[[i]], 10)
if (any(is.na(ds))) ds <- knnImputation(ds)
form <- as.formula(V10 ~ .)
base <- list(`4` = c(150, 0.01), `10` = c(300, 0.01), `12` = c(300, 0.01))[[as.character(i)]]
T8 <- list(
  `4`  = list(UNDERB = c(10,0.01,.4), UNDERT = c(10,0.01,.4), UNDERTPhi = c(10,0.01,.8), OVERB = c(10,0.001,5), OVERT = c(150,0.001,2), OVERTPhi = c(150,0.01,2),
              SMOTEB = c(150,0.001,.8,2), SMOTET = c(150,0.001,.6,2), SMOTETPhi = c(10,0.001,.8,2)),
  `10` = list(UNDERB = c(10,0.001,.1), UNDERT = c(150,0.001,.1), UNDERTPhi = c(300,0.001,.1), OVERB = c(10,0.001,2), OVERT = c(150,0.001,2), OVERTPhi = c(150,0.001,2),
              SMOTEB = c(10,0.001,.8,10), SMOTET = c(10,0.001,.6,5), SMOTETPhi = c(300,0.001,.6,3)),
  `12` = list(UNDERB = c(150,0.001,.2), UNDERT = c(300,0.001,.2), UNDERTPhi = c(150,0.001,.2), OVERB = c(10,0.001,10), OVERT = c(150,0.001,3), OVERTPhi = c(150,0.001,3),
              SMOTEB = c(10,0.001,.2,3), SMOTET = c(10,0.001,.8,5), SMOTETPhi = c(150,0.001,.4,2)))[[as.character(i)]]
wfs <- list(mc.svm = Workflow("mc.svm", cost = base[1], gamma = base[2]))
for (k in names(T8)) {
  v <- T8[[k]]; nm <- paste0("mc.svm_", k)
  wfs[[nm]] <- if (startsWith(k, "UNDER")) Workflow(nm, cost = v[1], gamma = v[2], un = v[3]) else
               if (startsWith(k, "OVER"))  Workflow(nm, cost = v[1], gamma = v[2], ov = v[3]) else
                                           Workflow(nm, cost = v[1], gamma = v[2], un = v[3], ov = v[4])
}
et <- EstimationTask("totTime", method = MonteCarlo(nReps = nreps, szTrain = .5, szTest = .25))
for (nm in names(wfs)) {
  out <- file.path(outdir, sprintf("table6_ds%d__%s.csv", i, nm))
  if (file.exists(out)) { cat("[skip]", nm, "\n"); next }
  res <- tryCatch(suppressWarnings(performanceEstimation(PredTask(form, ds), wfs[[nm]], et)), error = function(e) e)
  if (inherits(res, "error")) { cat(sprintf("[FAIL] %s: %s\n", nm, conditionMessage(res))); writeLines(conditionMessage(res), sub("\\.csv$", ".err", out)); next }
  sc <- t(sapply(1:nreps, function(it) getIterationsInfo(res, workflow = 1, task = 1, it = it)$evaluation))
  colnames(sc) <- c("prec", "rec", "F1"); write.csv(as.data.frame(sc), out, row.names = FALSE)
  cat(sprintf("[done] %-18s prec=%.3f rec=%.3f F1=%.3f\n", nm, mean(sc[, 1]), mean(sc[, 2]), mean(sc[, 3])))
}

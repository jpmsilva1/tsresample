# Installs everything the original R code needs. Run once:  Rscript install_packages.R
options(repos = c(CRAN = "https://cloud.r-project.org/"))
need <- c("xts", "zoo", "lubridate", "e1071", "randomForest", "rpart", "earth", "forecast",
          "remotes", "Hmisc", "fields", "ROCR", "gdata", "operators", "abind", "quantmod", "class", "parallelMap")
inst <- rownames(installed.packages())
for (p in setdiff(need, inst)) install.packages(p, dependencies = TRUE)
inst <- rownames(installed.packages())
if (!"performanceEstimation" %in% inst) {
  r <- try(install.packages("performanceEstimation"))
  if (!"performanceEstimation" %in% rownames(installed.packages()))
    remotes::install_github("ltorgo/performanceEstimation", upgrade = "never")
}
if (!"DMwR" %in% inst) {
  # archived on CRAN in 2021; the last version is in the archive
  remotes::install_url("https://cran.r-project.org/src/contrib/Archive/DMwR/DMwR_0.4.1.tar.gz", upgrade = "never")
}
if (!"UBL" %in% rownames(installed.packages())) {
  r <- try(install.packages("UBL"))
  if (!"UBL" %in% rownames(installed.packages()))
    remotes::install_url("https://cran.r-project.org/src/contrib/Archive/UBL/UBL_0.0.9.tar.gz", upgrade = "never")
}
if (!"uba" %in% rownames(installed.packages())) {
  # not on CRAN: build from the GitHub source (stale .o/.so files must be removed first)
  src <- file.path(tempdir(), "uba")
  system(paste("rm -rf", src, "&& git clone -q https://github.com/rpribeiro/uba", src))
  unlink(list.files(file.path(src, "src"), pattern = "\\.(o|so|rds)$", full.names = TRUE))
  install.packages(src, repos = NULL, type = "source")
}
for (p in c("xts", "lubridate", "performanceEstimation", "uba", "UBL", "DMwR", "e1071", "randomForest", "rpart", "earth", "forecast")) {
  cat(sprintf("%-22s %s\n", p, if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else "MISSING"))
}

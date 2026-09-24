# Rscript earth_all.R <windir> <nk> <degree> <thresh> <pattern>
# For every <pattern>_train.csv: fit earth, write <name>_earth_pred.csv, <name>_earth_model.csv
suppressPackageStartupMessages(library(earth))
a <- commandArgs(trailingOnly=TRUE)
dir <- a[1]; nk <- as.integer(a[2]); degree <- as.integer(a[3]); thresh <- as.numeric(a[4]); pat <- a[5]
files <- list.files(dir, pattern=paste0("^", pat, ".*_train\\.csv$"))
for (f in files) {
  name <- sub("_train\\.csv$", "", f)
  tr <- read.csv(file.path(dir, f)); te <- read.csv(file.path(dir, paste0(name, "_test.csv")))
  m <- earth(V10 ~ ., tr, nk=nk, degree=degree, thresh=thresh)
  p <- as.vector(predict(m, te))
  write.csv(data.frame(pred=p), file.path(dir, paste0(name, "_earth_pred.csv")), row.names=FALSE)
  sel <- m$selected.terms
  md <- data.frame(term=seq_len(nrow(m$dirs)), selected=as.integer(seq_len(nrow(m$dirs)) %in% sel),
                   coef=NA_real_, name=rownames(m$dirs))
  md$coef[sel] <- as.vector(m$coefficients)
  md <- cbind(md, as.data.frame(m$dirs), as.data.frame(m$cuts))
  names(md) <- c("term","selected","coef","name", paste0("dir", 1:ncol(m$dirs)), paste0("cut", 1:ncol(m$cuts)))
  write.csv(md, file.path(dir, paste0(name, "_earth_model.csv")), row.names=FALSE)
  cat(sprintf("%s: forward %d terms, selected %d, termcond %d, rss %.6g\n", name, nrow(m$dirs), length(sel), m$termcond, m$rss))
}

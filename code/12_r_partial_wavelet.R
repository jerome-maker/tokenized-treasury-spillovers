## 12_r_partial_wavelet.R
## Partial wavelet coherence (Section 3.7.3 of the underlying research design,
## previously left as a documented gap): both tokenized-Treasury market-cap
## returns and tech-equity returns could be jointly driven by the same
## short-rate factor (the Fed funds rate transmits into 3-month T-bill
## yields, which affect risk-free-referenced tokens directly and growth
## equities through the discount rate). A raw pairwise wavelet coherence
## cannot separate that common-factor channel from a direct token<->equity
## linkage; partial wavelet coherence, controlling for the 3-month T-bill
## yield change, can. Implemented via biwavelet::pwtc, the package named for
## this purpose in the original research design.

# If your R packages live in a non-default user library, uncomment and edit:
# .libPaths(c("/path/to/your/R/library", .libPaths()))
suppressPackageStartupMessages(library(biwavelet))

# Resolve repo root: works via `Rscript code/12_r_partial_wavelet.R` run from
# the repo root, or via source() with the working directory set to code/.
root <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1) {
    normalizePath(file.path(dirname(sub("^--file=", "", file_arg)), ".."))
  } else {
    normalizePath("..")
  }
})
proc_dir <- file.path(root, "data", "processed")

df <- read.csv(file.path(proc_dir, "returns_panel.csv"), stringsAsFactors = FALSE)
df$date <- as.Date(df$date)

# y = token/benchmark market-facing return, x1 = risky-asset return partialled
# against, x2 = control (3-month T-bill yield change, the most direct proxy
# for the Fed funds rate available in this pipeline; see main text Sec. 3).
pairs <- list(
  list(name = "BUIDL_FANG", y = "R_BUIDL_mkt", x1 = "R_FANG"),
  list(name = "OUSG_FANG",  y = "R_OUSG_mkt",  x1 = "R_FANG"),
  list(name = "sUSDS_FANG", y = "R_sUSDS_mkt", x1 = "R_FANG"),
  list(name = "OUSG_SOXX",  y = "R_OUSG_mkt",  x1 = "R_SOXX"),
  list(name = "IEF_GLD",    y = "R_IEF",       x1 = "R_GLD")
)
control_col <- "Y3M_chg"

bands <- list(c(2, 8), c(8, 32), c(32, 128))
band_names <- c("D1-D2 (2-8d)", "D3-D4 (8-32d)", "D5-D6 (32-128d)")

band_avg <- function(rsq, period, coi) {
  out <- numeric(length(bands))
  for (k in seq_along(bands)) {
    lo <- bands[[k]][1]; hi <- bands[[k]][2]
    mask_period <- period >= lo & period < hi
    vals <- c()
    for (ti in seq_len(ncol(rsq))) {
      valid <- mask_period & (period < coi[ti])
      if (any(valid)) vals <- c(vals, mean(rsq[valid, ti]))
    }
    out[k] <- if (length(vals)) mean(vals) else NA
  }
  out
}

rows <- list()
set.seed(20260819)

for (p in pairs) {
  sub <- df[, c("date", p$y, p$x1, control_col)]
  names(sub) <- c("date", "y", "x1", "x2")
  sub <- sub[complete.cases(sub), ]
  if (nrow(sub) < 100) {
    cat(sprintf("[%s] skipped: only %d obs\n", p$name, nrow(sub)))
    next
  }
  t_idx <- seq_len(nrow(sub))
  d_y <- cbind(t_idx, sub$y)
  d_x1 <- cbind(t_idx, sub$x1)
  d_x2 <- cbind(t_idx, sub$x2)

  cat(sprintf("[%s] n=%d: fitting plain WTC...\n", p$name, nrow(sub)))
  # nrands kept small: the band-averaged rsq reported below does not use the
  # Monte Carlo significance surface, only the coherence magnitude itself, so
  # a large surrogate count would only slow this down without changing the
  # reported numbers.
  wtc_res <- wtc(d_y, d_x1, nrands = 10, quiet = TRUE)
  cat(sprintf("[%s] fitting partial WTC (control=%s)...\n", p$name, control_col))
  pwtc_res <- pwtc(d_y, d_x1, d_x2, nrands = 10, quiet = TRUE)

  wtc_band <- band_avg(wtc_res$rsq, wtc_res$period, wtc_res$coi)
  pwtc_band <- band_avg(pwtc_res$rsq, pwtc_res$period, pwtc_res$coi)

  row <- data.frame(pair = p$name, n_obs = nrow(sub), band = band_names,
                     WTC = wtc_band, PWC = pwtc_band, attenuation = wtc_band - pwtc_band)
  rows[[p$name]] <- row
  cat(sprintf("[%s] WTC=%s | PWC=%s\n", p$name,
              paste(round(wtc_band, 3), collapse = ","), paste(round(pwtc_band, 3), collapse = ",")))

  saveRDS(list(wtc = wtc_res, pwtc = pwtc_res, dates = sub$date),
          file.path(proc_dir, paste0("pwtc_", p$name, ".rds")))
}

result_df <- do.call(rbind, rows)
write.csv(result_df, file.path(proc_dir, "r_partial_wavelet.csv"), row.names = FALSE)
cat("\n=== Summary ===\n")
print(result_df)

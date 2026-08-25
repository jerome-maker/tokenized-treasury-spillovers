## 10_r_tvpvar_frequency.R
## TVP-VAR based dynamic connectedness (Antonakakis, Chatziantoniou and
## Gabauer 2020) and its Baruník-Křehlík (2018) frequency-domain
## decomposition, via the ConnectednessApproach package. This replaces the
## Python rolling-window-VAR approximation as the primary time-varying
## spillover estimator in the main text.
##
## System R1: FANG, SOXX, SMH, SHV, IEF, BUIDL, OUSG, BTC, ETH, UNI, AAVE, GLD
## (12 variables; window governed by BUIDL's start, ~527 observations).
## sUSDS is excluded from this system (its ~370-obs window would force every
## other series to shrink too); it remains covered by the Python DCC/hedge
## results and the pairwise Bai-Perron/bootstrap analysis in script 11.

# If your R packages live in a non-default user library, uncomment and edit:
# .libPaths(c("/path/to/your/R/library", .libPaths()))
suppressPackageStartupMessages({
  library(ConnectednessApproach)
  library(zoo)
})

# Resolve repo root: works via `Rscript code/10_r_tvpvar_frequency.R` run from
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

sys_cols <- c(FANG = "R_FANG", SOXX = "R_SOXX", SMH = "R_SMH", SHV = "R_SHV", IEF = "R_IEF",
              BUIDL = "R_BUIDL_mkt", OUSG = "R_OUSG_mkt",
              BTC = "R_BTC", ETH = "R_ETH", UNI = "R_UNI", AAVE = "R_AAVE", GLD = "R_GLD")

sub <- df[, c("date", unname(sys_cols))]
names(sub) <- c("date", names(sys_cols))
sub <- sub[complete.cases(sub), ]
cat("System R1 sample:", nrow(sub), "obs,", as.character(min(sub$date)), "to", as.character(max(sub$date)), "\n")

X <- zoo(as.matrix(sub[, names(sys_cols)]), order.by = sub$date)

## ---- TVP-VAR, time-domain dynamic connectedness ----
## kappa1 = 0.99 (forgetting factor, coefficients), kappa2 = 0.99 (forgetting
## factor, error covariance) -- matches the underlying research design's
## stated defaults (Sec. 3.8.3), with a sensitivity check at (0.96, 0.94).
cache_time <- file.path(proc_dir, "_cache_time_cn.rds")
if (file.exists(cache_time)) {
  cat("Loading cached time-domain TVP-VAR fit\n")
  time_cn <- readRDS(cache_time)
} else {
  cat("\n=== Fitting TVP-VAR time-domain connectedness (kappa=0.99/0.99) ===\n")
  time_cn <- ConnectednessApproach(X, nlag = 1, nfore = 10, model = "TVP-VAR",
                                    connectedness = "Time",
                                    VAR_config = list(TVPVAR = list(kappa1 = 0.99, kappa2 = 0.99,
                                                                     prior = "BayesPrior", gamma = 0.01)))
  saveRDS(time_cn, cache_time)
}

# row/column label case differs between Time and Frequency modes ("NET" vs
# "Net"); match case-insensitively rather than hard-coding one spelling.
net_row <- function(tb) rownames(tb)[tolower(rownames(tb)) == "net"]
from_col <- function(tb) colnames(tb)[tolower(colnames(tb)) == "from"]
to_row <- function(tb) rownames(tb)[tolower(rownames(tb)) == "to"]

tb <- time_cn$TABLE
nvars <- ncol(X)
static_tbl <- data.frame(
  variable = colnames(X),
  TO = as.numeric(tb[to_row(tb), 1:nvars]),
  FROM = as.numeric(tb[1:nvars, from_col(tb)]),
  NET = as.numeric(tb[net_row(tb), 1:nvars])
)
write.csv(static_tbl, file.path(proc_dir, "r_tvpvar_static_table.csv"), row.names = FALSE)
cat("Static table (time-domain, System R1):\n")
print(static_tbl)
cat("Mean TCI:", mean(time_cn$TCI), "\n")

tci_ts <- data.frame(date = as.Date(rownames(time_cn$TCI)), TCI = as.numeric(time_cn$TCI))
write.csv(tci_ts, file.path(proc_dir, "r_tvpvar_tci_series.csv"), row.names = FALSE)

net_df <- as.data.frame(time_cn$NET)
net_df$date <- as.Date(rownames(time_cn$NET))
write.csv(net_df, file.path(proc_dir, "r_tvpvar_net_series.csv"), row.names = FALSE)

npdc_static <- time_cn$NPDC[, , dim(time_cn$NPDC)[3]]  # last-period NPDC as a static summary snapshot is not
# meaningful; instead average NPDC over time for a static pairwise summary:
npdc_avg <- apply(time_cn$NPDC, c(1, 2), mean)
write.csv(as.data.frame(npdc_avg), file.path(proc_dir, "r_tvpvar_npdc.csv"), row.names = TRUE)

## sensitivity check at (kappa1, kappa2) = (0.96, 0.94)
cache_alt <- file.path(proc_dir, "_cache_time_cn_alt.rds")
if (file.exists(cache_alt)) {
  cat("Loading cached sensitivity fit\n")
  time_cn_alt <- readRDS(cache_alt)
} else {
  cat("\n=== Sensitivity: kappa=0.96/0.94 ===\n")
  time_cn_alt <- ConnectednessApproach(X, nlag = 1, nfore = 10, model = "TVP-VAR",
                                        connectedness = "Time",
                                        VAR_config = list(TVPVAR = list(kappa1 = 0.96, kappa2 = 0.94,
                                                                         prior = "BayesPrior", gamma = 0.01)))
  saveRDS(time_cn_alt, cache_alt)
}
cat("Mean TCI (0.99/0.99):", mean(time_cn$TCI), " | Mean TCI (0.96/0.94):", mean(time_cn_alt$TCI), "\n")
write.csv(data.frame(kappa = c("0.99/0.99", "0.96/0.94"),
                      mean_TCI = c(mean(time_cn$TCI), mean(time_cn_alt$TCI))),
          file.path(proc_dir, "r_tvpvar_kappa_sensitivity.csv"), row.names = FALSE)

## ---- Frequency-domain (Baruník-Křehlík) decomposition ----
## Bands: short (1-5 days), medium (5-20 days), long (20+ days) -- matches
## Sec. 3.8.4 of the research design. The package auto-labels bands from the
## radian partition; a benign "partition does not cover the whole range"
## warning is emitted regardless of partition choice (confirmed on synthetic
## data) and does not affect the returned band boundaries or values.
cache_freq <- file.path(proc_dir, "_cache_freq_cn.rds")
if (file.exists(cache_freq)) {
  cat("Loading cached frequency-domain fit\n")
  freq_cn <- readRDS(cache_freq)
} else if (file.exists(file.path(proc_dir, "_tmp_freq_cn.rds"))) {
  cat("Loading previously-saved frequency-domain fit\n")
  freq_cn <- readRDS(file.path(proc_dir, "_tmp_freq_cn.rds"))
  saveRDS(freq_cn, cache_freq)
} else {
  cat("\n=== Fitting TVP-VAR frequency-domain connectedness ===\n")
  freq_cn <- suppressWarnings(ConnectednessApproach(
    X, nlag = 1, nfore = 100, model = "TVP-VAR", connectedness = "Frequency",
    VAR_config = list(TVPVAR = list(kappa1 = 0.99, kappa2 = 0.99, prior = "BayesPrior", gamma = 0.01)),
    Connectedness_config = list(FrequencyConnectedness = list(partition = c(pi, pi / 5, pi / 20, 0),
                                                                generalized = TRUE, scenario = "ABS"))))
  saveRDS(freq_cn, cache_freq)
}

band_labels <- colnames(freq_cn$TCI)  # e.g. "Total","1-5","5-20","20-Inf"
cat("Bands found:", paste(band_labels, collapse = ", "), "\n")

freq_rows <- list()
for (i in seq_along(band_labels)) {
  bn <- band_labels[i]
  tb_i <- freq_cn$TABLE[, , i]
  freq_rows[[bn]] <- data.frame(
    variable = colnames(X),
    TO = as.numeric(tb_i[to_row(tb_i), 1:nvars]),
    FROM = as.numeric(tb_i[1:nvars, from_col(tb_i)]),
    NET = as.numeric(tb_i[net_row(tb_i), 1:nvars]),
    band = bn
  )
}
freq_df <- do.call(rbind, freq_rows)
rownames(freq_df) <- NULL
write.csv(freq_df, file.path(proc_dir, "r_frequency_static_table.csv"), row.names = FALSE)
cat("Frequency-domain static table:\n")
print(freq_df)

tci_band_list <- list()
for (i in seq_along(band_labels)) {
  bn <- band_labels[i]
  tci_band_list[[bn]] <- data.frame(date = as.Date(rownames(freq_cn$TCI)),
                                     TCI = as.numeric(freq_cn$TCI[, i]), band = bn)
}
tci_band_df <- do.call(rbind, tci_band_list)
write.csv(tci_band_df, file.path(proc_dir, "r_frequency_tci_series.csv"), row.names = FALSE)

cat("\nDone. Outputs written to", proc_dir, "\n")

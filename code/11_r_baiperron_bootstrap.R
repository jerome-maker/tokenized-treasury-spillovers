## 11_r_baiperron_bootstrap.R
## (a) Exact multiple structural-change tests (Bai and Perron 1998, 2003) on
##     each pair's DCC correlation series rho_t, via strucchange::breakpoints
##     (BIC-optimal break count) and strucchange::sctest (supF test of the
##     no-break null) -- this was left undone in the Python-only version for
##     lack of the strucchange package, and is now run properly.
## (b) Moving block-bootstrap confidence intervals for each pair's mean DCC
##     correlation, via boot::tsboot (fixed block length set from each
##     series' own persistence, 1,000 replications, matching Sec. 3.6.4 (4)
##     and Sec. 3.9.1 of the underlying research design).

# If your R packages live in a non-default user library, uncomment and edit:
# .libPaths(c("/path/to/your/R/library", .libPaths()))
suppressPackageStartupMessages({
  library(strucchange)
  library(boot)
})

# Resolve repo root: works via `Rscript code/11_r_baiperron_bootstrap.R` run
# from the repo root, or via source() with the working directory set to code/.
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

dcc_files <- list.files(proc_dir, pattern = "^dcc_.*\\.csv$", full.names = TRUE)
cat("Found", length(dcc_files), "DCC pair files\n")

set.seed(20260818)

bp_rows <- list()
boot_rows <- list()

for (f in dcc_files) {
  pair_name <- sub("^dcc_", "", sub("\\.csv$", "", basename(f)))
  d <- read.csv(f, stringsAsFactors = FALSE)
  rho <- d$rho_dcc
  n <- length(rho)
  if (n < 60 || sd(rho, na.rm = TRUE) < 1e-8) {
    cat(sprintf("[%s] skipped: n=%d, sd=%.2e (flat/degenerate series)\n", pair_name, n, sd(rho, na.rm = TRUE)))
    next
  }
  # Quality flag: several pairs' DCC fit pins the news parameter a at 0 (see
  # 04_garch_dcc.py / dcc_adcc_summary.csv), leaving rho_t constant up to
  # ~1e-6 floating-point noise from the recursive Q_t computation. A supF
  # break test on such a series can report a formally "significant" break
  # purely because residual variance is near-zero (huge F-stat from
  # economically meaningless numerical noise), not a real structural change.
  # sd(rho) < 1e-3 is used as the cutoff (real dynamic pairs show sd on the
  # order of 0.05-0.2; degenerate ones are ~1e-6), and is reported alongside
  # the test rather than silently trusted.
  is_degenerate <- sd(rho, na.rm = TRUE) < 1e-3

  ## ---- (a) Bai-Perron structural breaks ----
  bp_res <- tryCatch({
    ts_rho <- ts(rho)
    bp <- breakpoints(ts_rho ~ 1, h = 0.15, breaks = 5)
    bic_tab <- summary(bp)$RSS["BIC", ]
    opt_m <- which.min(bic_tab) - 1  # breakpoints() summary includes m=0..5; index 1 == 0 breaks
    fs <- Fstats(ts_rho ~ 1, from = 0.15)
    sc <- sctest(fs, type = "supF")
    ci <- if (opt_m > 0) tryCatch(confint(bp, breaks = opt_m), error = function(e) NULL) else NULL
    list(opt_breaks = opt_m, supF_stat = unname(sc$statistic), supF_pvalue = sc$p.value,
         breakpoints = if (opt_m > 0) bp$breakpoints else NA, bic_table = bic_tab, ci = ci)
  }, error = function(e) {
    cat(sprintf("[%s] Bai-Perron FAILED: %s\n", pair_name, conditionMessage(e)))
    NULL
  })

  if (!is.null(bp_res)) {
    bp_dates <- if (!is.na(bp_res$breakpoints[1])) paste(d$date[bp_res$breakpoints], collapse = ";") else ""
    bp_rows[[pair_name]] <- data.frame(
      pair = pair_name, n_obs = n, sd_rho = sd(rho, na.rm = TRUE), degenerate = is_degenerate,
      opt_num_breaks = bp_res$opt_breaks,
      supF_stat = bp_res$supF_stat, supF_pvalue = bp_res$supF_pvalue,
      break_dates = bp_dates
    )
    flag <- if (is_degenerate) " [DEGENERATE: sd(rho)<1e-3, test not economically meaningful]" else ""
    cat(sprintf("[%s] Bai-Perron: %d break(s), supF=%.2f (p=%.4f), dates=%s%s\n",
                pair_name, bp_res$opt_breaks, bp_res$supF_stat, bp_res$supF_pvalue, bp_dates, flag))
  }

  ## ---- (b) moving block bootstrap CI for the mean correlation ----
  boot_res <- tryCatch({
    acf1 <- acf(rho, plot = FALSE, lag.max = 1)$acf[2]
    # block length: longer for more persistent series, capped for short samples
    block_len <- max(5, min(round(n^(1 / 3) / max(1 - abs(acf1), 0.05)), floor(n / 10)))
    mean_stat <- function(x) mean(x)
    tb <- tsboot(rho, statistic = mean_stat, R = 1000, l = block_len, sim = "fixed")
    ci <- boot.ci(tb, type = "perc")
    list(block_len = block_len, boot_mean = mean(tb$t), boot_se = sd(tb$t),
         ci_lo = ci$percent[4], ci_hi = ci$percent[5])
  }, error = function(e) {
    cat(sprintf("[%s] block bootstrap FAILED: %s\n", pair_name, conditionMessage(e)))
    NULL
  })

  if (!is.null(boot_res)) {
    boot_rows[[pair_name]] <- data.frame(
      pair = pair_name, n_obs = n, degenerate = is_degenerate, block_length = boot_res$block_len,
      point_mean = mean(rho), boot_mean = boot_res$boot_mean, boot_se = boot_res$boot_se,
      ci95_lo = boot_res$ci_lo, ci95_hi = boot_res$ci_hi
    )
    cat(sprintf("[%s] block-bootstrap (l=%d): mean=%.4f, 95%% CI=[%.4f, %.4f]\n",
                pair_name, boot_res$block_len, mean(rho), boot_res$ci_lo, boot_res$ci_hi))
  }
}

bp_df <- do.call(rbind, bp_rows)
boot_df <- do.call(rbind, boot_rows)
write.csv(bp_df, file.path(proc_dir, "r_baiperron_results.csv"), row.names = FALSE)
write.csv(boot_df, file.path(proc_dir, "r_blockbootstrap_results.csv"), row.names = FALSE)

cat("\n=== Bai-Perron summary ===\n"); print(bp_df)
cat("\n=== Block bootstrap summary ===\n"); print(boot_df)
cat("\nDone.\n")

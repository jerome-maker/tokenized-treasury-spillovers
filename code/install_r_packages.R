## install_r_packages.R
## Installs the R packages actually used by 10_r_tvpvar_frequency.R,
## 11_r_baiperron_bootstrap.R, and 12_r_partial_wavelet.R:
##   - ConnectednessApproach (Antonakakis-Chatziantoniou-Gabauer TVP-VAR and
##     Barunik-Krehlik frequency-domain connectedness; on CRAN)
##   - zoo, strucchange, boot, biwavelet (all on CRAN)
## Installs into the default user library (no hardcoded machine-specific
## path); run once before the R scripts in code/.

options(repos = c(CRAN = "https://cloud.r-project.org"))

pkgs <- c("ConnectednessApproach", "zoo", "strucchange", "boot", "biwavelet")

for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    cat("=== installing", p, "===\n")
    tryCatch(
      install.packages(p, dependencies = TRUE),
      error = function(e) cat("FAILED:", p, conditionMessage(e), "\n")
    )
  } else {
    cat(p, "already installed\n")
  }
}

cat("\n\n=== FINAL STATUS ===\n")
for (p in pkgs) {
  cat(p, ":", requireNamespace(p, quietly = TRUE), "\n")
}

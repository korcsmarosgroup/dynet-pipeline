# =============================================================================
# dynetR_significance() — manual p-value walkthrough
#
# This script reproduces the core mechanics of dynetR_significance() in plain,
# step-by-step R so you can inspect every intermediate object.
#
# Networks are kept tiny (4 nodes, ~3 edges each) so every matrix fits on screen.
# We use only 50 iterations so the whole script runs in a few seconds.
# =============================================================================

devtools::load_all(".")   # adjust if running from outside the package root
set.seed(99)

# -----------------------------------------------------------------------------
# 0. Define three small directed networks
#    Node "a" rewires heavily (different partners in each network).
#    Nodes "b","c","d" are stable.
# -----------------------------------------------------------------------------
nodes <- c("a", "b", "c", "d")

# Network A: a → b, b → c, c → d
A <- matrix(0, 4, 4, dimnames = list(nodes, nodes))
A["a", "b"] <- 1;  A["b", "c"] <- 1;  A["c", "d"] <- 1

# Network B: a → c,  b → c, c → d   (a switched from b to c)
B <- matrix(0, 4, 4, dimnames = list(nodes, nodes))
B["a", "c"] <- 1;  B["b", "c"] <- 1;  B["c", "d"] <- 1

# Network C: a → d,  b → c, c → d   (a switched again to d)
C <- matrix(0, 4, 4, dimnames = list(nodes, nodes))
C["a", "d"] <- 1;  C["b", "c"] <- 1;  C["c", "d"] <- 1

nets <- list(A = A, B = B, C = C)

cat("=== Input networks ===\n")
cat("Network A:\n"); print(A)
cat("Network B:\n"); print(B)
cat("Network C:\n"); print(C)

# -----------------------------------------------------------------------------
# 1. Observed rewiring scores  (what dynetR() computes)
# -----------------------------------------------------------------------------
obs <- dynetR(nets)
cat("\n=== Observed rewiring scores ===\n")
print(obs)
# Expected: node "a" has the highest rewiring score because it changes targets
# in every network.  b, c, d are stable so their scores should be near zero.

# -----------------------------------------------------------------------------
# 2. Reproduce ONE null-distribution draw by hand
#    (This is exactly what .randomize_network() + .compute_rewiring_scores() do
#     inside the loop.)
# -----------------------------------------------------------------------------
cat("\n=== One manual null draw ===\n")

# Randomize each network independently, preserving its degree sequence.
# igraph::rewire(keeping_degseq()) shuffles edge endpoints while keeping
# every node's in-degree and out-degree fixed.
set.seed(7)
rand_nets <- lapply(nets, function(mat) {
  g      <- igraph::graph_from_adjacency_matrix(mat, mode = "directed",
                                                weighted = TRUE, diag = TRUE)
  n_edge <- igraph::ecount(g)
  if (n_edge == 0) return(mat)
  g_rand <- igraph::rewire(g, with = igraph::keeping_degseq(loops = FALSE,
                                                            niter = n_edge * 10))
  rand_mat <- as.matrix(igraph::as_adjacency_matrix(g_rand, sparse = FALSE))
  rownames(rand_mat) <- colnames(rand_mat) <- rownames(mat)
  rand_mat
})

cat("Randomized network A:\n"); print(rand_nets$A)
cat("Randomized network B:\n"); print(rand_nets$B)
cat("Randomized network C:\n"); print(rand_nets$C)

# Compute rewiring on the randomized set
null_scores_one <- dynetR(rand_nets)
cat("\nNull rewiring scores (one draw):\n")
print(null_scores_one)

# Compare to observed
cat("\nObserved vs null (raw rewiring):\n")
comparison <- data.frame(
  node     = obs$name,
  observed = obs$rewiring,
  null     = null_scores_one$rewiring[match(obs$name, null_scores_one$name)],
  row.names = NULL
)
print(comparison)

# -----------------------------------------------------------------------------
# 3. Full null distribution: 50 iterations
# -----------------------------------------------------------------------------
N_ITER <- 50
cat(sprintf("\n=== Building null distribution (%d iterations) ===\n", N_ITER))

set.seed(42)
null_matrix <- matrix(NA_real_,
                      nrow = nrow(obs), ncol = N_ITER,
                      dimnames = list(obs$name, paste0("iter", seq_len(N_ITER))))

for (i in seq_len(N_ITER)) {
  rand <- lapply(nets, function(mat) {
    g      <- igraph::graph_from_adjacency_matrix(mat, mode = "directed",
                                                  weighted = TRUE, diag = TRUE)
    n_edge <- igraph::ecount(g)
    if (n_edge == 0) return(mat)
    g_rand <- igraph::rewire(g, with = igraph::keeping_degseq(loops = FALSE,
                                                              niter = n_edge * 10))
    rand_mat <- as.matrix(igraph::as_adjacency_matrix(g_rand, sparse = FALSE))
    rownames(rand_mat) <- colnames(rand_mat) <- rownames(mat)
    rand_mat
  })
  null_scores_i <- dynetR(rand)
  idx <- match(obs$name, null_scores_i$name)
  null_matrix[, i] <- null_scores_i$rewiring[idx]
}

cat("\nNull distribution matrix (rows = nodes, cols = iterations):\n")
print(round(null_matrix, 3))

# -----------------------------------------------------------------------------
# 4. Compute p-values manually
#    p-value = fraction of null scores >= observed score
# -----------------------------------------------------------------------------
cat("\n=== P-value calculation ===\n")
cat("Formula:  p(node) = mean( null_score[node, ] >= observed_score[node] )\n\n")

obs_rewiring <- setNames(obs$rewiring, obs$name)

p_values <- sapply(obs$name, function(node) {
  null_row  <- null_matrix[node, ]
  obs_score <- obs_rewiring[node]
  n_geq     <- sum(null_row >= obs_score)
  cat(sprintf(
    "  node '%s': observed = %.4f  |  null >= obs: %d/%d  |  p = %.3f\n",
    node, obs_score, n_geq, N_ITER, n_geq / N_ITER
  ))
  n_geq / N_ITER
})

cat("\nFinal p-values:\n")
print(round(p_values, 3))

# Verify these match dynetR_significance() output
cat("\n=== Cross-check against dynetR_significance() ===\n")
full_res <- dynetR_significance(nets, n_iter = N_ITER, seed = 42)
cat("dynetR_significance() p_values:\n")
print(round(setNames(full_res$scores$p_value, full_res$scores$name), 3))
cat("(Small differences from manual above are expected because dynetR_significance\n")
cat(" uses per-iteration seeds derived from seed+i rather than a shared RNG stream.)\n")

# -----------------------------------------------------------------------------
# 5. Visualise the null distribution for node "a"
# -----------------------------------------------------------------------------
cat("\n=== Null distribution histogram for node 'a' ===\n")

node_focus <- "a"
null_a     <- null_matrix[node_focus, ]
obs_a      <- obs_rewiring[node_focus]

cat(sprintf("Observed rewiring for '%s': %.4f\n", node_focus, obs_a))
cat(sprintf("Null range: [%.4f, %.4f],  mean: %.4f\n",
            min(null_a), max(null_a), mean(null_a)))
cat(sprintf("Fraction of null >= observed: %d/%d = %.3f  (this is the p-value)\n",
            sum(null_a >= obs_a), N_ITER, mean(null_a >= obs_a)))

# Simple ASCII histogram
breaks <- seq(floor(min(null_a) * 10) / 10,
              ceiling(max(c(null_a, obs_a)) * 10) / 10,
              by = 0.1)
counts <- hist(null_a, breaks = breaks, plot = FALSE)$counts
cat("\nNull distribution (each * = 1 iteration):\n")
for (k in seq_along(counts)) {
  bar_label <- sprintf("[%.1f, %.1f)", breaks[k], breaks[k + 1])
  marker    <- if (obs_a >= breaks[k] && obs_a < breaks[k + 1]) " <-- OBSERVED" else ""
  cat(sprintf("  %s | %s%s\n",
              bar_label, paste(rep("*", counts[k]), collapse = ""), marker))
}
if (obs_a >= tail(breaks, 1)) {
  cat(sprintf("  Observed (%.4f) is beyond the right edge of the null distribution.\n", obs_a))
}

#' @title Helper function to set rownames equal to colnames for a single matrix
#'
#' @description
#' Helps to set matching row and colum nnames
#' @param mat A single input matrix
#'

.set_rownames_to_colnames <- function(mat) {
  if (!is.null(colnames(mat))) {
    rownames(mat) <- colnames(mat)
  }
  return(mat)
}

#' @title Helper function to format weighted edge list dataframe
#'
#' @description Formats edge list to adjacency matrix
#'
#' @param el An input weighted edge list with from, to, and weight columns
#'
#'
.edgelist_to_adjacency <- function(el) {
  g1 <- graph_from_data_frame(el, directed = TRUE)
  gdf1 <- as_adjacency_matrix(g1, attr = "weight", sparse = F)
}



#' @title Helper function to format input data, matrix or edge list df
#' @description Formats input list data depending on input type
#' @param input_list A list of matrices or edge lists
#'
#'
format_indata <- function(input_list) {
  # checking if elements are matrices
  if (all(sapply(input_list, is.matrix))) {
    # Ensure rownames match colnames for matrices
    formatted_matrix_list <- lapply(input_list, .set_rownames_to_colnames)
  } else if (all(sapply(input_list, is.data.frame))) {
    formatted_matrix_list <- lapply(input_list, .edgelist_to_adjacency)
  } else if (all(sapply(input_list, is_igraph))) {
    formatted_matrix_list <- lapply(input_list, function(graph) {
      mat <- as_adjacency_matrix(graph, sparse = FALSE)
      # Get node names, if they exist; otherwise, use node indices
      node_names <- V(graph)$name
      if (is.null(node_names)) {
        node_names <- as.character(V(graph))
      }
      rownames(mat) <- node_names
      colnames(mat) <- node_names
      return(mat)
    })
  } else if (all(sapply(input_list, is.tbl_graph))) {
    formatted_matrix_list <- lapply(input_list, function(graph) {
      mat <- as_adjacency_matrix(graph, sparse = FALSE)
      node_names <- V(graph)$name
      if (is.null(node_names)) {
        node_names <- as.character(V(graph))
      }
      rownames(mat) <- node_names
      colnames(mat) <- node_names
      return(mat)
    })
  } else {
    stop("Input must be a matrix, edge list data frame, igraph object or tbl_graph object.")
  }
  return(formatted_matrix_list)
}



#' @title Helper function to expand adjacency matrices to include all unique nodes
#' @description Expands formatted adjacency matrices to matching sizes
#' @param adj_matrices An input list of adjacency matrices
#'
#'
#'
.expand_adjacency_matrices <- function(adj_matrices) {
  # Identify all unique nodes across all adjacency matrices
  all_nodes <- unique(unlist(lapply(adj_matrices, rownames)))

  # Function to expand a single adjacency matrix
  expand_matrix <- function(matrix, all_nodes) {
    current_nodes <- rownames(matrix)
    missing_nodes <- setdiff(all_nodes, current_nodes)

    # Create an expanded matrix with all nodes
    expanded_matrix <- Matrix(0,
                              nrow = length(all_nodes), ncol = length(all_nodes),
                              dimnames = list(all_nodes, all_nodes), sparse = F
    )

    # Fill in the existing values
    expanded_matrix[current_nodes, current_nodes] <- matrix[current_nodes, current_nodes]

    expanded_matrix <- as.matrix(expanded_matrix)

    return(expanded_matrix)
  }

  # Apply the expand_matrix function to each adjacency matrix in the list
  expanded_matrices <- lapply(adj_matrices, expand_matrix, all_nodes = all_nodes)

  return(expanded_matrices)
}


#' @title Helper function to create a union of multiple adjacency matrices
#' @description Builds a union adjacency matrix from a list of adjacency matrices
#' @param adj_matrices An input list of adjacency matrices
#'

.union_adjacency_matrices <- function(inlist) {
  # List of input adjacency matrices
  matrices <- inlist

  # Get the union of all nodes across all matrices
  all_nodes <- unique(unlist(lapply(matrices, rownames)))

  # Create an empty adjacency matrix for the union
  union_matrix <- matrix(0, nrow = length(all_nodes), ncol = length(all_nodes))
  rownames(union_matrix) <- all_nodes
  colnames(union_matrix) <- all_nodes

  # Populate the union matrix with edges from each input matrix
  for (adj in matrices) {
    nodes <- rownames(adj)
    for (i in seq_along(nodes)) {
      for (j in seq_along(nodes)) {
        if (adj[i, j] > 0) {
          union_matrix[nodes[i], nodes[j]] <- 1
        }
      }
    }
  }

  return(union_matrix)
}

#' @title Helper function to create a combined edgelist out of input adj. matrices
#' @description Builds an edgelist out of adj. matrices
#' @param adj_matrices An input  adjacency matrix
#'

.indata_to_edgelist <- function(ingraph){
  g<-graph.adjacency(ingraph,weighted=TRUE)
  out<-get.data.frame(g)
  return(out)
}

#' Helper function to summarise matrices
.sumMatrices <- function(matrices) {
  # Initialize a variable to hold the result
  result <- matrix(0, nrow = nrow(matrices[[1]]), ncol = ncol(matrices[[1]]))
  # Initialize a counter matrix with the same dimensions as the result matrix
  counter <- matrix(0, nrow = nrow(result), ncol = ncol(result))
  # Loop through the list of matrices and sum them
  for (matrix in matrices) {
    result <- result + matrix
    counter <- counter + (matrix != 0)
  }
  # Divide the result matrix by the counter matrix element-wise
  result <- result / counter
  result[!is.finite(result)] <- 0
  # Return the result
  return(result)
}

#' Helper function to get the square values of the values in the list of matrices
.squareMatrices <- function(matrices) {
  # Initialize a list to hold the squared matrices
  squared_matrices <- list()

  # Loop through the list of matrices and square each one
  for (matrix in matrices) {
    squared_matrices[[length(squared_matrices) + 1]] <- matrix^2
  }

  # Return the list of squared matrices
  return(squared_matrices)
}

#' Helper function to calculate the centroid distances (squared Euclidean distances per node)
#' For directed graphs, each node's rewiring is computed over all incident edges
#' (out-edges from the row + in-edges from the column, self-loops counted once).
.centroidDistance <- function(matrices) {
  lapply(matrices, function(m) rowSums(m) + colSums(m) - diag(m))
}

# Helper function to format matrix for structural rewiring
.structure_format <- function(inmat){
  outmat <- ifelse(inmat!=0,1,inmat)
  return(outmat)
}

#' @title Compute degree data frame from the union of input networks
#'
#' @description Internal helper. Builds the union adjacency matrix across all networks,
#'   constructs a directed igraph from it, and returns a data frame of node degrees.
#'   Separated from \code{.compute_rewiring_scores()} so it can be called once and
#'   reused across all null-distribution iterations.
#'
#' @param mat_list_str A list of (pre-expansion) adjacency matrices.
#'
.compute_degree_df <- function(mat_list_str) {
  unimatrix <- .union_adjacency_matrices(mat_list_str)
  # Use directed graph; degree(mode="all") counts self-loops twice (in + out),
  # so subtract diag(unimatrix) to count each self-loop once (matching Java/Cytoscape behaviour).
  directed_union_graph <- graph_from_adjacency_matrix(unimatrix, mode = "directed", diag = TRUE)
  edge_counts <- degree(directed_union_graph, mode = "all") - diag(unimatrix)
  edge_counts |>
    as.data.frame() |>
    rownames_to_column() |>
    rename("name" = "rowname", "degree" = "edge_counts") |>
    tibble()
}


#' @title Core rewiring score computation on pre-expanded matrices
#'
#' @description Internal helper that runs the standardisation → centroid → rewiring pipeline
#'   on an already-expanded list of adjacency matrices. Returns the output data frame plus,
#'   when \code{return_intermediate = TRUE}, the per-network centroid-distance vectors needed
#'   for pairwise score derivation.
#'
#' @param expanded_matrices A list of same-dimension, named adjacency matrices.
#' @param degree_df A data frame with columns \code{name} and \code{degree}, as returned
#'   by \code{.compute_degree_df()}. Passed in so degree need only be computed once.
#' @param return_intermediate Logical; if TRUE also return \code{centDist} list.
#'
.compute_rewiring_scores <- function(expanded_matrices, degree_df,
                                     return_intermediate = FALSE) {
  # Calculate non-zero means of matrices
  nonZeroMean <- .sumMatrices(expanded_matrices)

  # Calculate standardized weights (divided by non-zero means)
  standardised <- lapply(expanded_matrices, "/", nonZeroMean)
  standardisedNoNan <- lapply(standardised, function(x) replace(x, is.nan(x), 0))

  # Calculate centroids of all node states by summing
  standardSums <- Reduce("+", standardisedNoNan)

  # Calculate Euclidean distance from node to centroid
  centroid <- standardSums / length(standardisedNoNan)

  # Subtract centroid values from standardized matrix
  standardMinusCentroid <- lapply(standardisedNoNan, "-", centroid)
  sqr <- .squareMatrices(standardMinusCentroid)
  centDist <- .centroidDistance(sqr)
  centDistBound <- do.call(rbind, centDist)
  centFinalDist <- colSums(centDistBound)
  rewiring <- centFinalDist / (length(centDist) - 1)

  rewiring_df <- rewiring |>
    as.data.frame() |>
    tibble::rownames_to_column() |>
    rename("name" = "rowname")
  output_dataframe <- left_join(rewiring_df, degree_df, by = "name") |>
    mutate(degree_corrected_rewiring = rewiring / degree)

  if (return_intermediate) {
    return(list(scores = output_dataframe, centDist = centDist))
  }
  return(output_dataframe)
}


#' @title Degree-sequence–preserving network randomization
#'
#' @description Internal helper. Rewires edges of a single adjacency matrix while
#'   preserving both in- and out-degree sequences (for directed graphs) using
#'   \code{igraph::rewire(keeping_degseq(...))}.
#'
#' @param mat A square numeric adjacency matrix with row/col names.
#' @param directed Logical; if TRUE treat the graph as directed.
#' @param niter_multiplier Integer multiplier for the number of rewiring steps
#'   (\code{ecount(g) * niter_multiplier}). Default 10.
#'
.randomize_network <- function(mat, directed = TRUE, niter_multiplier = 10) {
  mode <- if (directed) "directed" else "undirected"
  g <- igraph::graph_from_adjacency_matrix(mat, mode = mode, weighted = TRUE, diag = TRUE)
  has_loops <- any(diag(mat) != 0)
  n_edges <- igraph::ecount(g)
  if (n_edges == 0) return(mat)   # nothing to rewire

  g_rand <- igraph::rewire(g, with = igraph::keeping_degseq(
    loops = has_loops,
    niter  = max(1L, as.integer(n_edges * niter_multiplier))
  ))

  # Shuffle original non-zero weights onto the new edge set
  # (keeping_degseq moves stubs but igraph does not carry weights;
  #  we permute the observed weights randomly across the rewired edges)
  orig_weights <- igraph::E(g)$weight
  if (!is.null(orig_weights)) {
    igraph::E(g_rand)$weight <- sample(orig_weights)
  }

  rand_mat <- as.matrix(igraph::as_adjacency_matrix(g_rand, attr = "weight", sparse = FALSE))
  # Restore dimnames
  node_names <- rownames(mat)
  rownames(rand_mat) <- node_names
  colnames(rand_mat) <- node_names
  rand_mat
}


#' @title Calculate the DyNet rewiring values for multiple graphs (adjacency matrices)
#'
#' @description Calculates the Dn rewiring values for the nodes included in the networks
#' as described in `Goenawan et al., 2016.`
#' @import igraph
#' @import readr
#' @import ggplot2
#' @import ggraph
#' @import tidygraph
#' @import dplyr
#' @import tibble
#' @import Matrix
#' @param matrix_list List of adjacency matrices corresponding to the input networks.
#' @param structure_only Logical, replaces non-zero adj. matrix values with 1. Default is FALSE.
#'
#' @export

dynetR <- function(matrix_list, structure_only = FALSE) {
  # Format input data
  matrix_list <- format_indata(matrix_list)

  # Replace with 0 / 1 if structural rewiring only
  if (structure_only) {
    matrix_list_str <- lapply(matrix_list, .structure_format)
  } else {
    matrix_list_str <- matrix_list
  }

  # Match matrix sizes
  expanded_matrices <- .expand_adjacency_matrices(matrix_list_str)
  degree_df <- .compute_degree_df(matrix_list_str)

  .compute_rewiring_scores(expanded_matrices, degree_df)
}


#' @title Node-level significance for DyNet rewiring scores via Monte Carlo permutation
#'
#' @description Generates an empirical null distribution of rewiring scores by repeatedly
#'   randomizing each input network with degree-sequence–preserving edge rewiring, then
#'   computing p-values as the fraction of null scores \eqn{\ge} the observed score.
#'   Adjusted q-values are also returned using a user-chosen method. Optionally computes
#'   pairwise p-values and q-values for every pair of conditions.
#'
#' @import igraph
#' @import dplyr
#' @import tibble
#' @import Matrix
#'
#' @param networks List of (named) adjacency matrices, edge-list data frames, igraph or
#'   tbl_graph objects — identical input format to \code{dynetR()}.
#' @param n_iter Integer. Number of permutation iterations. Default 1000.
#' @param directed Logical. Whether to treat networks as directed when randomizing.
#'   Default TRUE.
#' @param structure_only Logical. If TRUE, binarize matrices before scoring (matches
#'   the \code{structure_only} argument of \code{dynetR()}). Default FALSE.
#' @param pairwise Logical. If TRUE, also compute pairwise rewiring scores and their
#'   p-values / q-values for every pair of input conditions. Default FALSE.
#' @param p_adjust_method Character. Multiple-testing correction method passed to
#'   \code{\link[stats]{p.adjust}}. Any value accepted by \code{p.adjust.methods} is
#'   valid: \code{"BH"} (Benjamini-Hochberg FDR; default), \code{"bonferroni"},
#'   \code{"holm"}, \code{"hochberg"}, \code{"hommel"}, \code{"BY"} (Benjamini-Yekutieli),
#'   or \code{"none"} to skip adjustment.
#' @param n_cores Integer. Number of cores for parallel permutation iterations via
#'   \code{parallel::mclapply}. Default 1 (sequential). On Windows, \code{mclapply}
#'   falls back to 1 core regardless of this setting.
#' @param seed Integer or NULL. Random seed for reproducibility. Default NULL. Note:
#'   parallel runs use per-worker seeds derived from this value and may not reproduce
#'   the exact same sequence as the sequential run.
#' @param verbose Logical. Print progress every 100 iterations (sequential mode only).
#'   Default FALSE.
#'
#' @return A named list with:
#' \describe{
#'   \item{\code{scores}}{Data frame from \code{dynetR()} extended with columns
#'     \code{p_value}, \code{q_value} (adjusted per \code{p_adjust_method}),
#'     \code{p_value_degree_corrected}, and \code{q_value_degree_corrected}.}
#'   \item{\code{pairwise}}{(Only when \code{pairwise = TRUE}) Long data frame with columns
#'     \code{node}, \code{comparison}, \code{observed_score}, \code{p_value}, \code{q_value}.}
#'   \item{\code{n_iter}}{Number of iterations performed.}
#'   \item{\code{seed}}{Seed used (or NA if none was set).}
#' }
#'
#' @export

dynetR_significance <- function(networks,
                                n_iter          = 1000L,
                                directed        = TRUE,
                                structure_only  = FALSE,
                                pairwise        = FALSE,
                                p_adjust_method = "BH",
                                n_cores         = 1L,
                                seed            = NULL,
                                verbose         = FALSE) {

  p_adjust_method <- match.arg(p_adjust_method, p.adjust.methods)

  if (!is.null(seed)) set.seed(seed)

  # ── 1. Preprocess inputs ──────────────────────────────────────────────────
  mat_list <- format_indata(networks)
  if (structure_only) {
    mat_list <- lapply(mat_list, .structure_format)
  }
  expanded  <- .expand_adjacency_matrices(mat_list)
  degree_df <- .compute_degree_df(mat_list)   # computed ONCE; reused every iteration
  net_names <- names(mat_list)
  if (is.null(net_names)) net_names <- paste0("Net", seq_along(mat_list))

  # ── 2. Observed scores ────────────────────────────────────────────────────
  obs_full <- .compute_rewiring_scores(expanded, degree_df,
                                       return_intermediate = pairwise)
  if (pairwise) {
    obs_scores   <- obs_full$scores
    obs_centDist <- obs_full$centDist   # list: one named numeric vector per network
  } else {
    obs_scores <- obs_full
  }

  all_nodes <- obs_scores$name
  n_nodes   <- length(all_nodes)

  # ── 3. Pairwise setup ────────────────────────────────────────────────────
  if (pairwise) {
    pairs       <- combn(seq_along(mat_list), 2, simplify = FALSE)
    pair_labels <- vapply(pairs, function(p)
      paste0(net_names[p[1]], "_vs_", net_names[p[2]]), character(1))

    .pairwise_score <- function(centDist_list, idx_pair) {
      cd_i <- centDist_list[[idx_pair[1]]]
      cd_j <- centDist_list[[idx_pair[2]]]
      cd_i + cd_j
    }

    obs_pairwise_scores <- lapply(pairs, .pairwise_score, centDist_list = obs_centDist)
  }

  # ── 4. Single-iteration worker function ──────────────────────────────────
  .one_iter <- function(iter_seed) {
    if (!is.null(iter_seed)) set.seed(iter_seed)
    rand_list     <- lapply(mat_list, .randomize_network,
                            directed = directed, niter_multiplier = 10)
    rand_expanded <- .expand_adjacency_matrices(rand_list)
    rand_full     <- .compute_rewiring_scores(rand_expanded, degree_df,
                                              return_intermediate = pairwise)
    if (pairwise) {
      rand_scores   <- rand_full$scores
      rand_centDist <- rand_full$centDist
      pw_cols <- lapply(pairs, .pairwise_score, centDist_list = rand_centDist)
    } else {
      rand_scores <- rand_full
      pw_cols     <- NULL
    }
    idx <- match(all_nodes, rand_scores$name)
    list(
      raw = rand_scores$rewiring[idx],
      dc  = rand_scores$degree_corrected_rewiring[idx],
      pw  = pw_cols
    )
  }

  # ── 5. Run iterations ─────────────────────────────────────────────────────
  n_cores <- as.integer(n_cores)
  iter_seeds <- if (!is.null(seed)) seed + seq_len(n_iter) else vector("list", n_iter)

  if (n_cores > 1L) {
    results <- parallel::mclapply(iter_seeds, .one_iter, mc.cores = n_cores)
  } else {
    results <- vector("list", n_iter)
    for (i in seq_len(n_iter)) {
      if (verbose && i %% 100 == 0)
        message("dynetR_significance: iteration ", i, " / ", n_iter)
      results[[i]] <- .one_iter(iter_seeds[[i]])
    }
  }

  # ── 6. Collect null matrices ──────────────────────────────────────────────
  null_raw <- do.call(cbind, lapply(results, `[[`, "raw"))
  null_dc  <- do.call(cbind, lapply(results, `[[`, "dc"))
  rownames(null_raw) <- all_nodes
  rownames(null_dc)  <- all_nodes

  if (pairwise) {
    null_pw <- lapply(seq_along(pairs), function(pi)
      do.call(cbind, lapply(results, function(r) r$pw[[pi]][all_nodes])))
  }

  # ── 7. P-values and BH q-values ───────────────────────────────────────────
  p_raw <- rowMeans(null_raw >= obs_scores$rewiring,                  na.rm = TRUE)
  p_dc  <- rowMeans(null_dc  >= obs_scores$degree_corrected_rewiring, na.rm = TRUE)
  q_raw <- p.adjust(p_raw, method = p_adjust_method)
  q_dc  <- p.adjust(p_dc,  method = p_adjust_method)

  result_scores <- obs_scores |>
    mutate(
      p_value                      = p_raw,
      q_value                      = q_raw,
      p_value_degree_corrected     = p_dc,
      q_value_degree_corrected     = q_dc
    )

  out <- list(scores = result_scores,
              n_iter = n_iter,
              seed   = if (is.null(seed)) NA_integer_ else seed)

  # ── 8. Pairwise p/q-values ────────────────────────────────────────────────
  if (pairwise) {
    pw_rows <- lapply(seq_along(pairs), function(pi) {
      obs_pw <- obs_pairwise_scores[[pi]][all_nodes]
      p_pw   <- rowMeans(null_pw[[pi]] >= obs_pw, na.rm = TRUE)
      q_pw   <- p.adjust(p_pw, method = p_adjust_method)
      tibble::tibble(
        node           = all_nodes,
        comparison     = pair_labels[pi],
        observed_score = obs_pw,
        p_value        = p_pw,
        q_value        = q_pw
      )
    })
    out$pairwise <- dplyr::bind_rows(pw_rows)
  }

  return(out)
}

#' @title Visualize the DyNet rewiring results
#'
#' @description Generates a network graph visualization based on the output from `dynetR`.
#' @import igraph
#' @import ggraph
#' @import tidygraph
#' @import ggplot2
#' @import dplyr
#' @param matrix_list List of adjacency matrices corresponding to the input networks.
#' @param output_dataframe Dataframe output from the `dynetR` function.
#' @param structure_only Logical, should match the `structure_only` parameter used in `dynetR`. Default is FALSE.
#'
#' @export

dynetR_plot <- function(matrix_list, output_dataframe, structure_only = FALSE) {
  # Format input data
  matrix_list <- format_indata(matrix_list)

  # Replace with 0 / 1 if structural rewiring only
  if (structure_only) {
    matrix_list_str <- lapply(matrix_list, .structure_format)
  } else {
    matrix_list_str <- matrix_list
  }

  # Generate unimatrix
  unimatrix <- .union_adjacency_matrices(matrix_list_str)
  unimatrix_graph <- graph_from_adjacency_matrix(unimatrix, mode = "undirected", diag = TRUE) |>
    as_tbl_graph() |>
    activate(nodes) |>
    left_join(output_dataframe, by = "name") |>
    ggraph(layout = "nicely") +
    geom_edge_link(
      start_cap = circle(3, 'mm'),
      end_cap = circle(3, "mm"),
      width = 1,
      alpha = 0.5
    ) +
    geom_node_point(aes(fill = rewiring, size = degree), shape = 21) +
    scale_fill_distiller(palette = "Reds", direction = 1) +
    geom_node_text(
      aes(label = name, size = degree),
      repel = TRUE,
      fontface = "bold",
      max.overlaps = 15
    ) +
    scale_size_continuous(range = c(6, 10)) +
    theme_graph()

  return(unimatrix_graph)
}

#' @title Small multiples plot
#'
#' @description Draws a small multiples plot from the input networks focussing on a specific node
#' @import igraph
#' @import readr
#' @import ggplot2
#' @import ggraph
#' @import tidygraph
#' @import dplyr
#' @import tibble
#' @import Matrix
#' @param matrix_list List of named adjacency matrices corresponding to the input networks, same as input to dynetR.
#' @param focus_node Character, name of node to focus visualisation on.
#'
#' @export

small_multiples_plot <- function(input_list, focus_node){
  # data for small multiples plot
  # format input data
  matrix_list <- format_indata(input_list)

  ellist<-lapply(matrix_list, .indata_to_edgelist)
  long_df <- dplyr::bind_rows(ellist,.id = "id") |> dplyr::tibble()

  small_mult<-long_df |> dplyr::select(from,to, id) |> distinct() |> dplyr::filter(paste0(focus_node) %in% from | paste0(focus_node) %in% to) |>
    as_tbl_graph() |>
    activate(nodes) |>
    mutate(hl = case_when(
      name %in% paste0(focus_node) ~ 'Focus node',
      TRUE ~ 'Other nodes'
    )
    ) |>
    ggraph(layout='linear', circular = T)+
    geom_edge_link(start_cap = circle(3, 'mm'), end_cap = circle(3, "mm"), width =1, alpha = 0.3)+
    geom_node_point(aes(color = hl), size =5, show.legend = T)+
    geom_node_text(aes(label = name), repel = T, fontface = "bold", max.overlaps = 15) +
    facet_edges(~id)+
    theme_graph()+labs(color = 'Node status')
  return(small_mult)
}


#' Helper function to pretty print matrices
.mat_op_print <- function(..., width = 0) {
  # get arguments
  args <- list(...)
  chars <- sapply(args, is.character)

  # auxilliary function to create character of n spaces
  spaces <- function(n) paste(rep(" ", n), collapse = "")

  # convert vectors to row matrix
  vecs <- sapply(args, is.vector)
  args[vecs & !chars] <- lapply(args[vecs & !chars], function(v) matrix(v, ncol = 1))

  # convert all non-characters to character with format
  args[!chars] <- lapply(args[!chars], format, width = width)

  # print names as the first line, if present
  arg_names <- names(args)
  if (!is.null(arg_names)) {
    get_title <- function(x, name) {
      if (is.matrix(x)) {
        paste0(name, spaces(sum(nchar(x[1, ])) + ncol(x) - 1 - nchar(name)))
      } else {
        spaces(nchar(x))
      }
    }
    cat(mapply(get_title, args, arg_names), "\n")
  }

  # auxiliary function to create the lines
  get_line <- function(x, n) {
    if (is.matrix(x)) {
      if (nrow(x) < n) {
        spaces(sum(nchar(x[1, ])) + ncol(x) - 1)
      } else {
        paste(x[n, ], collapse = " ")
      }
    } else if (n == 1) {
      x
    } else {
      spaces(nchar(x))
    }
  }

  # print as many lines as needed for the matrix with most rows
  N <- max(sapply(args[!chars], nrow))
  for (n in 1:N) {
    cat(sapply(args, get_line, n), "\n")
  }
}

#' Helper function that prints the results of the subsequent steps
.dynet_verbose <- function(matrix_list) {
  # format input data
  matrix_list <- format_indata(matrix_list)

  # Match matrix sizes
  expanded_matrices <- .expand_adjacency_matrices(matrix_list)

  # get union matrix

  matrix_union <- .union_adjacency_matrices(expanded_matrices)

  # Calculate non-zero means of matrices
  nonZeroMean <- .sumMatrices(expanded_matrices)

  # Calculate standardised weights (divided by non-zero means)
  standardised <- lapply(expanded_matrices, "/", nonZeroMean)
  standardisedNoNan <- lapply(standardised, function(x) replace(x, is.nan(x), 0))

  # Calculate centroids of all node states by summing and subsequent division
  standardSums <- Reduce("+", standardisedNoNan)
  centroid <- standardSums / length(standardisedNoNan)

  # Subtract centroid values from standardised matrix
  standardMinusCentroid <- lapply(standardisedNoNan, "-", centroid)
  sqr <- .squareMatrices(standardMinusCentroid)
  centDist <- .centroidDistance(sqr)
  centDistBound <- do.call(rbind, centDist)
  # print(centDistBound)
  centFinalDist <- colSums(centDistBound)
  print(length(centFinalDist))
  rewiring <- centFinalDist / (length(centDist) - 1)
  print("Input matrices:")
  .mat_op_print(A = matrix_list[[1]], " , ", B = matrix_list[[2]])
  print("Non-zero mean:")
  print(nonZeroMean)
  print("Standardised weights (divided by non-zero means:")
  .mat_op_print(A = standardisedNoNan[[1]], " , ", B = standardisedNoNan[[2]])
  print("Centroid:")
  print(centroid)
  print("Standardised weight - centroid")
  .mat_op_print(A = standardMinusCentroid[[1]], " , ", B = standardMinusCentroid[[2]])
  print("Square of standard minus centroid")
  .mat_op_print(A = sqr[[1]], " , ", B = sqr[[2]])
  print("Square root of standard minus centroid row sums")
  print(centDistBound)
  print("Final rewiring value")
  print(rewiring)
  print(matrix_union)
}

# Jaccard Index Function for Adjacency Matrices
.jaccard_index <- function(adj1, adj2) {
  # adj1 and adj2 are adjacency matrices

  # Extract edges from the adjacency matrices
  # For directed graphs, consider all non-zero entries
  edges1 <- which(adj1 != 0, arr.ind = TRUE)
  edges2 <- which(adj2 != 0, arr.ind = TRUE)

  # Convert edges to data frames for merging
  set1 <- as.data.frame(edges1)
  set2 <- as.data.frame(edges2)

  # Combine both sets of edges and find unique edges (Union)
  union_set <- unique(rbind(set1, set2))

  # Find the intersection of edges
  intersection_set <- merge(set1, set2, by = c("row", "col"))

  # Jaccard index calculation
  jaccard <- nrow(intersection_set) / nrow(union_set)

  return(jaccard)
}

#' @title Calculate Jaccard Indices
#'
#' @description Function to Calculate Jaccard Indices for a List of Networks. Returns a matrix containing the Jaccard indexes for all compared networks in the list.
#' @import igraph
#' @import readr
#' @import ggplot2
#' @import ggraph
#' @import tidygraph
#' @import dplyr
#' @import tibble
#' @import Matrix
#' @param networks List of named adjacency matrices corresponding to the input networks, same as input to dynetR.
#' @param focus_node Character, name of node to focus visualisation on.
#'
#' @export

calculate_jaccard_indices <- function(networks) {

  networks <- format_indata(networks)
  # networks: a list of adjacency matrices
  n <- length(networks)

  # Get or assign network names
  if (is.null(names(networks))) {
    network_names <- paste0("Network_", seq_len(n))
  } else {
    network_names <- names(networks)
  }

  # Initialize a matrix to store Jaccard indices
  jaccard_matrix <- matrix(0, n, n)
  colnames(jaccard_matrix) <- network_names
  rownames(jaccard_matrix) <- network_names

  # Calculate Jaccard indices for all pairs of networks
  for (i in 1:n) {
    for (j in i:n) {
      jaccard_value <- .jaccard_index(networks[[i]], networks[[j]])
      jaccard_matrix[i, j] <- jaccard_value
      jaccard_matrix[j, i] <- jaccard_value  # Since the Jaccard index is symmetric
    }
  }

  return(jaccard_matrix)
}

#' @title Compare targeting
#'
#' @description Function to Calculate differential targeting values for all compared networks. Returns a dataframe containing the compared node, networks, targeting values, absoulte difference and log2 ratio of targeting values.
#' @import igraph
#' @import readr
#' @import ggplot2
#' @import ggraph
#' @import tidygraph
#' @import dplyr
#' @import tibble
#' @import Matrix
#' @param input_list List of named adjacency matrices corresponding to the input networks, same as input to dynetR.
#'
#' @export

compare_targeting <- function(input_list) {
  # Format the input data
  formatted_matrix_list <- format_indata(input_list)

  # Compute targeting measures for each network
  targeting_list <- lapply(seq_along(formatted_matrix_list), function(i) {
    adj_matrix <- formatted_matrix_list[[i]]

    # Convert adjacency matrix to an igraph object with weights
    g <- graph_from_adjacency_matrix(adj_matrix, mode = 'directed', weighted = TRUE)

    # Convert to tbl_graph for tidygraph operations
    g_tbl <- as_tbl_graph(g)

    # Compute in-degree centrality (targeting) with weights
    df <- g_tbl %>%
      activate(nodes) %>%
      mutate(targeting = centrality_degree(mode = 'in', weights = weight)) %>%
      as_tibble()

    # Add network identifier
    df$network_id <- i
    return(df[, c('name', 'targeting', 'network_id')])
  })

  # Combine all targeting data into one long dataframe
  all_targeting <- bind_rows(targeting_list)

  # Generate all unique pairs of networks
  pairs <- combn(unique(all_targeting$network_id), 2)

  # Initialize a list to store comparison results
  result_list <- list()

  # Compare targeting measures for each pair of networks
  for (k in 1:ncol(pairs)) {
    i <- pairs[1, k]
    j <- pairs[2, k]

    df_i <- all_targeting %>% filter(network_id == i)
    df_j <- all_targeting %>% filter(network_id == j)

    # Merge dataframes on 'name' and calculate deltaTargeting
    df_compare <- df_i %>%
      inner_join(df_j, by = 'name', suffix = c('_i', '_j')) %>%
      mutate(
        compared_networks = paste0(i, '_vs_', j),
        targetingNet1 = paste0(targeting_i),
        targetingNet2 = paste0(targeting_j),
        deltaTargeting = abs(targeting_i - targeting_j),
        log2TargetingFC = log2(targeting_i / targeting_j)
      ) %>%
      select(name, compared_networks,targetingNet1,targetingNet2, deltaTargeting,,log2TargetingFC)

    # Store the comparison result
    result_list[[k]] <- df_compare
  }

  # Combine all comparison results into one dataframe
  all_results <- bind_rows(result_list)

  return(all_results)
}

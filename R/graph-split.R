#' Create Train/Validation/Test Split for Graph Data
#'
#' Creates splits for graph neural networks following rsample's structure.
#' Unlike traditional ML, GCNs use the full graph during training but only compute
#' loss on labeled (training) nodes.
#'
#' @param data Data frame or adjacency matrix. The full dataset to split.
#' @param prop Numeric vector of length 2 or 3. Proportions for splits.
#'   - Length 2: c(train, test) - creates train/test split
#'   - Length 3: c(train, val, test) - creates train/val/test split
#'   Must sum to 1.0. Default: c(0.6, 0.2, 0.2)
#' @param seed Integer or NULL. Random seed for reproducibility. Default: NULL
#'
#' @return A graph_split object (list) containing:
#'   - data: Original data
#'   - train_id: Integer vector of training indices
#'   - val_id: Integer vector of validation indices (or NULL if length(prop) == 2)
#'   - test_id: Integer vector of test indices
#'
#' @details
#' The proportions must sum to 1.0. The function creates non-overlapping splits where
#' each row belongs to exactly one split.
#'
#' For GCN training, you use:
#' - Full X and A_sparse for all forward passes
#' - IDs to select which predictions to use for loss computation
#'
#' @examples
#' \dontrun{
#' # Standard 60/20/20 split
#' split <- graph_split(A_sparse, seed = 42)
#'
#' # Custom split (70/15/15)
#' split <- graph_split(A_sparse, prop = c(0.7, 0.15, 0.15))
#'
#' # Two-way split (80/20 train/test)
#' split <- graph_split(A_sparse, prop = c(0.8, 0.2))
#'
#' # Use in training
#' predictions <- model(X, A_sparse)
#' train_loss <- nnf_mse_loss(predictions[split$train_id], y[split$train_id])
#' }
graph_split <- function(data, prop = c(0.6, 0.2, 0.2), seed = NULL) {
  # Validate prop
  if (!length(prop) %in% c(2, 3)) {
    stop("prop must be a numeric vector of length 2 or 3")
  }

  if (abs(sum(prop) - 1.0) > 1e-6) {
    stop(sprintf("prop must sum to 1.0, got %.2f", sum(prop)))
  }

  if (any(prop < 0)) {
    stop("All proportions must be non-negative")
  }

  # Get number of observations
  n <- NROW(data)

  # Set seed if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Random permutation of indices
  indices <- sample(1:n)

  if (length(prop) == 2) {
    # Two-way split: train/test
    n_train <- floor(n * prop[1])

    train_id <- indices[1:n_train]
    val_id <- NULL
    test_id <- indices[(n_train + 1):n]
  } else {
    # Three-way split: train/val/test
    n_train <- floor(n * prop[1])
    n_val <- floor(n * prop[2])

    train_id <- indices[1:n_train]
    val_id <- indices[(n_train + 1):(n_train + n_val)]
    test_id <- indices[(n_train + n_val + 1):n]
  }

  structure(
    list(
      data = data,
      train_id = train_id,
      val_id = val_id,
      test_id = test_id
    ),
    prop = prop,
    class = "graph_split"
  )
}

#' @export
print.graph_split <- function(x, ...) {
  cat("<graph_split>\n")
  vctrs::obj_str(unclass(x))
}

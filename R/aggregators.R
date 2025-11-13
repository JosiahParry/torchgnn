forward <- S7::new_generic("forward", "x", function(x, adj, tensor, ...) {
  if (!inherits(adj, "torch_tensor")) {
    cli::cli_abort("{.arg adj} must be a torch tensor")
  }

  if (!adj$is_sparse()) {
    cli::cli_abort("{.arg adj} must be a sparse tensor")
  }

  S7::S7_dispatch()
})

#' Message Passing Aggregators
#'
#' @description
#' Aggregators combine neighbor node features in graph neural networks.
#' Each aggregator implements a different reduction operation (sum, mean, max, etc.)
#' to aggregate features from neighboring nodes.
#'
#' @param adj Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining
#'   graph structure. Must be a sparse COO tensor.
#' @param tensor Torch tensor `n_nodes x n_features`. Node feature matrix.
#'   Can be dense or sparse.
#' @param ... Additional arguments passed to specific aggregator methods.
#'
#' @details
#' Available aggregators:
#' - `SumAggregator()`: Sum of neighbor features
#' - `MeanAggregator()`: Mean of neighbor features (with row normalization)
#' - `MaxAggregator()`: Element-wise maximum of neighbor features
#' - `MinAggregator()`: Element-wise minimum of neighbor features
#' - `ProductAggregator()`: Element-wise product of neighbor features
#' - `VarAggregator()`: Variance of neighbor features
#' - `StdAggregator()`: Standard deviation of neighbor features
#' - `LSTMAggregator()`: Not-imlemented
#' - `SoftmaxAggregator()`: Not-imlemented
#'
#' @rdname aggregator
#' @export
Aggregator <- S7::new_class(
  "Aggregator",
  properties = list(
    name = S7::class_character,
    learnable = S7::class_logical
  ),
  abstract = TRUE
)

S7::method(forward, Aggregator) <- function(x, adj, tensor, ...) {
  stop("forward() must be implemented by subclass")
}


#' @export
#' @rdname aggregator
SumAggregator <- S7::new_class(
  "SumAggregator",
  parent = Aggregator,
  constructor = function() {
    S7::new_object(
      S7::S7_object(),
      name = "sum",
      learnable = FALSE
    )
  }
)

S7::method(forward, SumAggregator) <- function(x, adj, tensor, ...) {
  torch_mm(adj, tensor)
}

#' @export
#' @rdname aggregator
MeanAggregator <- S7::new_class(
  "MeanAggregator",
  parent = Aggregator,
  constructor = function() {
    S7::new_object(
      S7::S7_object(),
      name = "mean",
      learnable = FALSE
    )
  }
)

S7::method(forward, MeanAggregator) <- function(x, adj, tensor, ...) {
  adj_norm <- adj_row_normalize(adj)
  torch_mm(adj_norm, tensor)
}

#' @export
#' @rdname aggregator
MaxAggregator <- S7::new_class(
  "MaxAggregator",
  parent = Aggregator,
  constructor = function() {
    S7::new_object(
      S7::S7_object(),
      name = "max",
      learnable = FALSE
    )
  }
)

S7::method(forward, MaxAggregator) <- function(x, adj, tensor, ...) {
  n_nodes <- tensor$size(1)
  n_features <- tensor$size(2)
  idx <- adj$indices() + 1L

  target_idx <- idx[1, ]
  source_idx <- idx[2, ]
  src_features <- tensor[source_idx, ]

  result <- torch_full(c(n_nodes, n_features), -Inf)
  index_expanded <- target_idx$unsqueeze(2)$expand(c(-1, n_features))

  result$scatter_reduce_(
    1,
    index_expanded,
    src_features,
    "amax",
    include_self = FALSE
  )
}

#' @export
#' @rdname aggregator
MinAggregator <- S7::new_class(
  "MinAggregator",
  parent = Aggregator,
  constructor = function() {
    S7::new_object(
      S7::S7_object(),
      name = "min",
      learnable = FALSE
    )
  }
)

S7::method(forward, MinAggregator) <- function(x, adj, tensor, ...) {
  n_nodes <- tensor$size(1)
  n_features <- tensor$size(2)
  idx <- adj$indices() + 1L

  target_idx <- idx[1, ]
  source_idx <- idx[2, ]
  src_features <- tensor[source_idx, ]

  result <- torch_full(c(n_nodes, n_features), Inf)
  index_expanded <- target_idx$unsqueeze(2)$expand(c(-1, n_features))

  result$scatter_reduce_(
    1,
    index_expanded,
    src_features,
    "amin",
    include_self = FALSE
  )
}

#' @export
#' @rdname aggregator
ProductAggregator <- S7::new_class(
  "ProductAggregator",
  parent = Aggregator,
  constructor = function() {
    S7::new_object(
      S7::S7_object(),
      name = "prod",
      learnable = FALSE
    )
  }
)

S7::method(forward, ProductAggregator) <- function(x, adj, tensor, ...) {
  n_nodes <- tensor$size(1)
  n_features <- tensor$size(2)
  idx <- adj$indices() + 1L

  target_idx <- idx[1, ]
  source_idx <- idx[2, ]
  src_features <- tensor[source_idx, ]

  result <- torch_ones(c(n_nodes, n_features))
  index_expanded <- target_idx$unsqueeze(2)$expand(c(-1, n_features))

  result$scatter_reduce_(
    1,
    index_expanded,
    src_features,
    "prod",
    include_self = FALSE
  )
}

#' @export
#' @rdname aggregator
VarAggregator <- S7::new_class(
  "VarAggregator",
  parent = Aggregator,
  constructor = function() {
    S7::new_object(
      S7::S7_object(),
      name = "var",
      learnable = FALSE
    )
  }
)

S7::method(forward, VarAggregator) <- function(x, adj, tensor, ...) {
  n_nodes <- tensor$size(1)
  n_features <- tensor$size(2)
  idx <- adj$indices() + 1L

  target_idx <- idx[1, ]
  source_idx <- idx[2, ]
  src_features <- tensor[source_idx, ]

  index_expanded <- target_idx$unsqueeze(2)$expand(c(-1, n_features))

  neighbor_mean <- torch_zeros(c(n_nodes, n_features))
  neighbor_mean$scatter_reduce_(
    1,
    index_expanded,
    src_features,
    "mean",
    include_self = FALSE
  )

  squared_features <- src_features$pow(2)
  mean_of_squares <- torch_zeros(c(n_nodes, n_features))
  mean_of_squares$scatter_reduce_(
    1,
    index_expanded,
    squared_features,
    "mean",
    include_self = FALSE
  )

  mean_of_squares - neighbor_mean$pow(2)
}

#' @export
#' @rdname aggregator
StdAggregator <- S7::new_class(
  "StdAggregator",
  parent = Aggregator,
  constructor = function() {
    S7::new_object(
      S7::S7_object(),
      name = "std",
      learnable = FALSE
    )
  }
)

S7::method(forward, StdAggregator) <- function(x, adj, tensor, ...) {
  n_nodes <- tensor$size(1)
  n_features <- tensor$size(2)
  idx <- adj$indices() + 1L

  target_idx <- idx[1, ]
  source_idx <- idx[2, ]
  src_features <- tensor[source_idx, ]

  index_expanded <- target_idx$unsqueeze(2)$expand(c(-1, n_features))

  neighbor_mean <- torch_zeros(c(n_nodes, n_features))
  neighbor_mean$scatter_reduce_(
    1,
    index_expanded,
    src_features,
    "mean",
    include_self = FALSE
  )

  squared_features <- src_features$pow(2)
  mean_of_squares <- torch_zeros(c(n_nodes, n_features))
  mean_of_squares$scatter_reduce_(
    1,
    index_expanded,
    squared_features,
    "mean",
    include_self = FALSE
  )

  variance <- mean_of_squares - neighbor_mean$pow(2)
  variance$sqrt()
}

#' @export
#' @rdname aggregator
LSTMAggregator <- S7::new_class(
  "LSTMAggregator",
  parent = Aggregator,
  properties = list(
    lstm = S7::class_any
  ),
  constructor = function(in_features, hidden_features = NULL) {
    if (is.null(hidden_features)) {
      hidden_features <- in_features
    }

    lstm <- torch::nn_lstm(
      input_size = in_features,
      hidden_size = hidden_features,
      batch_first = TRUE
    )

    S7::new_object(
      S7::S7_object(),
      name = "lstm",
      learnable = TRUE,
      lstm = lstm
    )
  }
)

S7::method(forward, LSTMAggregator) <- function(x, adj, tensor, ...) {
  stop("LSTMAggregator not yet implemented for sparse adjacency")
}

#' @export
#' @rdname aggregator
SoftmaxAggregator <- S7::new_class(
  "SoftmaxAggregator",
  parent = Aggregator,
  properties = list(
    scorer = S7::class_any,
    in_features = S7::class_integer
  ),
  constructor = function(in_features, learn = TRUE) {
    scorer <- if (learn) {
      torch::nn_linear(in_features, 1)
    } else {
      NULL
    }

    S7::new_object(
      S7::S7_object(),
      name = "softmax",
      learnable = learn,
      scorer = scorer,
      in_features = as.integer(in_features)
    )
  }
)

S7::method(forward, SoftmaxAggregator) <- function(x, tensor, dim = 1, ...) {
  if (!is.null(x@scorer)) {
    scores <- x@scorer(tensor)$squeeze(-1)
  } else {
    scores <- torch::torch_ones(tensor$size()[1:2])
  }

  weights <- torch::nnf_softmax(scores, dim = dim)

  if (length(weights$size()) < length(tensor$size())) {
    weights <- weights$unsqueeze(-1)
  }

  torch::torch_sum(weights * tensor, dim = dim)
}

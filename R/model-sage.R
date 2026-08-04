#' Multi-layer GraphSAGE Model (Hamilton et al. 2017)
#'
#' Stacks multiple GraphSAGE layers with configurable aggregation functions.
#'
#' @details
#' Architecture:
#' - L hidden SAGE layers with configurable activation
#' - 1 output SAGE layer with optional output activation
#' - Total layers = length(hidden_dims) + 1
#'
#' Each layer aggregates neighbor features using the specified aggregator, then
#' combines with self features via concatenation or addition.
#'
#' @param in_features Integer. Number of input features per node
#' @param hidden_dims Integer vector. Dimensions of hidden layers (length = L)
#' @param out_features Integer. Number of output features (typically 1 for regression)
#' @param aggregator Aggregator S7 object. Aggregation function for all layers.
#'   Default: `MeanAggregator()`
#' @param activation Function. Activation for hidden layers. Default: nnf_relu
#' @param out_activation Function or NULL. Activation for output layer. Default: NULL
#' @param dropout Numeric. Dropout rate (0-1) applied after each hidden layer. Default: 0
#' @param concat Logical. If TRUE, concatenates self and neighbor features. If FALSE,
#'   adds them. Default: TRUE
#' @param norm `nn_module` generator or NULL. Normalization applied after each
#'   hidden layer, before the activation. Called once per hidden layer with that
#'   layer's output dimension. Default: NULL
#'
#' @section Forward pass:
#' @param x Tensor `n_nodes x in_features`. Node feature matrix (dense or sparse)
#' @param adj Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining graph
#'   structure. Must be a sparse COO tensor.
#' @param batch Tensor or NULL. Batch vector assigning each node to a graph, using
#'   1-based graph indices. Passed to `norm`. If NULL, all nodes are treated as a
#'   single graph.
#'
#' @return Tensor `n_nodes x out_features`. Final predictions
#'
#' @examples
#' \dontrun{
#' # Binary classification with sigmoid and mean aggregation
#' model <- model_sage(14, c(56, 56), 1, output_activation = nnf_sigmoid)
#'
#' # Multi-class with softmax and max aggregation
#' model <- model_sage(
#'   14,
#'   c(32, 32),
#'   10,
#'   aggregator = MaxAggregator(),
#'   output_activation = function(x) nnf_softmax(x, dim = -1)
#' )
#'
#' # Regression with sum aggregation
#' model <- model_sage(14, c(64, 64), 1, aggregator = SumAggregator())
#'
#' # With dropout and custom activation
#' model <- model_sage(
#'   14,
#'   c(56, 56),
#'   1,
#'   activation = nnf_tanh,
#'   dropout = 0.5
#' )
#'
#' # With normalization after each hidden layer
#' model <- model_sage(14, c(56, 32), 1, norm = layer_layer_norm)
#'
#' model <- model_sage(
#'   14,
#'   c(56, 32),
#'   1,
#'   norm = \(d) layer_layer_norm(d, mode = "node")
#' )
#' }
#'
#' @references
#' Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning
#' on large graphs. Advances in Neural Information Processing Systems, 30.
#' <doi:10.48550/arXiv.1706.02216>
#' @export
model_sage <- nn_module(
  "SAGEModel",

  initialize = function(
    in_features,
    hidden_dims,
    out_features,
    aggregator = MeanAggregator(),
    activation = nnf_relu,
    out_activation = NULL,
    dropout = 0,
    concat = TRUE,
    norm = NULL
  ) {
    layers <- list()

    # Input to first hidden layer
    layers[[1]] <- layer_sage(
      in_features,
      hidden_dims[1],
      aggregator = aggregator,
      concat = concat
    )

    # Additional hidden layers
    if (length(hidden_dims) > 1) {
      for (i in 2:length(hidden_dims)) {
        layers[[i]] <- layer_sage(
          hidden_dims[i - 1],
          hidden_dims[i],
          aggregator = aggregator,
          concat = concat
        )
      }
    }

    # Output layer
    layers[[length(layers) + 1]] <- layer_sage(
      hidden_dims[length(hidden_dims)],
      out_features,
      aggregator = aggregator,
      concat = concat
    )

    self$layers <- nn_module_list(layers)

    if (is.null(norm)) {
      self$norms <- NULL
    } else {
      self$norms <- nn_module_list(lapply(hidden_dims, norm))
    }

    self$activation <- activation
    self$out_activation <- out_activation
    self$dropout_rate <- dropout
  },

  normalize = function(x, i, batch) {
    if (is.null(self$norms)) {
      return(x)
    }
    self$norms[[i]](x, batch)
  },

  regularize = function(x) {
    if (!self$training || self$dropout_rate <= 0) {
      return(x)
    }
    nnf_dropout(x, p = self$dropout_rate)
  },

  forward = function(x, adj, batch = NULL) {
    n_hidden <- length(self$layers) - 1

    for (i in seq_len(n_hidden)) {
      x <- self$layers[[i]](x, adj)
      x <- self$normalize(x, i, batch)
      x <- self$activation(x)
      x <- self$regularize(x)
    }

    x <- self$layers[[n_hidden + 1]](x, adj)

    if (is.null(self$out_activation)) {
      return(x)
    }
    self$out_activation(x)
  }
)

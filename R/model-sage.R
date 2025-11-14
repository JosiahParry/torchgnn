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
#'
#' @section Forward pass:
#' @param x Tensor `n_nodes x in_features`. Node feature matrix (dense or sparse)
#' @param adj Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining graph
#'   structure. Must be a sparse COO tensor.
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
    concat = TRUE
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
    self$activation <- activation
    self$out_activation <- out_activation
    self$dropout_rate <- dropout
  },

  forward = function(x, adj) {
    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](x, adj)

      if (i < length(self$layers)) {
        x <- self$activation(x)
        if (self$training && self$dropout_rate > 0) {
          x <- nnf_dropout(x, p = self$dropout_rate)
        }
      } else if (!is.null(self$out_activation)) {
        x <- self$out_activation(x)
      }
    }
    x
  }
)

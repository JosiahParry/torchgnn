#' Multi-layer Graph Attention Network Model (Veličković et al. 2018)
#'
#' Stacks multiple GAT layers with multi-head attention.
#'
#' @details
#' Architecture:
#' - L hidden GAT layers with configurable activation
#' - 1 output GAT layer with optional output activation
#' - Total layers = length(hidden_dims) + 1
#'
#' Each layer uses multi-head attention to learn importance weights for neighbor
#' aggregation. Hidden layers typically concatenate attention heads, while the
#' output layer averages them.
#'
#' @param in_features Integer. Number of input features per node
#' @param hidden_dims Integer vector. Dimensions of hidden layers (length = L)
#' @param out_features Integer. Number of output features (typically 1 for regression)
#' @param heads Integer. Number of attention heads for hidden layers. Default: 8
#' @param out_heads Integer. Number of attention heads for output layer. Default: 1
#' @param activation Function. Activation for hidden layers. Default: [nnf_elu]
#' @param out_activation Function or NULL. Activation for output layer. Default: NULL
#' @param dropout Numeric. Dropout rate (0-1) applied to attention and features. Default: 0.6
#' @param att_dropout Numeric. Dropout rate for attention coefficients. Default: 0.6
#' @param negative_slope Numeric. Negative slope for LeakyReLU in attention. Default: 0.2
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
#' # Binary classification with 8-head attention
#' model <- gat_model(14, c(8, 8), 1, output_activation = nnf_sigmoid)
#'
#' # Multi-class with 4 heads
#' model <- gat_model(
#'   14,
#'   c(16, 16),
#'   3,
#'   heads = 4,
#'   output_activation = function(x) nnf_softmax(x, dim = -1)
#' )
#'
#' # Regression with custom dropout
#' model <- gat_model(14, c(32, 32), 1, dropout = 0.5, att_dropout = 0.5)
#' }
#'
#' @references
#' Veličković P., Cucurull, G., Casanova, A., Romero, A., Li, P., & Bengio, Y. (2018).
#' Graph Attention Networks. International Conference on Learning Representations.
#' <doi:10.48550/arXiv.1710.10903>
#' @export
model_gat <- nn_module(
  "GATModel",

  initialize = function(
    in_features,
    hidden_dims,
    out_features,
    heads = 8,
    out_heads = 1,
    activation = nnf_elu,
    out_activation = NULL,
    dropout = 0.6,
    att_dropout = 0.6,
    negative_slope = 0.2
  ) {
    layers <- list()

    # Input to first hidden layer
    layers[[1]] <- layer_gat(
      in_features,
      hidden_dims[1],
      heads = heads,
      concat = TRUE,
      dropout = att_dropout,
      negative_slope = negative_slope
    )

    # Additional hidden layers
    if (length(hidden_dims) > 1) {
      for (i in 2:length(hidden_dims)) {
        layers[[i]] <- layer_gat(
          hidden_dims[i - 1] * heads,
          hidden_dims[i],
          heads = heads,
          concat = TRUE,
          dropout = att_dropout,
          negative_slope = negative_slope
        )
      }
    }

    # Output layer (average heads instead of concatenate)
    layers[[length(layers) + 1]] <- layer_gat(
      hidden_dims[length(hidden_dims)] * heads,
      out_features,
      heads = out_heads,
      concat = FALSE,
      dropout = att_dropout,
      negative_slope = negative_slope
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

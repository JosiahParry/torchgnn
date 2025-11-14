#' Multi-layer Graph Isomorphism Network Model (Xu et al. 2019)
#'
#' Stacks multiple GIN layers with MLP transformations.
#'
#' @details
#' Architecture:
#' - L hidden GIN layers with MLP transformations
#' - 1 output GIN layer with optional output activation
#' - Total layers = length(hidden_dims) + 1
#'
#' Each layer aggregates neighbor features via summation, adds weighted self
#' features, and applies a 2-layer MLP transformation.
#'
#' @param in_features Integer. Number of input features per node
#' @param hidden_dims Integer vector. Dimensions of hidden layers (length = L)
#' @param out_features Integer. Number of output features (typically 1 for regression)
#' @param eps Numeric. Initial value for epsilon. Default: 0
#' @param learn_eps Logical. Whether to learn epsilon parameters. Default: FALSE
#' @param dropout Numeric. Dropout rate (0-1) applied between layers. Default: 0.5
#' @param out_activation Function or NULL. Activation for output layer. Default: NULL
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
#' # Binary classification
#' model <- model_gin(14, c(64, 64), 1, output_activation = nnf_sigmoid)
#'
#' # Multi-class classification
#' model <- model_gin(
#'   14,
#'   c(64, 64),
#'   3,
#'   output_activation = function(x) nnf_softmax(x, dim = -1)
#' )
#'
#' # With learnable epsilon
#' model <- model_gin(14, c(128), 1, learn_eps = TRUE)
#' }
#'
#' @references
#' Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph
#' Neural Networks? International Conference on Learning Representations.
#' <doi:10.48550/arXiv.1810.00826>
#' @export
model_gin <- nn_module(
  "GINModel",

  initialize = function(
    in_features,
    hidden_dims,
    out_features,
    eps = 0,
    learn_eps = FALSE,
    dropout = 0.5,
    out_activation = NULL
  ) {
    layers <- list()

    layers[[1]] <- layer_gin(
      in_features,
      hidden_dims[1],
      eps = eps,
      learn_eps = learn_eps
    )

    if (length(hidden_dims) > 1) {
      for (i in 2:length(hidden_dims)) {
        layers[[i]] <- layer_gin(
          hidden_dims[i - 1],
          hidden_dims[i],
          eps = eps,
          learn_eps = learn_eps
        )
      }
    }

    layers[[length(layers) + 1]] <- layer_gin(
      hidden_dims[length(hidden_dims)],
      out_features,
      eps = eps,
      learn_eps = learn_eps
    )

    self$layers <- nn_module_list(layers)
    self$out_activation <- out_activation
    self$dropout_rate <- dropout
  },

  forward = function(x, adj) {
    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](x, adj)

      if (i < length(self$layers)) {
        x <- nnf_relu(x)
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

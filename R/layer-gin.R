#' Graph Isomorphism Network Layer (Xu et al. 2019)
#'
#' @description
#' Implements the Graph Isomorphism Network (GIN) layer:
#'
#' \deqn{\mathbf{h}_i^{(k)} = \text{MLP}^{(k)}\left((1 + \epsilon^{(k)}) \cdot \mathbf{h}_i^{(k-1)} + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(k-1)}\right)}
#'
#' This layer:
#' 1. Aggregates neighbor features via summation
#' 2. Adds weighted self features using learnable epsilon
#' 3. Applies MLP transformation
#'
#' Parameters:
#' - MLP: Multi-layer perceptron (typically 2 layers)
#' - epsilon: Learnable or fixed weight for self features
#'
#' @details
#' The MLP is constructed as a sequence of Linear-BatchNorm-ReLU-Linear layers.
#' The epsilon parameter can be learned or fixed at 0.
#'
#' @param in_features Integer. Number of input features per node
#' @param out_features Integer. Number of output features per node
#' @param eps Numeric. Initial value for epsilon. Default: 0
#' @param learn_eps Logical. Whether to learn epsilon parameter. Default: FALSE
#'
#' @section Forward pass:
#' @param x Tensor `n_nodes x in_features`. Node feature matrix
#' @param adj Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining graph
#'   structure. Must be a sparse COO tensor.
#'
#' @return Tensor `n_nodes x out_features`. Transformed node features
#'
#' @references
#' Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph
#' Neural Networks? International Conference on Learning Representations.
#' <doi:10.48550/arXiv.1810.00826>
#' @export
layer_gin <- nn_module(
  "GINLayer",

  initialize = function(
    in_features,
    out_features,
    eps = 0,
    learn_eps = FALSE
  ) {
    self$in_features <- in_features
    self$out_features <- out_features

    if (learn_eps) {
      self$eps <- nn_parameter(torch_tensor(eps))
    } else {
      self$eps <- eps
    }

    self$mlp <- nn_sequential(
      nn_linear(in_features, out_features),
      nn_batch_norm1d(out_features),
      nn_relu(),
      nn_linear(out_features, out_features)
    )
  },

  forward = function(x, adj) {
    neighbor_sum <- torch_mm(adj, x)
    self_weighted <- (1 + self$eps) * x
    combined <- self_weighted + neighbor_sum
    self$mlp(combined)
  }
)

#' Layer Normalization (Ba et al. 2016)
#'
#' @description
#' Applies layer normalization to node features:
#'
#' \deqn{\mathbf{x}' = \frac{\mathbf{x} - \mathrm{E}[\mathbf{x}]}{\sqrt{\mathrm{Var}[\mathbf{x}] + \epsilon}} \odot \gamma + \beta}
#'
#' The mean and variance are computed over the elements selected by `mode`:
#'
#' - `"graph"`: statistics are computed across all nodes *and* all channels of a
#'   graph, giving a single mean and variance per graph. When `batch` is supplied,
#'   each graph in the mini-batch is normalized independently.
#' - `"node"`: statistics are computed across the channels of each node
#'   independently, giving one mean and variance per node.
#'
#' @details
#' `"graph"` mode removes graph-level location and scale from the representation,
#' which is what makes it useful for inductive settings where a model trained on
#' one graph is applied to another. If the new graph's features or degree
#' distribution sit at a different scale, un-normalized layers propagate that
#' shift into a systematic bias in the predictions. `"node"` mode is the
#' conventional layer normalization of transformer architectures and does not
#' depend on the graph partition.
#'
#' Parameters (when `affine = TRUE`):
#' - \eqn{\gamma}: `in_features` learnable scale, initialized to 1
#' - \eqn{\beta}: `in_features` learnable shift, initialized to 0
#'
#' @param in_features Integer. Number of input features per node
#' @param eps Numeric. Value added to the denominator for numerical stability.
#'   Default: 1e-5
#' @param affine Logical. If TRUE, adds learnable scale and shift parameters.
#'   Default: TRUE
#' @param mode Character. Either `"graph"` or `"node"`. Default: `"graph"`
#'
#' @section Forward pass:
#' @param x Tensor `n_nodes x in_features`. Node feature matrix
#' @param batch Tensor or NULL. Batch vector assigning each node to a graph, using
#'   1-based graph indices (e.g. `c(1,1,2,2,2)`). If NULL, all nodes are treated
#'   as belonging to a single graph. Ignored when `mode = "node"`.
#' @param batch_size Integer or NULL. Number of graphs. Calculated from `batch`
#'   if NULL.
#'
#' @return Tensor `n_nodes x in_features`. Normalized node features
#'
#' @examples
#' \dontrun{
#' norm <- layer_layer_norm(16)
#'
#' # Single graph
#' norm(x)
#'
#' # Mini-batch of graphs, normalized independently
#' norm(x, batch = torch_tensor(c(1, 1, 2, 2), dtype = torch_long()))
#'
#' # Per-node normalization
#' norm <- layer_layer_norm(16, mode = "node")
#' }
#'
#' @references
#' Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization.
#' <doi:10.48550/arXiv.1607.06450>
#' @export
layer_layer_norm <- nn_module(
  "LayerNorm",

  initialize = function(
    in_features,
    eps = 1e-5,
    affine = TRUE,
    mode = c("graph", "node")
  ) {
    mode <- rlang::arg_match(mode)

    self$in_features <- in_features
    self$eps <- eps
    self$mode <- mode

    if (affine) {
      self$weight <- nn_parameter(torch_ones(in_features))
      self$bias <- nn_parameter(torch_zeros(in_features))
    } else {
      self$weight <- NULL
      self$bias <- NULL
    }
  },

  forward = function(x, batch = NULL, batch_size = NULL) {
    if (self$mode == "node") {
      out <- nnf_layer_norm(x, self$in_features, eps = self$eps)
      if (!is.null(self$weight)) {
        out <- out * self$weight + self$bias
      }
      return(out)
    }

    if (is.null(batch)) {
      out <- (x - x$mean()) / (x$var(unbiased = FALSE) + self$eps)$sqrt()
      if (!is.null(self$weight)) {
        out <- out * self$weight + self$bias
      }
      return(out)
    }

    if (is.null(batch_size)) {
      batch_size <- batch$max()$item()
    }

    n_features <- x$size(2)
    index <- batch$unsqueeze(2)$expand(c(-1, n_features))

    # Nodes per graph, scaled by channels: the number of elements each
    # graph-level statistic is computed over
    counts <- torch_zeros(batch_size, device = x$device)
    counts$scatter_add_(
      1,
      batch,
      torch_ones_like(batch, dtype = x$dtype)
    )
    n_elem <- (counts * n_features)$clamp(min = 1)$unsqueeze(2)

    sums <- torch_zeros(batch_size, n_features, device = x$device)
    sums$scatter_add_(1, index, x)
    mean <- sums$sum(dim = 2, keepdim = TRUE) / n_elem

    centered <- x - mean$index_select(1, batch)

    sq_sums <- torch_zeros(batch_size, n_features, device = x$device)
    sq_sums$scatter_add_(1, index, centered$pow(2))
    var <- sq_sums$sum(dim = 2, keepdim = TRUE) / n_elem

    out <- centered / (var + self$eps)$sqrt()$index_select(1, batch)

    if (!is.null(self$weight)) {
      out <- out * self$weight + self$bias
    }

    out
  }
)

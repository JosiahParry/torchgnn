# Multi-layer Graph Isomorphism Network Model (Xu et al. 2019)

Stacks multiple GIN layers with MLP transformations.

## Usage

``` r
model_gin(
  in_features,
  hidden_dims,
  out_features,
  eps = 0,
  learn_eps = FALSE,
  dropout = 0.5,
  out_activation = NULL
)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- hidden_dims:

  Integer vector. Dimensions of hidden layers (length = L)

- out_features:

  Integer. Number of output features (typically 1 for regression)

- eps:

  Numeric. Initial value for epsilon. Default: 0

- learn_eps:

  Logical. Whether to learn epsilon parameters. Default: FALSE

- dropout:

  Numeric. Dropout rate (0-1) applied between layers. Default: 0.5

- out_activation:

  Function or NULL. Activation for output layer. Default: NULL

## Value

Tensor `n_nodes x out_features`. Final predictions

## Details

Architecture:

- L hidden GIN layers with MLP transformations

- 1 output GIN layer with optional output activation

- Total layers = length(hidden_dims) + 1

Each layer aggregates neighbor features via summation, adds weighted
self features, and applies a 2-layer MLP transformation.

## Forward pass

`model(x, adj)`

- `x`: Tensor `n_nodes x in_features`. Node feature matrix.

- `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix
  defining graph structure.

## References

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are
Graph Neural Networks? International Conference on Learning
Representations. <doi:10.48550/arXiv.1810.00826>

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
adj <- adj_from_edgelist(from = c(1, 2, 3, 4), to = c(2, 3, 4, 1))
x <- torch::torch_randn(4, 14)

# Binary classification
model <- model_gin(14, c(64, 64), 1, out_activation = torch::nnf_sigmoid)
model(x, adj)

# Multi-class classification
model <- model_gin(
  14,
  c(64, 64),
  3,
  out_activation = function(x) torch::nnf_softmax(x, dim = -1)
)
model(x, adj)

# With learnable epsilon
model_gin(14, c(128), 1, learn_eps = TRUE)
}
```

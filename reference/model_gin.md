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

- x:

  Tensor `n_nodes x in_features`. Node feature matrix (dense or sparse)

- adj:

  Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining
  graph structure. Must be a sparse COO tensor.

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

## References

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are
Graph Neural Networks? International Conference on Learning
Representations. <doi:10.48550/arXiv.1810.00826>

## Examples

``` r
if (FALSE) { # \dontrun{
# Binary classification
model <- model_gin(14, c(64, 64), 1, output_activation = nnf_sigmoid)

# Multi-class classification
model <- model_gin(
  14,
  c(64, 64),
  3,
  output_activation = function(x) nnf_softmax(x, dim = -1)
)

# With learnable epsilon
model <- model_gin(14, c(128), 1, learn_eps = TRUE)
} # }
```

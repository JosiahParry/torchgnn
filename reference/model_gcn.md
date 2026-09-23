# Multi-layer GCN Model

Stacks multiple GCN layers to create a deep graph convolutional network.

## Usage

``` r
model_gcn(
  in_features,
  hidden_dims,
  out_features,
  activation = nnf_relu,
  out_activation = NULL,
  dropout = 0,
  normalize = TRUE
)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- hidden_dims:

  Integer vector. Dimensions of hidden layers (length = L)

- out_features:

  Integer. Number of output features (typically 1 for regression)

- activation:

  Function. Activation for hidden layers. Default: nnf_relu

- out_activation:

  Function or NULL. Activation for output layer. Default: NULL

- dropout:

  Numeric. Dropout rate (0-1) applied after each hidden layer. Default:
  0

- normalize:

  Logical. Whether to add self-loops and apply symmetric normalization.
  Default: TRUE

## Value

Tensor `n_nodes x out_features`. Final predictions

## Details

Architecture:

- L hidden GCN layers with configurable activation

- 1 output GCN layer with optional output activation

- Total layers = length(hidden_dims) + 1

For example, hidden_dims = c(56, 56) creates:

- Layer 1: in_features → 56 (activation)

- Layer 2: 56 → 56 (activation)

- Layer 3: 56 → out_features (output_activation)

Uses `gcn_conv_layer` which automatically handles adding self-loops and
symmetric normalization when `normalize = TRUE`.

## Forward pass

`model(x, adj)`

- `x`: Tensor `n_nodes x in_features`. Node feature matrix.

- `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix
  defining graph structure. When `normalize = TRUE`, self-loops are
  added and symmetric normalization is applied by each layer.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
adj <- adj_from_edgelist(from = c(1, 2, 3, 4), to = c(2, 3, 4, 1))
x <- torch::torch_randn(4, 14)

# Regression (no output activation)
model <- model_gcn(14, c(64, 64), 1)
model(x, adj)

# Binary classification with sigmoid
model <- model_gcn(14, c(56, 56), 1, out_activation = torch::nnf_sigmoid)
model(x, adj)

# Multi-class with softmax
model <- model_gcn(
  14,
  c(32, 32),
  10,
  out_activation = function(x) torch::nnf_softmax(x, dim = -1)
)
model(x, adj)

# With dropout and tanh activation
model_gcn(14, c(56, 56), 1, activation = torch::torch_tanh, dropout = 0.5)
}
```

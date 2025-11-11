# Multi-layer GCN Model

Stacks multiple GCN layers to create a deep graph convolutional network.

## Usage

``` r
gcn_model(
  in_features,
  hidden_dims,
  out_features,
  activation = nnf_relu,
  output_activation = NULL,
  dropout = 0
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

- output_activation:

  Function or NULL. Activation for output layer. Default: NULL

- dropout:

  Numeric. Dropout rate (0-1) applied after each hidden layer. Default:
  0

- x:

  Tensor `n_nodes x in_features`. Node feature matrix

- adj:

  Tensor `n_nodes x n_nodes`. Adjacency matrix. Expected to be
  row-normalized \\D^{-1}A\\ where \\D\\ is the degree matrix. Can be
  binary or weighted

- edge_weight:

  Tensor `n_nodes x n_nodes` or NULL. Optional edge weights to apply to
  the adjacency matrix. Passed through to all layers. If NULL, uses
  values from `adj`. Default: NULL

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

## Forward pass

## Examples

``` r
if (FALSE) { # \dontrun{
# Binary classification with sigmoid
model <- gcn_model(14, c(56, 56), 1, output_activation = nnf_sigmoid)

# Multi-class with softmax
model <- gcn_model(
  14,
  c(32, 32),
  10,
  output_activation = function(x) nnf_softmax(x, dim = -1)
)

# Regression (no output activation)
model <- gcn_model(14, c(64, 64), 1)

# With dropout and tanh activation
model <- gcn_model(14, c(56, 56), 1,
                   activation = nnf_tanh,
                   dropout = 0.5)
} # }
```

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

- dropout:

  Numeric. Dropout rate (0-1) applied after each hidden layer. Default:
  0

- normalize:

  Logical. Whether to add self-loops and apply symmetric normalization.
  Default: TRUE

- output_activation:

  Function or NULL. Activation for output layer. Default: NULL

- x:

  Tensor `n_nodes x in_features`. Node feature matrix

- adj:

  Tensor `n_nodes x n_nodes`. Binary adjacency matrix (0/1) defining
  graph structure. When `normalize = TRUE`, self-loops are added and
  symmetric normalization is applied automatically

- edge_weight:

  Tensor `n_nodes x n_nodes` or NULL. Optional edge weights to apply to
  the adjacency structure. If NULL, treats all edges as having weight 1.
  Passed through to all layers. Default: NULL

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

## Examples

``` r
if (FALSE) { # \dontrun{
# Binary classification with sigmoid
model <- model_gcn(14, c(56, 56), 1, output_activation = nnf_sigmoid)

# Multi-class with softmax
model <- model_gcn(
  14,
  c(32, 32),
  10,
  output_activation = function(x) nnf_softmax(x, dim = -1)
)

# Regression (no output activation)
model <- model_gcn(14, c(64, 64), 1)

# With dropout and tanh activation
model <- model_gcn(14, c(56, 56), 1,
                   activation = nnf_tanh,
                   dropout = 0.5)
} # }
```

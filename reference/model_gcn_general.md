# Multi-layer Generalized GCN Model (Hamilton 2020)

Stacks multiple generalized GCN layers with neighbor/self weight
separation. Optional row-normalization is applied internally if
`normalize = TRUE`.

## Usage

``` r
model_gcn_general(
  in_features,
  hidden_dims,
  out_features,
  activation = nnf_relu,
  out_activation = NULL,
  dropout = 0,
  normalize = TRUE
)
```

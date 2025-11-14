# GCN Convolutional Layer (Kipf & Welling 2016)

Implements the basic GCN layer from Kipf & Welling (2016): \$\$H^{(k)} =
\sigma(\tilde{A} H^{(k-1)} W + b)\$\$

## Usage

``` r
layer_gcn(in_features, out_features, bias = TRUE, normalize = TRUE)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- out_features:

  Integer. Number of output features per node

- bias:

  Logical. Add learnable bias. Default: TRUE

- normalize:

  Logical. Whether to add self-loops and compute symmetric normalization
  on-the-fly. Default: TRUE

- x:

  Tensor `n_nodes x in_features`. Node feature matrix

- adj:

  Tensor `n_nodes x n_nodes`. Adjacency matrix defining graph structure.
  Can be binary (0/1) or weighted. If `edge_weight` is provided, `adj`
  should be binary and weights will be applied from `edge_weight`

## Value

Tensor `n_nodes x out_features`. Transformed node features

## Details

Where \\\tilde{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}\\ is the
symmetrically normalized adjacency matrix with self-loops, and
\\\tilde{D}\\ is the degree matrix of \\A + I\\.

This is the standard GCN layer that uses:

- Single weight matrix \\W\\

- Symmetric normalization with self-loops

- Optional on-the-fly normalization

When `normalize = TRUE` (default), the layer computes \\\tilde{A}\\
on-the-fly from the input adjacency matrix by adding self-loops and
applying symmetric normalization. When `normalize = FALSE`, you must
pass in a pre-normalized adjacency matrix.

Parameters:

- \\W\\: `in_features x out_features` learnable weight matrix

- \\b\\: `out_features` learnable bias term (optional)

## Forward pass

## References

Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with
graph convolutional networks. arXiv preprint arXiv:1609.02907.
<doi:10.48550/arXiv.1609.02907>

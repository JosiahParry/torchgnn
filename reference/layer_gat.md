# Graph Attention Network Layer (Veličković et al. 2018)

Implements the Graph Attention Network (GAT) layer:

\$\$\mathbf{h}\_i^{(l+1)} = \sigma\left(\sum\_{j \in \mathcal{N}(i)}
\alpha\_{ij} \mathbf{W}^{(l)} \mathbf{h}\_j^{(l)}\right)\$\$

where the attention coefficients \\\alpha\_{ij}\\ are computed as:

\$\$\alpha\_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T
\[\mathbf{W}\mathbf{h}\_i \|\| \mathbf{W}\mathbf{h}\_j\]))}{\sum\_{k \in
\mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T
\[\mathbf{W}\mathbf{h}\_i \|\| \mathbf{W}\mathbf{h}\_k\]))}\$\$

This layer:

1.  Applies linear transformation to node features

2.  Computes attention coefficients for each edge

3.  Normalizes attention weights via softmax over neighbors

4.  Aggregates neighbor features weighted by attention

Parameters:

- \\W\\: `in_features x out_features` learnable weight matrix

- \\a\\: `2 * out_features` learnable attention vector

## Usage

``` r
layer_gat(
  in_features,
  out_features,
  heads = 1,
  concat = TRUE,
  dropout = 0,
  negative_slope = 0.2,
  bias = TRUE
)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- out_features:

  Integer. Number of output features per node (per head)

- heads:

  Integer. Number of attention heads. Default: 1

- concat:

  Logical. If TRUE, concatenate multi-head outputs. If FALSE, average
  them. Default: TRUE

- dropout:

  Numeric. Dropout rate (0-1) applied to attention coefficients.
  Default: 0

- negative_slope:

  Numeric. Negative slope for LeakyReLU. Default: 0.2

- bias:

  Logical. Add learnable bias. Default: TRUE

- x:

  Tensor `n_nodes x in_features`. Node feature matrix

- adj:

  Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining
  graph structure. Must be a sparse COO tensor.

## Value

Tensor `n_nodes x (out_features * heads)` if concat=TRUE, else
`n_nodes x out_features`

## Details

Multi-head attention is supported via the `heads` parameter. When
`heads > 1`:

- If `concat = TRUE`: outputs are concatenated (output size =
  `out_features * heads`)

- If `concat = FALSE`: outputs are averaged (output size =
  `out_features`)

## Forward pass

## References

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., &
Bengio, Y. (2018). Graph Attention Networks. International Conference on
Learning Representations. <doi:10.48550/arXiv.1710.10903>

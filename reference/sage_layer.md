# GraphSAGE Layer (Hamilton et al. 2017)

Implements the GraphSAGE (Graph Sample and Aggregate) layer:

\$\$\mathbf{h}\_{\mathcal{N}(v)}^{(k)} =
\text{AGGREGATE}\left(\\\mathbf{h}\_u^{(k-1)} : u \in
\mathcal{N}(v)\\\right)\$\$

\$\$\mathbf{h}\_v^{(k)} = \sigma\left(\mathbf{W}^{(k)} \cdot
\text{CONCAT}\left(\mathbf{h}\_v^{(k-1)},
\mathbf{h}\_{\mathcal{N}(v)}^{(k)}\right)\right)\$\$

This layer:

1.  Aggregates neighbor features using the specified aggregator

2.  Concatenates node's own features with aggregated neighbor features

3.  Applies linear transformation and optional normalization

Parameters:

- \\W\\: `(in_features + aggregated_features) x out_features` learnable
  weight matrix

- \\b\\: `out_features` learnable bias term (optional)

## Usage

``` r
sage_layer(
  in_features,
  out_features,
  aggregator = MeanAggregator(),
  bias = TRUE,
  concat = TRUE
)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- out_features:

  Integer. Number of output features per node

- aggregator:

  Aggregator S7 object. Default:
  [`MeanAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)

- bias:

  Logical. Add learnable bias. Default: TRUE

- concat:

  Logical. If TRUE, concatenates self and neighbor features. If FALSE,
  adds them. Default: TRUE

- x:

  Tensor `n_nodes x in_features`. Node feature matrix (dense or sparse)

- adj:

  Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining
  graph structure. Must be a sparse COO tensor.

## Value

Tensor `n_nodes x out_features`. Transformed node features

## Details

The aggregator should be an S7 Aggregator object (e.g.,
[`MeanAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md),
[`MaxAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)).
Each aggregator is responsible for its own normalization. For example,
[`MeanAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
applies row normalization internally, while
[`MaxAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
uses the adjacency structure without normalization.

## Forward pass

## References

Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation
learning on large graphs. Advances in Neural Information Processing
Systems, 30. <doi:10.48550/arXiv.1706.02216>

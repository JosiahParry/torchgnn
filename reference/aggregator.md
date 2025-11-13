# Message Passing Aggregators

Aggregators combine neighbor node features in graph neural networks.
Each aggregator implements a different reduction operation (sum, mean,
max, etc.) to aggregate features from neighboring nodes.

## Usage

``` r
Aggregator(name = character(0), learnable = logical(0))

SumAggregator()

MeanAggregator()

MaxAggregator()

MinAggregator()

ProductAggregator()

VarAggregator()

StdAggregator()

LSTMAggregator(in_features, hidden_features = NULL)

SoftmaxAggregator(in_features, learn = TRUE)
```

## Arguments

- adj:

  Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining
  graph structure. Must be a sparse COO tensor.

- tensor:

  Torch tensor `n_nodes x n_features`. Node feature matrix. Can be dense
  or sparse.

- ...:

  Additional arguments passed to specific aggregator methods.

## Details

Available aggregators:

- `SumAggregator()`: Sum of neighbor features

- `MeanAggregator()`: Mean of neighbor features (with row normalization)

- `MaxAggregator()`: Element-wise maximum of neighbor features

- `MinAggregator()`: Element-wise minimum of neighbor features

- `ProductAggregator()`: Element-wise product of neighbor features

- `VarAggregator()`: Variance of neighbor features

- `StdAggregator()`: Standard deviation of neighbor features

- `LSTMAggregator()`: Not-imlemented

- `SoftmaxAggregator()`: Not-imlemented

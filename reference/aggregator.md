# Message Passing Aggregators

Aggregators combine neighbor node features in graph neural networks.
Each aggregator implements a different reduction over the features of a
node's neighbors, and is passed to a layer that consumes one, such as
[`layer_sage()`](https://josiahparry.github.io/torchgnn/reference/layer_sage.md).

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
```

## Arguments

- name:

  Character scalar. Short identifier for the reduction, such as `"sum"`
  or `"mean"`.

- learnable:

  Logical scalar. Whether the aggregator holds parameters that are
  updated during training.

## Value

An S7 object inheriting from `Aggregator`, with properties `name` and
`learnable`.

## Details

Available aggregators:

- `SumAggregator()`: Sum of neighbor features

- `MeanAggregator()`: Mean of neighbor features (with row normalization)

- `MaxAggregator()`: Element-wise maximum of neighbor features

- `MinAggregator()`: Element-wise minimum of neighbor features

- `ProductAggregator()`: Element-wise product of neighbor features

- `VarAggregator()`: Variance of neighbor features

- `StdAggregator()`: Standard deviation of neighbor features

`Aggregator()` is the abstract parent class and cannot be instantiated
directly. It is exported so that user-defined aggregators can subclass
it and register a `forward()` method.

## See also

[`layer_sage()`](https://josiahparry.github.io/torchgnn/reference/layer_sage.md),
which takes an aggregator.

## Examples

``` r
MeanAggregator()
#> <torchgnn::MeanAggregator>
#>  @ name     : chr "mean"
#>  @ learnable: logi FALSE

SumAggregator()
#> <torchgnn::SumAggregator>
#>  @ name     : chr "sum"
#>  @ learnable: logi FALSE

# Aggregators are passed to the layers that consume them
S7::prop(MaxAggregator(), "name")
#> [1] "max"
```

# Layer Normalization (Ba et al. 2016)

Applies layer normalization to node features:

\$\$\mathbf{x}' = \frac{\mathbf{x} -
\mathrm{E}\[\mathbf{x}\]}{\sqrt{\mathrm{Var}\[\mathbf{x}\] + \epsilon}}
\odot \gamma + \beta\$\$

The mean and variance are computed over the elements selected by `mode`:

- `"graph"`: statistics are computed across all nodes *and* all channels
  of a graph, giving a single mean and variance per graph. When `batch`
  is supplied, each graph in the mini-batch is normalized independently.

- `"node"`: statistics are computed across the channels of each node
  independently, giving one mean and variance per node.

## Usage

``` r
layer_layer_norm(
  in_features,
  eps = 1e-05,
  affine = TRUE,
  mode = c("graph", "node")
)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- eps:

  Numeric. Value added to the denominator for numerical stability.
  Default: 1e-5

- affine:

  Logical. If TRUE, adds learnable scale and shift parameters. Default:
  TRUE

- mode:

  Character. Either `"graph"` or `"node"`. Default: `"graph"`

## Value

Tensor `n_nodes x in_features`. Normalized node features

## Details

`"graph"` mode removes graph-level location and scale from the
representation, which is what makes it useful for inductive settings
where a model trained on one graph is applied to another. If the new
graph's features or degree distribution sit at a different scale,
un-normalized layers propagate that shift into a systematic bias in the
predictions. `"node"` mode is the conventional layer normalization of
transformer architectures and does not depend on the graph partition.

Parameters (when `affine = TRUE`):

- \\\gamma\\: `in_features` learnable scale, initialized to 1

- \\\beta\\: `in_features` learnable shift, initialized to 0

## Forward pass

`layer(x, batch = NULL, batch_size = NULL)`

- `x`: Tensor `n_nodes x in_features`. Node feature matrix.

- `batch`: Tensor or `NULL`. Batch vector assigning each node to a
  graph, using 1-based graph indices (e.g. `c(1, 1, 2, 2, 2)`). If
  `NULL`, all nodes are treated as belonging to a single graph. Ignored
  when `mode = "node"`.

- `batch_size`: Integer or `NULL`. Number of graphs. Calculated from
  `batch` if `NULL`.

## References

Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization.
<doi:10.48550/arXiv.1607.06450>

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
x <- torch::torch_randn(4, 16)

norm <- layer_layer_norm(16)

# Single graph
norm(x)

# Mini-batch of graphs, normalized independently
batch <- torch::torch_tensor(c(1, 1, 2, 2), dtype = torch::torch_long())
norm(x, batch = batch)

# Per-node normalization
norm <- layer_layer_norm(16, mode = "node")
norm(x)
}
```

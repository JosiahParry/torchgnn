# Changelog

## torchgnn 0.1.0

- First release.

- Message-passing layers:
  [`layer_gcn()`](https://josiahparry.github.io/torchgnn/reference/layer_gcn.md),
  [`layer_gcn_general()`](https://josiahparry.github.io/torchgnn/reference/layer_gcn_general.md),
  [`layer_sage()`](https://josiahparry.github.io/torchgnn/reference/layer_sage.md),
  [`layer_gat()`](https://josiahparry.github.io/torchgnn/reference/layer_gat.md),
  [`layer_gin()`](https://josiahparry.github.io/torchgnn/reference/layer_gin.md)
  and
  [`layer_regconv()`](https://josiahparry.github.io/torchgnn/reference/layer_regconv.md).

- Multi-layer models:
  [`model_gcn()`](https://josiahparry.github.io/torchgnn/reference/model_gcn.md),
  [`model_gcn_general()`](https://josiahparry.github.io/torchgnn/reference/model_gcn_general.md),
  [`model_sage()`](https://josiahparry.github.io/torchgnn/reference/model_sage.md),
  [`model_gat()`](https://josiahparry.github.io/torchgnn/reference/model_gat.md)
  and
  [`model_gin()`](https://josiahparry.github.io/torchgnn/reference/model_gin.md).

- Graph-level pooling with
  [`pool_global_add()`](https://josiahparry.github.io/torchgnn/reference/pooling.md),
  [`pool_global_mean()`](https://josiahparry.github.io/torchgnn/reference/pooling.md)
  and
  [`pool_global_max()`](https://josiahparry.github.io/torchgnn/reference/pooling.md),
  and normalization with
  [`layer_layer_norm()`](https://josiahparry.github.io/torchgnn/reference/layer_layer_norm.md).

- Graph construction and preparation helpers:
  [`adj_from_edgelist()`](https://josiahparry.github.io/torchgnn/reference/adj_from_edgelist.md),
  [`nodes_to_tensor()`](https://josiahparry.github.io/torchgnn/reference/nodes_to_tensor.md),
  [`gcn_normalize()`](https://josiahparry.github.io/torchgnn/reference/adjacency.md),
  [`adj_row_normalize()`](https://josiahparry.github.io/torchgnn/reference/adjacency.md),
  [`add_graph_self_loops()`](https://josiahparry.github.io/torchgnn/reference/adjacency.md)
  and
  [`graph_split()`](https://josiahparry.github.io/torchgnn/reference/graph_split.md).

- Requires ‘torch’ 0.17 or later, which removes the need for manual
  1-based index adjustments. Related
  <https://github.com/mlverse/torch/issues/1460>.

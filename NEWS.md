# torchgnn 0.1.0

* First release.

* Message-passing layers: `layer_gcn()`, `layer_gcn_general()`, `layer_sage()`,
  `layer_gat()`, `layer_gin()` and `layer_regconv()`.

* Multi-layer models: `model_gcn()`, `model_gcn_general()`, `model_sage()`,
  `model_gat()` and `model_gin()`.

* Graph-level pooling with `pool_global_add()`, `pool_global_mean()` and
  `pool_global_max()`, and normalization with `layer_layer_norm()`.

* Graph construction and preparation helpers: `adj_from_edgelist()`,
  `nodes_to_tensor()`, `gcn_normalize()`, `adj_row_normalize()`,
  `add_graph_self_loops()` and `graph_split()`.

* Requires 'torch' 0.17 or later, which removes the need for manual 1-based
  index adjustments. Related <https://github.com/mlverse/torch/issues/1460>.

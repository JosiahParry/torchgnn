# Package index

## Layers

Graph neural network layer implementations

- [`layer_gcn()`](https://josiahparry.github.io/torchgnn/reference/layer_gcn.md)
  : GCN Convolutional Layer (Kipf & Welling 2016)
- [`layer_gcn_general()`](https://josiahparry.github.io/torchgnn/reference/layer_gcn_general.md)
  : Generalized GCN Layer (Hamilton 2020)
- [`layer_sage()`](https://josiahparry.github.io/torchgnn/reference/layer_sage.md)
  : GraphSAGE Layer (Hamilton et al. 2017)
- [`layer_gat()`](https://josiahparry.github.io/torchgnn/reference/layer_gat.md)
  : Graph Attention Network Layer (Veličković et al. 2018)
- [`layer_gin()`](https://josiahparry.github.io/torchgnn/reference/layer_gin.md)
  : Graph Isomorphism Network Layer (Xu et al. 2019)
- [`layer_regconv()`](https://josiahparry.github.io/torchgnn/reference/layer_regconv.md)
  : Regional GCN Convolutional Layer (Guo et al. 2025)

## Models

Pre-built graph neural network models

- [`model_gcn()`](https://josiahparry.github.io/torchgnn/reference/model_gcn.md)
  : Multi-layer GCN Model
- [`model_gcn_general()`](https://josiahparry.github.io/torchgnn/reference/model_gcn_general.md)
  : Multi-layer Generalized GCN Model (Hamilton 2020)
- [`model_sage()`](https://josiahparry.github.io/torchgnn/reference/model_sage.md)
  : Multi-layer GraphSAGE Model (Hamilton et al. 2017)
- [`model_gat()`](https://josiahparry.github.io/torchgnn/reference/model_gat.md)
  : Multi-layer Graph Attention Network Model (Veličković et al. 2018)
- [`model_gin()`](https://josiahparry.github.io/torchgnn/reference/model_gin.md)
  : Multi-layer Graph Isomorphism Network Model (Xu et al. 2019)

## Graph Utilities

Functions for constructing and manipulating graphs

- [`graph_split()`](https://josiahparry.github.io/torchgnn/reference/graph_split.md)
  : Create Train/Validation/Test Split for Graph Data
- [`adj_from_edgelist()`](https://josiahparry.github.io/torchgnn/reference/adj_from_edgelist.md)
  : Create Sparse Adjacency Matrix from Edge List
- [`nodes_to_tensor()`](https://josiahparry.github.io/torchgnn/reference/nodes_to_tensor.md)
  : Convert Node Features to Tensor
- [`gcn_normalize()`](https://josiahparry.github.io/torchgnn/reference/adjacency.md)
  [`adj_row_normalize()`](https://josiahparry.github.io/torchgnn/reference/adjacency.md)
  [`add_graph_self_loops()`](https://josiahparry.github.io/torchgnn/reference/adjacency.md)
  : Add self-loops to a graph

## Pooling

Global pooling operations for graph-level representations

- [`pool_global_add()`](https://josiahparry.github.io/torchgnn/reference/pooling.md)
  [`pool_global_mean()`](https://josiahparry.github.io/torchgnn/reference/pooling.md)
  [`pool_global_max()`](https://josiahparry.github.io/torchgnn/reference/pooling.md)
  : Graph Pooling Functions

## Aggregators

Message passing aggregators for combining neighbor features

- [`Aggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  [`SumAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  [`MeanAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  [`MaxAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  [`MinAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  [`ProductAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  [`VarAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  [`StdAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  [`LSTMAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  [`SoftmaxAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
  : Message Passing Aggregators

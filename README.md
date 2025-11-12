# torchgnn


Graph Neural Networks for [`{torch}`](https://mlverse.github.io/torch/)
in R.

**Currently supports:** Graph Convolutional Networks (GCN) as described
in Kipf & Welling (2017) and the Generalized GCN as described by Hamilton (2020).

## Installation

Install the development version:

``` r
pak::pak("josiahparry/torchgnn")
```

## Key Functions

### Model Building

- `gcn_conv_model()`: High-level GCN model with multiple layers
- `gcn_general_model()`: Generalized GCN with multiple layers
- `gcn_conv_layer()`: Standard GCN layer
- `gcn_general_layer()`: Generlized GCN layer from Hamilton 2020

### Data Preparation

- `adj_from_edgelist(from, to)`: Create sparse adjacency matrix from
  edge list (supports character IDs)
- `nodes_to_tensor(nodes, adj, node_id)`: Convert node features
  dataframe to tensor with correct ordering
- `graph_split(X, seed)`: Create train/validation/test splits

## Usage

### Creating Models

``` r
library(torchgnn)

# Multi-layer GCN
model <- gcn_conv_model(
  in_features = 500,
  hidden_dims = c(32, 16),
  out_features = 3,
  dropout = 0.5
)

# Single layer
layer <- gcn_layer(500, 32)
```

### Working with Graph Data

The package provides helpers for converting graph data to tensors:

``` r
# Create adjacency matrix (handles character node IDs)
adj <- adj_from_edgelist(edges$from, edges$to)

# Convert node features (automatically orders by adjacency matrix)
X <- nodes_to_tensor(node_features, adj, node_id = "node_id_column")
```

## Example

This is taken from the examples directory.

Train a GCN on the PubMed Diabetes citation network for document
classification:

``` r
library(torch)
library(dplyr)
```


    Attaching package: 'dplyr'

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

``` r
library(torchgnn)
library(nanoparquet)

# read to tempfiles
# data from https://linqs.org/datasets/#pubmed-diabetes
nodes_tmp <- tempfile("nodes", fileext = ".parquet")
edges_tmp <- tempfile("edges", fileext = ".parquet")

download.file(
  "https://github.com/JosiahParry/torchgnn/raw/refs/heads/main/examples/data/pubmed-diabetes/edges.parquet",
  edges_tmp, quiet = TRUE
)
download.file(
  "https://github.com/JosiahParry/torchgnn/raw/refs/heads/main/examples/data/pubmed-diabetes/nodes.parquet",
  nodes_tmp, quiet = TRUE
)

# read in the nodes and edges
nodes <- read_parquet(nodes_tmp)
edges <- read_parquet(edges_tmp)

# create our adjacency matrix from IDs
adj <- adj_from_edgelist(edges$from, edges$to)

# create our training data tensor
X <- nodes_to_tensor(
  select(nodes, -label),
  adj,
  node_id = "paper_id"
)

# create our target class tensor
Y <- nodes_to_tensor(select(nodes, 1:2), adj, "paper_id")$to(
  dtype = torch_long()
)$squeeze()

# perform a train, test, validation split
split <- graph_split(X, seed = 42)

# define a 2 layer GCN model
model <- gcn_conv_model(
  # number of variables
  in_features = 500,
  # define our hidden layers
  hidden_dims = 16,
  # predicting 3 labels
  out_features = 3,
  dropout = 0.5
)

# use ADAM optimizer
optimizer <- optim_adam(model$parameters, lr = 0.01)

# set our number of epochs
n_epochs <- 200

# create our training loop
for (epoch in 1:n_epochs) {
  model$train()
  optimizer$zero_grad()

  out <- model(X, adj)
  loss <- nnf_cross_entropy(out[split$train_id, ], Y[split$train_id])

  loss$backward()
  optimizer$step()

  model$eval()
  with_no_grad({
    val_out <- model(X, adj)
    val_pred <- val_out[split$val_id, ]$argmax(dim = 2)
    val_acc <- (val_pred == Y[split$val_id])$to(
      dtype = torch_float()
    )$mean()$item()
  })

  # print info every 20 epochs
  if (epoch %% 20 == 0) {
    cat(sprintf(
      "Epoch %d | Loss: %.4f | Val Acc: %.4f\n",
      epoch,
      loss$item(),
      val_acc
    ))
  }
}
```

    Epoch 20 | Loss: 0.8090 | Val Acc: 0.7933
    Epoch 40 | Loss: 0.5746 | Val Acc: 0.8357
    Epoch 60 | Loss: 0.4738 | Val Acc: 0.8491
    Epoch 80 | Loss: 0.4336 | Val Acc: 0.8595
    Epoch 100 | Loss: 0.4034 | Val Acc: 0.8648
    Epoch 120 | Loss: 0.3803 | Val Acc: 0.8671
    Epoch 140 | Loss: 0.3683 | Val Acc: 0.8689
    Epoch 160 | Loss: 0.3611 | Val Acc: 0.8722
    Epoch 180 | Loss: 0.3499 | Val Acc: 0.8722
    Epoch 200 | Loss: 0.3436 | Val Acc: 0.8729

``` r
model$eval()
with_no_grad({
  test_out <- model(X, adj)
  test_pred <- test_out[split$test_id, ]$argmax(dim = 2)
  test_acc <- (test_pred == Y[split$test_id])$to(
    dtype = torch_float()
  )$mean()$item()
})

cat(sprintf("Test Accuracy: %.4f\n", test_acc))
```

    Test Accuracy: 0.8694

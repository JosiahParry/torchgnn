# data from https://linqs.org/datasets/#pubmed-diabetes
library(dplyr)
library(torch)
library(nanoparquet)

# read in the nodes and edges
nodes <- read_parquet("examples/data/pubmed-diabetes/nodes.parquet")
edges <- read_parquet("examples/data/pubmed-diabetes/edges.parquet")

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
  hidden_dims = c(16, 16),
  # predicting 3 labels
  out_features = 3,
  dropout = 0.5
)

# use ADAM optimizer
optimizer <- optim_adam(model$parameters, lr = 0.01)

# set our number of epochs
n_epochs <- 500

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

  if (epoch %% 10 == 0) {
    cat(sprintf(
      "Epoch %d | Loss: %.4f | Val Acc: %.4f\n",
      epoch,
      loss$item(),
      val_acc
    ))
  }
}

model$eval()
with_no_grad({
  test_out <- model(X, adj)
  test_pred <- test_out[split$test_id, ]$argmax(dim = 2)
  test_acc <- (test_pred == Y[split$test_id])$to(
    dtype = torch_float()
  )$mean()$item()
})

cat(sprintf("Test Accuracy: %.4f\n", test_acc))

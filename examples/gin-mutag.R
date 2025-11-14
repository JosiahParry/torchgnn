# https://chrsmrrs.github.io/datasets/docs/datasets/
library(torch)
library(torchgnn)

adj_raw <- readr::read_csv(
  "examples/data/tudatasets/MUTAG/MUTAG_A.txt",
  col_names = c("from", "to")
)

edge_label <- readr::read_csv(
  "examples/data/tudatasets/MUTAG/MUTAG_edge_labels.txt",
  col_names = "edge_label"
)

g_ind <- readr::read_csv(
  "examples/data/tudatasets/MUTAG/MUTAG_graph_indicator.txt",
  col_names = "graph_id"
)

graph_label <- readr::read_csv(
  "examples/data/tudatasets/MUTAG/MUTAG_graph_labels.txt",
  col_names = "label"
)

node_label <- readr::read_csv(
  "examples/data/tudatasets/MUTAG/MUTAG_node_labels.txt",
  col_names = "node_label"
)

node_label_tensor <- torch_tensor(
  node_label$node_label + 1L,
  dtype = torch_long()
)


set.seed(1)
torch_manual_seed(1)

# Learnable node embeddings (following GIN paper)
embedding_layer <- nn_embedding(num_embeddings = 7, embedding_dim = 64)

# Create adjacency matrix for all graphs
adj <- adj_from_edgelist(
  from = adj_raw$from,
  to = adj_raw$to,
  n = nrow(node_label),
  symmetric = TRUE
)

# Batch vector (which graph each node belongs to)
batch <- torch_tensor(g_ind$graph_id, dtype = torch_long())

# Graph labels (target for classification)
# we need to make this 1 based indexed
y <- torch_tensor(
  ifelse(graph_label$label == -1, 1L, 2L),
  dtype = torch_long()
)

# Build GIN model (5 layers as in original paper)
model <- model_gin(
  in_features = 64,
  hidden_dims = c(64, 64),
  out_features = 2,
  learn_eps = TRUE,
  dropout = 0.5
)

# Split the 188 graphs into train/val/test
n_graphs <- nrow(graph_label)
split <- graph_split(graph_label, seed = 42)

# Optimizer (include embedding layer parameters)
optimizer <- optim_adam(
  c(model$parameters, embedding_layer$parameters),
  lr = 0.01
)

# Training loop
epochs <- 200

for (epoch in 1:epochs) {
  model$train()
  optimizer$zero_grad()

  # Compute node embeddings
  x <- embedding_layer(node_label_tensor)

  # Forward pass on ALL nodes
  node_embeddings <- model(x, adj)
  graph_embeddings <- pool_global_add(
    node_embeddings,
    batch,
    size = n_graphs
  )

  # Loss on training graphs only
  loss <- nnf_cross_entropy(
    graph_embeddings[split$train_id, ],
    y[split$train_id]
  )

  # Backward pass
  loss$backward()
  optimizer$step()

  # Evaluation
  if (epoch %% 10 == 0) {
    model$eval()
    with_no_grad({
      x_eval <- embedding_layer(node_label_tensor)
      node_embeddings_eval <- model(x_eval, adj)
      graph_embeddings_eval <- pool_global_add(
        node_embeddings_eval,
        batch,
        size = n_graphs
      )
      preds <- graph_embeddings_eval$argmax(dim = 2)
      train_acc <- (preds[split$train_id] == y[split$train_id])$to(
        dtype = torch_float()
      )$mean()$item()
      val_acc <- (preds[split$val_id] == y[split$val_id])$to(
        dtype = torch_float()
      )$mean()$item()

      cat(sprintf(
        "Epoch %d | Loss: %.4f | Train: %.4f | Val: %.4f\n",
        epoch,
        loss$item(),
        train_acc,
        val_acc
      ))
    })
  }
}

# Final test accuracy
model$eval()
with_no_grad({
  x_test <- embedding_layer(node_label_tensor)
  node_embeddings <- model(x_test, adj)
  graph_embeddings <- pool_global_add(node_embeddings, batch, size = n_graphs)
  test_pred <- graph_embeddings[split$test_id, ]$argmax(dim = 2)
  test_acc <- (test_pred == y[split$test_id])$to(
    dtype = torch_float()
  )$mean()$item()
})

sprintf("Test Accuracy: %.4f\n", test_acc)

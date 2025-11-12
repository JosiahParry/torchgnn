library(dplyr)
library(readr)

nbs <- read_table(
  "examples/data/cora/cora.cites",
  col_names = c("from", "to")
)

# Create node ID mapping
node_mapping <- tibble(paper_id = unique(c(nbs$from, nbs$to))) |>
  mutate(node_idx = row_number())

# Remap edge list
edges_remapped <- nbs |>
  left_join(node_mapping, by = c("from" = "paper_id")) |>
  rename(from_idx = node_idx) |>
  left_join(node_mapping, by = c("to" = "paper_id")) |>
  rename(to_idx = node_idx)

adj <- adj_from_edgelist(
  edges_remapped$from_idx,
  edges_remapped$to_idx
)


content <- read_table(
  "examples/data/cora/cora.content",
  col_names = FALSE,
  col_types = cols(.default = col_double(), X1435 = col_character())
)

# create x
X <- torch_tensor(as.matrix(content_aligned[, 2:1434]))

# Prepare features and labels
content_aligned <- content |>
  rename(paper_id = X1) |>
  left_join(node_mapping, by = "paper_id") |>
  arrange(node_idx)

# extract the y value
y <- torch_tensor(
  match(content_aligned$X1435, unique(content_aligned$X1435)),
  dtype = torch_long()
)

# Create train/val/test mask
split <- graph_split(X, seed = 42)

# Build model (Cora has 7 classes)
model <- gcn_conv_model(
  in_features = 1433,
  hidden_dims = c(32, 16),
  out_features = 7,
  dropout = 0.5
)


# define our optimizer
optimizer <- optim_adam(model$parameters, lr = 0.01)

n_epochs <- 500
# Training loop
for (epoch in 1:n_epochs) {
  model$train()
  optimizer$zero_grad()

  # Forward pass
  out <- model(X, adj)

  # Compute loss only on training nodes
  loss <- nnf_cross_entropy(out[split$train_id, ], y[split$train_id])

  # Backprop
  loss$backward()
  optimizer$step()

  # Evaluation on validation set
  model$eval()
  with_no_grad({
    val_out <- model(X, adj)
    val_pred <- val_out[split$val_id, ]$argmax(dim = 2)
    val_acc <- (val_pred == y[split$val_id])$to(
      dtype = torch_float()
    )$mean()$item()
  })

  if (epoch %% 50 == 0) {
    cat(sprintf(
      "Epoch %d | Loss: %.4f | Val Acc: %.4f\n",
      epoch,
      loss$item(),
      val_acc
    ))
  }
}

# Final test accuracy
model$eval()
with_no_grad({
  test_out <- model(X, adj)
  test_pred <- test_out[split$test_id, ]$argmax(dim = 2)
  test_acc <- (test_pred == y[split$test_id])$to(
    dtype = torch_float()
  )$mean()$item()
})


cat(sprintf("Test Accuracy: %.4f\n", test_acc))


# Build generalized GCN model
model <- gcn_general_model(
  in_features = 1433,
  hidden_dims = c(32, 16),
  out_features = 7,
  dropout = 0.5,
  out_activation = NULL,
  normalize = TRUE
)

# Optimizer
optimizer <- optim_adam(model$parameters, lr = 0.01)

n_epochs <- 500

for (epoch in 1:n_epochs) {
  model$train()
  optimizer$zero_grad()

  # Forward pass
  out <- model(X, adj)

  # Compute loss only on training nodes
  loss <- nnf_cross_entropy(out[split$train_id, ], y[split$train_id])

  # Backpropagation
  loss$backward()
  optimizer$step()

  # Evaluation on validation set
  model$eval()
  with_no_grad({
    val_out <- model(X, adj)
    val_pred <- val_out[split$val_id, ]$argmax(dim = 2)
    val_acc <- (val_pred == y[split$val_id])$to(
      dtype = torch_float()
    )$mean()$item()
  })

  if (epoch %% 50 == 0) {
    cat(sprintf(
      "Epoch %d | Loss: %.4f | Val Acc: %.4f\n",
      epoch,
      loss$item(),
      val_acc
    ))
  }
}

# Final test accuracy
model$eval()
with_no_grad({
  test_out <- model(X, adj)
  test_pred <- test_out[split$test_id, ]$argmax(dim = 2)
  test_acc <- (test_pred == y[split$test_id])$to(
    dtype = torch_float()
  )$mean()$item()
})
cat(sprintf("Test Accuracy: %.4f\n", test_acc))

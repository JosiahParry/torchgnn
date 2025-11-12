library(dplyr)
library(readr)
library(tidyr)

# The file uses sparse format: only non-zero features are included per row
# Format: paper_id \t label=X \t w-word1=val1 \t w-word2=val2 \t ... \t summary=...

# Read all lines (skip header rows)
lines <- readLines(
  "examples/data/pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab"
)
data_lines <- lines[-(1:2)]

# Parse each line
parse_line <- function(line) {
  fields <- strsplit(line, "\t")[[1]]
  paper_id <- fields[1]

  # Extract label
  label_str <- fields[2]
  label <- as.integer(sub("label=", "", label_str))

  # Extract word features (skip first, last fields, and summary if present)
  feature_fields <- fields[-c(1, 2, length(fields))]

  # Parse each feature
  features <- sapply(
    feature_fields,
    function(f) {
      parts <- strsplit(f, "=")[[1]]
      word <- parts[1]
      value <- as.numeric(parts[2])
      c(word = word, value = value)
    },
    USE.NAMES = FALSE
  )

  if (length(features) > 0) {
    feature_df <- data.frame(
      paper_id = paper_id,
      label = label,
      word = features[1, ],
      value = as.numeric(features[2, ]),
      stringsAsFactors = FALSE
    )
  } else {
    feature_df <- data.frame(
      paper_id = paper_id,
      label = label,
      word = character(0),
      value = numeric(0),
      stringsAsFactors = FALSE
    )
  }

  feature_df
}

# Parse all lines and combine
cat("Parsing", length(data_lines), "papers...\n")
all_features <- lapply(data_lines, parse_line)
features_long <- bind_rows(all_features)

# Get unique paper metadata
nodes_meta <- features_long |>
  select(paper_id, label) |>
  distinct()

# Convert to wide format (sparse to dense)
features_wide <- features_long |>
  pivot_wider(
    id_cols = c(paper_id, label),
    names_from = word,
    values_from = value,
    values_fill = 0
  ) |>
  rename_with(heck::to_snek_case)

# Read the edges file (citation network)
# Format: edge_id \t paper:from_id \t | \t paper:to_id
edges <- read_tsv(
  "examples/data/pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab",
  skip = 2,
  col_names = c("edge_id", "from", "sep", "to"),
  show_col_types = FALSE
) |>
  mutate(
    from = sub("paper:", "", from),
    to = sub("paper:", "", to)
  ) |>
  select(from, to)

# Summary
cat("\nDataset Summary:\n")
cat("Nodes:", nrow(features_wide), "\n")
cat("Edges:", nrow(edges), "\n")
cat("Classes:", paste(sort(unique(features_wide$label)), collapse = ", "), "\n")
cat("Features:", ncol(features_wide) - 2, "words\n")


nanoparquet::write_parquet(
  features_wide,
  "examples/data/pubmed-diabetes/nodes.parquet",
  compression = "gz"
)

nanoparquet::write_parquet(
  arrange(edges, from, to),
  encoding = "RLE_DICTIONARY",
  "examples/data/pubmed-diabetes/edges.parquet",
  compression = "gz"
)

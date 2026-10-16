# Features

- same features as original paper
- number of patches undefined, but 60 is chosen for subsampling later (or 200 for large, but not used)
- aggregation (mean) over all windows for each track before computing the unigrams (unclear if specified in paper)

# Datasets

- training on fma_small (8000 files) with GMM and K-Means (nano model size)
- training on fma_medium (25000 files) with K-Means (micro model size)

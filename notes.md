# Features

- same features as original paper
- number of patches undefined, but 60 is chosen for subsampling later (or 200 for large, but not used)
- aggregation (sum) over all windows for each track before computing the unigrams (following figure in slides)

# Datasets

- training on fma_small (8000 files) with GMM and K-Means (nano model size)
- training on fma_medium (25000 files) with K-Means (micro model size)
- https://github.com/jongpillee/music_dataset_split/tree/master/GTZAN_split used for GTZAN

from __future__ import annotations


import logging
from pathlib import Path
from time import perf_counter
from typing import cast

import joblib
import numpy as np
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from features import _flatten_feature_by_dimension, _restore_feature_shape, extract_dataset_features
from helpers import TaskResult, atomic_write, get_output_dir, write_feature_archive, data_from_file


FEATURE_NAMES = ("mfcc", "cqt", "rhythmic", "spectral", "patches")
PCA_COMPONENT_TARGETS: dict[str, float | str] = {
	"mfcc": 0.95,
	"cqt": 0.94,
	"rhythmic": "mle",
	"spectral": "mle",
	"patches": 0.99,
}
FINAL_PCA_TARGET = 0.97

@data_from_file(
	dataset_name="fma_small",
	input_filename="features.npz",
	output_filename="sampled_features.npz",
	overwrite=False
)
def sample_dataset_feature_windows(
	source,
	n_samples: int = 60,
	random_state: int | None = 0,
):
	"""Randomly sample windows independently for every audio segment.

	The input arrays must have shape ``(segments, dimensions, windows)``.
	Sampling is performed without replacement within each segment and feature
	group. The output keeps the same layout with ``n_samples`` windows:
	``(segments, dimensions, n_samples)``. The original archive is unchanged.
	"""
	if n_samples < 1:
		raise ValueError("n_samples must be at least 1")
	
	rng = np.random.default_rng(random_state)
	sampled_features: dict[str, np.ndarray] = {}

	for feature_number, name in enumerate(FEATURE_NAMES, start=1):
		data = np.asarray(source[name])
		if data.ndim != 3:
			raise ValueError(
				f"Expected {name} to have shape "
				f"(segments, dimensions, windows), got {data.shape}"
			)
		segments, dimensions, windows = data.shape
		if n_samples > windows:
			raise ValueError(
				f"Cannot sample {n_samples} windows from {name}: "
				f"only {windows} windows are available per segment"
			)
		logging.info(
			"Sampling feature %d/%d: %s (%d windows per segment -> %d)",
			feature_number,
			len(FEATURE_NAMES),
			name,
			windows,
			n_samples,
		)
		indices = np.stack(
			[
				rng.choice(windows, size=n_samples, replace=False)
				for _ in range(segments)
			]
		)
		broadcast_indices = np.broadcast_to(
			indices[:, None, :],
			(segments, dimensions, n_samples),
		)
		sampled_features[name] = np.take_along_axis(
			data,
			broadcast_indices,
			axis=2,
		)
		logging.info(
			"Finished sampling feature %d/%d: %s, output shape=%s",
			feature_number,
			len(FEATURE_NAMES),
			name,
			sampled_features[name].shape,
		)

	return TaskResult(
		result=sampled_features, 
		results_file_saver=lambda path: write_feature_archive(
			path,
			sampled_features,
			FEATURE_NAMES,
		), 
		timer_log=lambda elapsed: logging.info(
			"Window sampling completed in %.2f seconds", elapsed
		)
	)


@data_from_file(
	dataset_name="fma_small",
	input_filename="sampled_features.npz",
	output_filename="scaled_features.npz",
	overwrite=False
)
def compute_G1(
	source
):
	"""Compute per-segment feature means and standard deviations.

	The unsampled input arrays must have shape
	``(segments, dimensions, windows)``. For every feature group, the window
	axis is reduced independently for each 9-second segment. The output archive
	contains ``<feature>_mean`` and ``<feature>_std`` arrays with shape
	``(segments, dimensions)``.
	"""
	statistics: dict[str, np.ndarray] = {}
	segment_count: int | None = None

	for feature_number, name in enumerate(FEATURE_NAMES, start=1):
		data = np.asarray(source[name])
		if data.ndim != 3:
			raise ValueError(
				f"Expected {name} to have shape "
				f"(segments, dimensions, windows), got {data.shape}"
			)
		if segment_count is None:
			segment_count = data.shape[0]
		elif data.shape[0] != segment_count:
			raise ValueError(
				f"All feature groups must have the same segment count; "
				f"{name} has {data.shape[0]}, expected {segment_count}"
			)
		logging.info(
			"Computing G1 %d/%d for %s over %d windows",
			feature_number,
			len(FEATURE_NAMES),
			name,
			data.shape[2],
		)
		statistics[f"{name}_mean"] = np.mean(
			data,
			axis=2,
			dtype=np.float32,
		)
		statistics[f"{name}_std"] = np.std(
			data,
			axis=2,
			dtype=np.float32,
		)

	return TaskResult(
		result=statistics, 
		results_file_saver=lambda path: np.savez_compressed(file=str(path), **statistics),  # type: ignore
		timer_log=lambda elapsed: logging.info( # type: ignore
			"G1 completed for %d feature arrays in %.2f seconds",
			len(FEATURE_NAMES),
			elapsed,
		)
	)



@data_from_file(
	dataset_name="fma_small",
	input_filename="sampled_features.npz",
	overwrite=False,
	output_filename="scaled_features.npz",
	model_filename="scalers.joblib",
)
def scale_dataset_features(
	source
):
	"""Standardize extracted features into a sibling archive.

	A separate scaler is fitted for each named feature array, with one scaler
	column per flattened feature dimension. CQT uses a no-op StandardScaler so
	it remains unchanged while retaining the same persisted scaler interface.
	"""
	
	transformed_features: dict[str, np.ndarray] = {}
	scalers: dict[str, StandardScaler] = {}

	for feature_number, name in enumerate(FEATURE_NAMES, start=1):
		logging.info(
			"Scaling feature %d/%d: loading %s",
			feature_number,
			len(FEATURE_NAMES),
			name,
		)
		data = np.asarray(source[name])
		flattened = _flatten_feature_by_dimension(data)
		scaler = (
			StandardScaler(with_mean=False, with_std=False)
			if name == "cqt"
			else StandardScaler()
		)
		transformed = scaler.fit_transform(flattened)
		transformed_features[name] = _restore_feature_shape(
			transformed.astype(np.float32, copy=False),
			data.shape,
		)
		scalers[name] = scaler
		logging.info(
			"Finished scaling feature %d/%d: %s",
			feature_number,
			len(FEATURE_NAMES),
			name,
		)
	logging.info("Scaled %d feature arrays", len(FEATURE_NAMES))

	return TaskResult(
		result=transformed_features, 
		results_file_saver=lambda path: write_feature_archive(path, transformed_features, FEATURE_NAMES), 
		model_file_saver=lambda path: joblib.dump(scalers, path), 
		timer_log=lambda elapsed: logging.info(
			"Scaling completed in %.2f seconds", elapsed
		))

@data_from_file(
	dataset_name="fma_small",
	input_filename="scaled_features.npz",
	overwrite=False,
	output_filename="pca_features.npz",
	model_filename="pcas.joblib",
)
def whitening_pca_dataset_features(
	source
):
	"""Whiten scaled features with full regular PCA.

	MFCC, CQT, and patches retain enough components to reach their configured
	explained-variance targets. Rhythmic and spectral use Minka's maximum
	likelihood dimensionality estimation. Each input row is flattened before
	fitting and each output row contains the selected whitened components.
	"""
	
	transformed_features: dict[str, np.ndarray] = {}
	pcas: dict[str, PCA] = {}

	for feature_number, name in enumerate(FEATURE_NAMES, start=1):
		logging.info(
			"PCA feature %d/%d: loading %s",
			feature_number,
			len(FEATURE_NAMES),
			name,
		)
		data = np.asarray(source[name])
		flattened = _flatten_feature_by_dimension(data)
		target = PCA_COMPONENT_TARGETS[name]
		logging.info(
			"Fitting PCA for %s on shape %s with target=%s",
			name,
			flattened.shape,
			"Minka MLE" if target == "mle" else f"{target:.0%}",
		)
		pca = PCA(n_components=target, svd_solver="full", whiten=True)
		transformed = pca.fit_transform(flattened).astype(
			np.float32, copy=False
		)
		transformed_features[name] = _restore_feature_shape(
			transformed,
			(data.shape[0], pca.n_components_, data.shape[2]),
		)
		pcas[name] = pca
		logging.info(
			"Finished PCA feature %d/%d: %s (%d components)",
			feature_number,
			len(FEATURE_NAMES),
			name,
			pca.n_components_,
		)

	logging.info("PCA completed for %d feature arrays", len(FEATURE_NAMES))

	return TaskResult(
		result=transformed_features, 
		results_file_saver=lambda path: write_feature_archive(path, transformed_features, FEATURE_NAMES), 
		model_file_saver=lambda path: joblib.dump(pcas, path), 
		timer_log=lambda elapsed: logging.info(
			"PCA completed in %.2f seconds", elapsed
		))


@data_from_file(
	dataset_name="fma_small",
	input_filename="pca_features.npz",
	overwrite=False,
	output_filename="kmeans_labels.npz",
	model_filename="kmeans.joblib",
)
def kmeans_dataset_features(
	source,
	n_clusters: int = 512, # K in paper
):
	"""Cluster each PCA feature group independently.

	Each segment-window pair is treated as one vector. The window axis is
	flattened into the observation axis while PCA components remain the
	feature dimensions. A separate K-means model is fitted for each feature
	group. Labels are saved in ``kmeans_labels.npz`` and models in
	``kmeans.joblib``.
	"""
	if n_clusters < 1:
		raise ValueError("n_clusters must be at least 1")
	
	labels: dict[str, np.ndarray] = {}
	models: dict[str, KMeans] = {}

	for feature_number, name in enumerate(FEATURE_NAMES, start=1):
		data = np.asarray(source[name])
		vectors = np.ascontiguousarray(
			_flatten_feature_by_dimension(data),
			dtype=np.float32,
		)
		logging.info(
			"Fitting K-means feature %d/%d: %s, shape=%s, k=%d",
			feature_number,
			len(FEATURE_NAMES),
			name,
			vectors.shape,
			n_clusters,
		)
		model = KMeans( 
			n_clusters=n_clusters,
			init="k-means++",
			n_init=1,
			max_iter=50,
			tol=1e-3,
			algorithm="elkan",
			random_state=0,
		)
		labels[name] = model.fit_predict(vectors).astype(np.int32, copy=False)
		models[name] = model
		logging.info(
			"Finished K-means feature %d/%d: %s, inertia=%.6g",
			feature_number,
			len(FEATURE_NAMES),
			name,
			model.inertia_,
		)
	logging.info("K-means completed for %d feature arrays", len(FEATURE_NAMES))

	return TaskResult(
		result=labels, 
		results_file_saver=lambda path: write_feature_archive(path, labels, FEATURE_NAMES), 
		model_file_saver=lambda path: joblib.dump(models, path), 
		timer_log=lambda elapsed: logging.info(
			"K-means completed in %.2f seconds", elapsed
		))

@data_from_file(
	dataset_name="fma_small",
	input_filename="pca_features.npz",
	overwrite=False,
	output_filename="gmm_labels.npz",
	model_filename="gmm.joblib",
)
def gmm_dataset_features(
	source,
	output_dir: Path,
	n_clusters: int = 512, # K in paper
):
	"""Cluster each PCA feature group independently.

	Each segment-window pair is treated as one vector. The window axis is
	flattened into the observation axis while PCA components remain the
	feature dimensions. A separate GMM model is fitted for each feature
	group. Labels are saved in ``gmm_labels.npz`` and models in
	``gmm.joblib``.
	"""
	if n_clusters < 1:
		raise ValueError("n_clusters must be at least 1")
	
	kmeans_models_path = output_dir / "kmeans.joblib"
	
	if not kmeans_models_path.is_file():
		raise FileNotFoundError(
			"K-means models do not exist: "
			f"{kmeans_models_path}; run kmeans_dataset_features first"
		)


	labels: dict[str, np.ndarray] = {}
	models: dict[str, GaussianMixture] = {}
	kmeans_models = cast(dict[str, KMeans], joblib.load(kmeans_models_path))

	for feature_number, name in enumerate(FEATURE_NAMES, start=1):
		data = np.asarray(source[name], dtype=np.float64)
		vectors = _flatten_feature_by_dimension(data)
		logging.info(
			"Fitting GMM feature %d/%d: %s, shape=%s, k=%d",
			feature_number,
			len(FEATURE_NAMES),
			name,
			vectors.shape,
			n_clusters,
		)
		kmeans_model = kmeans_models.get(name)
		if kmeans_model is None:
			raise ValueError(
				f"K-means model archive is missing the {name} feature"
			)
		kmeans_cluster_count = kmeans_model.cluster_centers_.shape[0]
		if kmeans_cluster_count != n_clusters:
			raise ValueError(
				f"K-means model for {name} has {kmeans_cluster_count} "
				f"clusters, but GMM requested {n_clusters}"
			)
		means_init = np.asarray(
			kmeans_model.cluster_centers_,
			dtype=np.float64,
		)
		model = GaussianMixture(
			n_components=n_clusters,
			covariance_type="diag",
			means_init=means_init,
			n_init=1,
			max_iter=30,
			tol=1e-3,
			random_state=0,
			reg_covar=1e-4
		)
		model.fit(vectors)
		labels[name] = model.predict_proba(vectors).astype(np.float64, copy=False)
		models[name] = model
		logging.info(
			"Finished GMM feature %d/%d: %s, lower bound=%.6g",
			feature_number,
			len(FEATURE_NAMES),
			name,
			model.lower_bound_,
		)

	logging.info("GMM completed for %d feature arrays", len(FEATURE_NAMES))

	return TaskResult(
		result=labels, 
		results_file_saver=lambda path: write_feature_archive(path, labels, FEATURE_NAMES), 
		model_file_saver=lambda path: joblib.dump(models, path), 
		timer_log=lambda elapsed: logging.info(
			"GMM completed in %.2f seconds", elapsed
		))


def _write_unigram_archive(
	path: Path,
	unigrams: dict[str, np.ndarray | sparse.csr_matrix],
	sparse_output: bool,
	cluster_ids: dict[str, np.ndarray],
) -> None:
	"""Write one unigram matrix per feature group to a single NPZ archive."""
	payload: dict[str, np.ndarray] = {}
	if sparse_output:
		for name in FEATURE_NAMES:
			matrix = sparse.csr_matrix(unigrams[name])
			payload[f"{name}_data"] = matrix.data
			payload[f"{name}_indices"] = matrix.indices
			payload[f"{name}_indptr"] = matrix.indptr
			payload[f"{name}_shape"] = np.asarray(matrix.shape, dtype=np.int64)
			payload[f"{name}_cluster_ids"] = cluster_ids[name]
	else:
		for name in FEATURE_NAMES:
			payload[name] = np.asarray(unigrams[name], dtype=np.float32)
			payload[f"{name}_cluster_ids"] = cluster_ids[name]
	np.savez_compressed(path, **payload)  # pyright: ignore[reportArgumentType]

def build_unigram_kmm(assignments, name: str, min_presence: float, windows_per_segment: int, inference=False, unigram_archive_path: str = ""):
	if assignments.ndim != 1:
		raise ValueError(
			f"K-means assignments for {name} must be 1D, got {assignments.shape}"
		)
	if assignments.size == 0:
		raise ValueError(f"K-means assignments for {name} are empty")
	if np.any(assignments < 0):
		raise ValueError(f"K-means assignments for {name} contain negatives")

	if (not inference):
		n_clusters = int(assignments.max()) + 1
		presence = np.bincount(
			assignments.astype(np.int64, copy=False),
			minlength=n_clusters,
		) / assignments.size

		keep = presence >= min_presence
		kept_ids = np.flatnonzero(keep)
		if kept_ids.size == 0:
			raise ValueError(
				f"All K-means codebook columns for {name} are below "
				f"min_presence={min_presence}"
			)
		assignment_ids = assignments.astype(np.int64, copy=False)
		row_mask = keep[assignment_ids]
		observation_count = assignments.size
		if observation_count % windows_per_segment != 0:
			raise ValueError(
				f"K-means assignments for {name} contain "
				f"{observation_count} observations, which is not divisible "
				f"by windows_per_segment={windows_per_segment}"
			)
		track_count = observation_count // windows_per_segment
		rows = np.flatnonzero(row_mask)
		remapped = np.searchsorted(kept_ids, assignment_ids[row_mask])
		window_rows = rows // windows_per_segment
	else:
		kept_ids = np.array(_read_unigram_archive(Path(unigram_archive_path), "kmeans")[1][name])
		assignment_ids = assignments.astype(np.int64, copy=False)

		# Keep only assignments belonging to the retained clusters
		row_mask = np.isin(assignment_ids, kept_ids)

		# Original positions of the retained observations
		rows = np.flatnonzero(row_mask)

		# Remap original cluster IDs to [0, len(kept_ids) - 1]
		remapped = np.searchsorted(kept_ids, assignment_ids[row_mask])

		observation_count = assignments.size
		if observation_count % windows_per_segment != 0:
			raise ValueError(
				f"K-means assignments for {name} contain "
				f"{observation_count} observations, which is not divisible "
				f"by windows_per_segment={windows_per_segment}"
			)

		track_count = observation_count // windows_per_segment
		window_rows = rows // windows_per_segment

	return window_rows, remapped, track_count, kept_ids

def build_unigram_gmm(assignments, name: str, min_presence: float, windows_per_segment: int, inference=False, unigram_archive_path: str = ""):
	if assignments.ndim != 2:
		raise ValueError(
			f"GMM probabilities for {name} must be 2D, got {assignments.shape}"
		)
	if assignments.shape[1] < 1:
		raise ValueError(f"GMM probabilities for {name} have no codebook vectors")
	if np.any(~np.isfinite(assignments)) or np.any(assignments < 0):
		raise ValueError(f"GMM probabilities for {name} are invalid")
	if (not inference):
		presence = assignments.mean(axis=0)
		keep = presence >= min_presence
		kept_ids = np.flatnonzero(keep)
		if kept_ids.size == 0:
			raise ValueError(
				f"All GMM codebook columns for {name} are below "
				f"min_presence={min_presence}"
			)
		probabilities = assignments[:, kept_ids]
		row_totals = probabilities.sum(axis=1, keepdims=True)
		if np.any(row_totals <= 0):
			raise ValueError(
				f"Filtering GMM codebook columns leaves empty rows for {name}"
			)
		observation_count = assignments.shape[0]
		if observation_count % windows_per_segment != 0:
			raise ValueError(
				f"GMM probabilities for {name} contain "
				f"{observation_count} observations, which is not divisible "
				f"by windows_per_segment={windows_per_segment}"
			)
		track_count = observation_count // windows_per_segment
		normalized = probabilities / row_totals
	else:
		kept_ids = np.array(_read_unigram_archive(Path(unigram_archive_path), "gmm")[1][name])
		# Keep only the codebook columns retained during training
		probabilities = assignments[:, kept_ids]

		# Renormalize probabilities after filtering columns
		row_totals = probabilities.sum(axis=1, keepdims=True)
		if np.any(row_totals <= 0):
			raise ValueError(
				f"Filtering GMM codebook columns leaves empty rows for {name}"
			)

		normalized = probabilities / row_totals

		observation_count = assignments.shape[0]
		if observation_count % windows_per_segment != 0:
			raise ValueError(
				f"GMM probabilities for {name} contain "
				f"{observation_count} observations, which is not divisible "
				f"by windows_per_segment={windows_per_segment}"
			)

		track_count = observation_count // windows_per_segment


	return normalized, track_count, kept_ids

@data_from_file(
	dataset_name="fma_small",
	input_filename=lambda **params: f"{params['clustering']}_labels.npz",
	overwrite=False,
	output_filename=lambda **params: f"{params['clustering']}_unigrams.npz",
)
def build_unigram_dataset_features(
	source,
	clustering: str,
	min_presence: float = 0.001,
	windows_per_segment: int = 60,
	inference=False,
	unigram_archive_path: str = ""
):
	"""Convert clustering assignments into one matrix per feature group.

	K-means assignments are stored as sparse CSR one-hot matrices. GMM
	assignments are stored as dense float32 posterior-probability matrices.
	Input rows represent flattened segment-window observations. The returned
	matrices have one row per segment and one column per retained codebook
	vector; each row is the sum of its ``windows_per_segment`` windows.
	Codebook columns present in less than ``min_presence`` of the observations
	are removed. For GMM, posterior probability mass is used as the presence
	measure and retained windows are renormalized before summing.
	"""
	if clustering not in ("kmeans", "gmm"):
		raise ValueError("clustering must be either 'kmeans' or 'gmm'")
	if not 0 <= min_presence <= 1:
		raise ValueError("min_presence must be between 0 and 1")
	if windows_per_segment < 1:
		raise ValueError("windows_per_segment must be at least 1")

	unigrams: dict[str, np.ndarray | sparse.csr_matrix] = {}
	cluster_ids: dict[str, np.ndarray] = {}
	for feature_number, name in enumerate(FEATURE_NAMES, start=1):
		assignments = np.asarray(source[name])
		if clustering == "kmeans":
			window_rows, remapped, track_count, kept_ids = build_unigram_kmm(
				assignments, name, min_presence, windows_per_segment, inference=inference, unigram_archive_path=unigram_archive_path
			)
			unigrams[name] = sparse.csr_matrix(
				(
					np.ones(window_rows.size, dtype=np.float32),
					(window_rows, remapped),
				),
				shape=(track_count, kept_ids.size),
			)
			cluster_ids[name] = kept_ids.astype(np.int32, copy=False)
		else:
			normalized, track_count, kept_ids = build_unigram_gmm(
				assignments, name, min_presence, windows_per_segment, inference=inference, unigram_archive_path=unigram_archive_path
			)
			unigrams[name] = normalized.reshape(
				track_count,
				windows_per_segment,
				kept_ids.size,
			).sum(axis=1)
			cluster_ids[name] = kept_ids.astype(np.int32, copy=False)
		
		logging.info(
			"Built unigram %d/%d for %s: shape=%s",
			feature_number,
			len(FEATURE_NAMES),
			name,
			np.asarray(unigrams[name].shape),
		)
		logging.info(
			"Retained %d/%d codebook columns for %s",
			len(cluster_ids[name]),
			len(kept_ids),
			name,
		)


	return TaskResult(
		result=lambda path: (path, unigrams),
		results_file_saver=lambda path: _write_unigram_archive(
			path,
			unigrams,
			sparse_output=clustering == "kmeans",
			cluster_ids=cluster_ids,
		),
		timer_log=lambda elapsed: logging.info(
			"%s unigram matrices completed in %.2f seconds",
			clustering.upper(),
			elapsed,
		)
	)


def _read_unigram_archive(
	input_path: Path,
	clustering: str,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, np.ndarray]]:
	"""Load all feature unigram matrices from a clustering archive."""
	unigrams: dict[str, sparse.csr_matrix] = {}
	cluster_ids: dict[str, np.ndarray] = {}
	with np.load(input_path, allow_pickle=False) as source:
		for name in FEATURE_NAMES:
			if clustering == "kmeans":
				matrix = sparse.csr_matrix(
					(
						source[f"{name}_data"],
						source[f"{name}_indices"],
						source[f"{name}_indptr"],
					),
					shape=tuple(source[f"{name}_shape"]),
					dtype=np.float32,
				)
				cluster_ids[name] = source[f"{name}_cluster_ids"]
			else:
				matrix = sparse.csr_matrix(
					np.asarray(source[name], dtype=np.float32)
				)
				cluster_ids[name] = source[f"{name}_cluster_ids"]
			unigrams[name] = matrix
	return unigrams, cluster_ids

def wmf_dataset_features(
	dataset_name: str = "fma_small",
	unigram_matrix: np.ndarray | sparse.spmatrix | None = None,
	alpha: float = 5,
	lambda_: float = 0.5,
	factors: int = 128,
	output_name: str = "wmf",
	overwrite: bool = False,
) -> Path:
	"""Fit implicit ALS on a supplied unigram matrix and save user factors.

	Rows are 9-second segments and columns are codebook vectors. ALS is fitted
	on the complete matrix and produces one user-factor vector per segment. The
	full ALS model, including item factors, is saved for later fitting.
	"""
	if unigram_matrix is None:
		raise ValueError("unigram_matrix must be provided directly")
	if alpha <= 0:
		raise ValueError("alpha must be greater than 0")
	if lambda_ < 0:
		raise ValueError("lambda_ must be non-negative")
	if factors < 1:
		raise ValueError("factors must be at least 1")
	if not output_name:
		raise ValueError("output_name must not be empty")
	output_dir = get_output_dir(dataset_name)
	output_path = output_dir / f"{output_name}_user_factors.npz"
	model_path = output_dir / f"{output_name}.joblib"
	if output_path.exists() and not overwrite:
		raise FileExistsError(
			f"{output_path} already exists; pass overwrite=True to replace it"
		)
	if model_path.exists() and not overwrite:
		raise FileExistsError(
			f"{model_path} already exists; pass overwrite=True to replace it"
		)

	started_at = perf_counter()
	unigram = sparse.csr_matrix(unigram_matrix, dtype=np.float32)
	unigram_shape = cast(tuple[int, int], unigram.shape)
	if unigram_shape[0] == 0 or unigram_shape[1] == 0:
		raise ValueError(f"Unigram matrix is empty: {unigram_shape}")

	logging.info(
		"Fitting WMF on unigram: shape=%s, alpha=%g, lambda=%g, factors=%d",
		unigram_shape,
		alpha,
		lambda_,
		factors,
	)
	model = AlternatingLeastSquares(
		factors=factors,
		regularization=lambda_,
		alpha=alpha,
		dtype=np.float32,
		random_state=0,
	)
	model.fit(unigram)
	user_factors = np.asarray(
		cast(np.ndarray, model.user_factors),
		dtype=np.float32,
	)

	atomic_write(
		output_path,
		lambda path: np.savez_compressed(
			path,
			user_factors=user_factors,
			factors=np.asarray(factors, dtype=np.int64),
		),
		".tmp.npz",
	)
	atomic_write(
		model_path,
		lambda path: joblib.dump(model, path),
		".tmp.joblib",
	)
	logging.info(
		"WMF completed: %d segment vectors with shape %s; model saved to %s",
		unigram_shape[0],
		user_factors.shape,
		model_path,
	)
	logging.info("WMF completed in %.2f seconds", perf_counter() - started_at)
	return output_path

def extract_and_sample_features(
	dataset_name: str = "fma_small",
	sample_rate: int = 22050,
	hop_size: int = 512,
	n_samples: int = 60,
):
	"""Extract features and sample windows, replacing previous outputs."""
	extract_dataset_features(
		dataset_name,
		sample_rate=sample_rate,
		hop_size=hop_size,
		n_samples=n_samples,
		overwrite=True,
	)
	return sample_dataset_feature_windows(
		dataset_name=dataset_name,
		n_samples=n_samples,
		overwrite=True,
	)

def train_quantization_stage(
	clustering: str,
	n_clusters: int,
	dataset_name: str = "fma_small",
) -> object:
	"""Run scaling, PCA, and the selected clustering stage from scratch."""
	if clustering not in ("kmeans", "gmm"):
		raise ValueError("clustering must be either 'kmeans' or 'gmm'")
	if n_clusters < 1:
		raise ValueError("n_clusters must be at least 1")
	scale_dataset_features(dataset_name, overwrite=True)
	whitening_pca_dataset_features(dataset_name, overwrite=True)
	kmeans_labels = kmeans_dataset_features(
		dataset_name,
		n_clusters=n_clusters,
		overwrite=True,
	)
	if clustering == "kmeans":
		return kmeans_labels
	return gmm_dataset_features(
		dataset_name,
		n_clusters=n_clusters,
		overwrite=True,
	)

def train_factorization_stage(
	dataset_name: str = "fma_small",
	clustering: str = "kmeans",
	alpha: float = 5,
	lambda_: float = 0.5,
	factors_default: int = 128,
	windows_per_segment: int = 60,
	min_presence: float = 0.001,
) -> Path:
	"""Build unigrams and fit WMF per feature plus on their concatenation.

	The clustering output is converted into one unigram per feature group.
	Each unigram is factorized independently, and ``qf_total`` is factorized
	after horizontally concatenating all five unigram matrices. All models and
	one flattened user-factor archive per result are retained.
	"""
	if clustering not in ("kmeans", "gmm"):
		raise ValueError("clustering must be either 'kmeans' or 'gmm'")
	output_dir = get_output_dir(dataset_name)
	unigram_path = cast(tuple[Path, object], build_unigram_dataset_features(
		dataset_name=dataset_name,
		clustering=clustering,
		min_presence=min_presence,
		windows_per_segment=windows_per_segment,
		overwrite=True,
	))[0]

	unigrams = cast(
		dict[str, sparse.csr_matrix],
		_read_unigram_archive(unigram_path, clustering)[0],
	)
	row_counts = {
		cast(tuple[int, int], matrix.shape)[0] for matrix in unigrams.values()
	}
	if len(row_counts) != 1:
		raise ValueError("All unigram matrices must have the same number of rows")

	started_at = perf_counter()
	for feature_number, name in enumerate(FEATURE_NAMES, start=1):
		logging.info(
			"Factorization stage %d/%d: WMF for %s",
			feature_number,
			len(FEATURE_NAMES) + 1,
			name,
		)

		factors = factors_default if name != "rhythmic" else factors_default // 2
		wmf_dataset_features(
			dataset_name,
			unigrams[name],
			alpha=alpha,
			lambda_=lambda_,
			factors=factors,
			output_name=f"wmf_{name}",
			overwrite=True,
		)

	qf_total = cast(
		sparse.csr_matrix,
		sparse.hstack(
			[unigrams[name] for name in FEATURE_NAMES],
		),
	)
	logging.info(
		"Factorization stage %d/%d: WMF for qf_total, shape=%s",
		len(FEATURE_NAMES) + 1,
		len(FEATURE_NAMES) + 1,
		qf_total.shape,
	)
	wmf_dataset_features(
		dataset_name,
		qf_total,
		alpha=alpha,
		lambda_=lambda_,
		factors=factors_default * 2,
		output_name="qf_total",
		overwrite=True,
	)
	logging.info(
		"Factorization stage completed in %.2f seconds",
		perf_counter() - started_at,
	)
	return output_dir


def postprocessing_embeddings(
	dataset_name: str = "fma_small",
	explained_variance: float = FINAL_PCA_TARGET,
	g1_filename: str = "G1.npz",
	output_filename: str = "embeddings.npz",
	overwrite: bool = False,
) -> Path:
	"""Scale and reduce G1 and WMF representations into final embeddings.

	For each feature, its G1 mean, G1 standard deviation, and WMF user factors
	are concatenated horizontally and transformed with a dedicated
	``RobustScaler``. The qf_total user factors form a separate scaled block.
	All scaled blocks retain the same audio-segment rows and are concatenated
	before a final PCA.
	"""
	if not 0 < explained_variance <= 1:
		raise ValueError("explained_variance must be in the interval (0, 1]")
	output_dir = get_output_dir(dataset_name)
	g1_path = output_dir / g1_filename
	output_path = output_dir / output_filename
	scalers_path = output_dir / "postprocessing_scalers.joblib"
	pca_path = output_dir / "postprocessing_pca.joblib"
	for path in (g1_path,):
		if not path.is_file():
			raise FileNotFoundError(f"G1 archive does not exist: {path}")
	for path in (output_path, scalers_path, pca_path):
		if path.exists() and not overwrite:
			raise FileExistsError(
				f"{path} already exists; pass overwrite=True to replace it"
			)

	started_at = perf_counter()
	blocks: list[np.ndarray] = []
	scalers: dict[str, RobustScaler] = {}
	segment_count: int | None = None

	with np.load(g1_path, allow_pickle=False) as g1_source:
		for name in (*FEATURE_NAMES, "qf_total"):
			factor_path = output_dir / f"wmf_{name}_user_factors.npz"
			if name == "qf_total":
				factor_path = output_dir / "qf_total_user_factors.npz"
			if not factor_path.is_file():
				raise FileNotFoundError(
					f"WMF user-factor archive does not exist: {factor_path}"
				)
			with np.load(factor_path, allow_pickle=False) as factor_source:
				user_factors = np.asarray(
					factor_source["user_factors"],
					dtype=np.float32,
				)
			if user_factors.ndim != 2:
				raise ValueError(
					f"Expected {name} user factors to be 2D, got "
					f"{user_factors.shape}"
				)
			if segment_count is None:
				segment_count = user_factors.shape[0]
			elif user_factors.shape[0] != segment_count:
				raise ValueError(
					f"{name} has {user_factors.shape[0]} rows, expected "
					f"{segment_count}"
				)

			if name == "qf_total":
				block = user_factors
			else:
				mean = np.asarray(g1_source[f"{name}_mean"], dtype=np.float32)
				std = np.asarray(g1_source[f"{name}_std"], dtype=np.float32)
				if mean.ndim != 2 or std.shape != mean.shape:
					raise ValueError(
						f"Invalid G1 arrays for {name}: mean={mean.shape}, "
						f"std={std.shape}"
					)
				if mean.shape[0] != user_factors.shape[0]:
					raise ValueError(
						f"G1 and WMF row counts differ for {name}: "
						f"{mean.shape[0]} vs {user_factors.shape[0]}"
					)
				block = np.hstack((mean, std, user_factors))

			scaler = RobustScaler()
			scaled_block = scaler.fit_transform(block).astype(
				np.float32,
				copy=False,
			)
			scalers[name] = scaler
			blocks.append(scaled_block)
			logging.info(
				"Postprocessing block %d/%d: %s, shape=%s",
				len(blocks),
				len(FEATURE_NAMES) + 1,
				name,
				block.shape,
			)

	combined = np.hstack(blocks).astype(np.float32, copy=False)
	pca = PCA(
		n_components=explained_variance,
		svd_solver="full",
	)
	embeddings = pca.fit_transform(combined).astype(np.float32, copy=False)

	atomic_write(
		output_path,
		lambda path: np.savez_compressed(
			file=str(path),
			embeddings=embeddings,
			explained_variance=np.asarray(explained_variance, dtype=np.float32),
		),
		".tmp.npz",
	)
	atomic_write(
		scalers_path,
		lambda path: joblib.dump(scalers, path),
		".tmp.joblib",
	)
	atomic_write(
		pca_path,
		lambda path: joblib.dump(pca, path),
		".tmp.joblib",
	)
	logging.info(
		"Postprocessing completed: combined shape=%s, embeddings shape=%s, "
		"components=%d in %.2f seconds",
		combined.shape,
		embeddings.shape,
		pca.n_components_,
		perf_counter() - started_at,
	)
	return output_path

def train_qfm_model(
	dataset_name: str = "fma_small",
	alpha: float = 5,
	lambda_: float = 0.5,
	factors: int = 128,
	windows_per_segment: int = 60,
	min_presence: float = 0.001,
	clustering: str = "kmeans",
	sample_rate: int = 22050,
	hop_size: int = 512,
	n_clusters: int = 8,
	extract_features: bool = False,
) -> Path:
	"""Train a QFM model on the specified dataset."""
	if (extract_features):
		extract_and_sample_features(
			dataset_name,
			n_samples=windows_per_segment,
			sample_rate=sample_rate,
			hop_size=hop_size,
		)
	compute_G1(
		dataset_name,
		overwrite=True,
	)
	train_quantization_stage(
		clustering=clustering,
		n_clusters=n_clusters,
		dataset_name=dataset_name,
	)
	train_factorization_stage(
		dataset_name,
		clustering=clustering,
		alpha=alpha,
		lambda_=lambda_,
		factors_default=factors,
		windows_per_segment=windows_per_segment,
		min_presence=min_presence,
	)
	return postprocessing_embeddings(
		dataset_name,
		overwrite=True,
	)

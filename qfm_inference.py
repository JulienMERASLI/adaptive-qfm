from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal, cast

import joblib
import numpy as np
from scipy import sparse
from tqdm import tqdm

from features import FeatureExtractor, _flatten_feature_by_dimension, _restore_feature_shape, convert_flattened_to_dict, flatten_dict
from qfm_training import FEATURE_NAMES, build_unigram_dataset_features, compute_G1, sample_dataset_feature_windows
from threadpoolctl import threadpool_limits

def _parallel_dict_map(mapping, function):
	with ThreadPoolExecutor() as executor:
		return dict(executor.map(function, mapping.items()))

def transform(X, transformation, feature_sets_kept, sparse_format=False, G1=False, dimensions=None):
	if (dimensions is None):
		dimensions = {key: X[key].shape[1] for key in X}
	flattened_X = flatten_dict(X, sparse=sparse_format)
	feature_sets_kept_G = {}
	if (G1):
		for feature_name in feature_sets_kept:
			feature_sets_kept_G[f"{feature_name}_mean"] = feature_sets_kept[feature_name]
			feature_sets_kept_G[f"{feature_name}_std"] = feature_sets_kept[feature_name]
	X_transformed = convert_flattened_to_dict(transformation(flattened_X), dimensions, feature_sets_kept_G if G1 else feature_sets_kept)
	if (sparse_format):
		for feature_name in X_transformed:
			X_transformed[feature_name] = sparse.csr_matrix(X_transformed[feature_name])
	return X_transformed

def infer_qfm_embeddings(features: dict[str, np.ndarray], model_path="training/fma_small", clustering_method="kmeans", to_return: Literal["G1", "unigram"] | None = None, transformation_G: Callable | None = None, transformation_U: Callable | None = None, feature_sets_kept={feature_name: True for feature_name in FEATURE_NAMES}) -> np.ndarray:
	if (to_return != "unigram"):
		def restore_feature(item):
			feature_name, feature = item
			return feature_name, _restore_feature_shape(feature, (feature.shape[0] // 60, feature.shape[1], 60))

		original_features = _parallel_dict_map(features, restore_feature)
		g1 = compute_G1.compute(original_features).result
		if (to_return == "G1"):
			return g1 # pyright: ignore[reportReturnType]
		if (transformation_G is not None):
			g1 = transform(g1, transformation_G, feature_sets_kept, G1=True)
	scalers = joblib.load(f"{model_path}/scalers.joblib")
	pcas = joblib.load(f"{model_path}/pcas.joblib")
	clustering = joblib.load(f"{model_path}/{clustering_method}.joblib")


	scaled_features = {}

	for feature_name in features:
		scaled_features[feature_name] = scalers[feature_name].transform(features[feature_name])


	whitened_features = {	
		feature_name: pcas[feature_name].transform(scaled_features[feature_name]) 
		for feature_name in features
	}


	if (clustering_method == "kmeans"):
		cluster_assignments = {
			feature_name: clustering[feature_name].predict(whitened_features[feature_name].astype(np.float32, copy=False)) 
			for feature_name in features
		}
	else:
		cluster_assignments = {
			feature_name: clustering[feature_name].predict_proba(whitened_features[feature_name]) 
			for feature_name in features
		}



	unigrams = build_unigram_dataset_features.compute(cluster_assignments, clustering=clustering_method, windows_per_segment=60, inference=True, unigram_archive_path=f"{model_path}/{clustering_method}_unigrams.npz").result("")[1]
	if (to_return == "unigram"):
		return unigrams # pyright: ignore[reportReturnType]

	if (transformation_U is not None):
		unigrams = transform(unigrams, transformation_U, feature_sets_kept, sparse_format=True)

	wmf_embeddings = compute_wmf(unigrams, model_path=model_path)

	postprocessing_scalers = joblib.load(f"{model_path}/postprocessing_scalers.joblib")
	postprocessing_pca = joblib.load(f"{model_path}/postprocessing_pca.joblib")

	return postprocessing(postprocessing_scalers, postprocessing_pca, g1, wmf_embeddings) # pyright: ignore[reportPossiblyUnboundVariable]

def compute_wmf(unigrams, model_path="training/fma_small"):
	qf_total = cast(
		sparse.csr_matrix,
		sparse.hstack(list(unigrams.values())),
	)
	wmf_inputs = dict(unigrams)
	wmf_inputs["qf_total"] = qf_total

	def compute_embedding(item):
		feature_name, unigram = item
		model_name = "qf_total" if feature_name == "qf_total" else f"wmf_{feature_name}"
		wmf_model = joblib.load(f"{model_path}/{model_name}.joblib")
		new_user_ids = np.arange(
			len(wmf_model.user_factors),
			len(wmf_model.user_factors) + unigram.shape[0], # type: ignore
		)

		return feature_name, wmf_model.recalculate_user(new_user_ids, unigram)

	with threadpool_limits(limits=1, user_api="blas"):
		return _parallel_dict_map(wmf_inputs, compute_embedding)
	

def postprocessing(scalers, pca, g1, wmf_embeddings):
	scaled_blocks = {}
	for feature_name in wmf_embeddings:
		if (feature_name != "qf_total"):
			block = np.hstack((g1[f"{feature_name}_mean"], g1[f"{feature_name}_std"], wmf_embeddings[feature_name]))
		else:
			block = wmf_embeddings[feature_name]
		
		scaled_blocks[feature_name] = scalers[feature_name].transform(block)


	combined = np.hstack([scaled_blocks[feature_name] for feature_name in scaled_blocks]).astype(np.float32, copy=False)

	final_embeddings = pca.transform(combined)

	return final_embeddings

def extract_and_infer_qfm_embeddings(audio_inputs: list[np.ndarray], model_path="training/fma_small", clustering_method="kmeans", sample_rate=22050, hop_size=512, features_path: str | None = None, to_return: Literal["G1", "unigram"] | None = None, transformation_G: Callable | None = None, transformation_U: Callable | None = None, feature_sets_kept={feature_name: True for feature_name in FEATURE_NAMES}) -> np.ndarray:
	"""
	Infer QFM embeddings for a list of audio inputs.

	Parameters:
	- audio_inputs: List of numpy arrays representing audio signals.
	- model_path: Path to the folder containing the pre-trained QFM model.
	- clustering_method: Clustering method used for training ("kmeans" or "gmm").
	- sample_rate: Sample rate of the audio signals.
	- hop_size: Hop size for feature extraction.

	Returns:
	- embeddings: List of inferred embeddings for each audio input.
	"""
	features = extract_features(audio_inputs, sample_rate, hop_size, features_path)

	def flatten_feature(item):
		feature_name, feature = item
		return feature_name, _flatten_feature_by_dimension(feature)

	flattened_features = _parallel_dict_map(features, flatten_feature)

	return infer_qfm_embeddings(flattened_features, model_path=model_path, clustering_method=clustering_method, to_return=to_return, transformation_G=transformation_G, transformation_U=transformation_U, feature_sets_kept=feature_sets_kept)
	

def extract_features(audio_inputs, sample_rate, hop_size, features_path, sets_included={feature_name: True for feature_name in FEATURE_NAMES}):
	features = None
	if (features_path is not None) and Path(features_path).exists():
		with np.load(features_path) as cached_features:
			features = {
				feature_name: cached_features[feature_name]
				for feature_name in cached_features
			}

	else:
		started_at = perf_counter()
		with ThreadPoolExecutor() as executor:
			futures = [
				executor.submit(
					FeatureExtractor,
					audio,
					sr=sample_rate,
					hopsize=hop_size,
					sets_included=sets_included,
				)
				for audio in audio_inputs
			]
			future_indices = {future: index for index, future in enumerate(futures)}

			feature_extractors: list[None | FeatureExtractor] = [None] * len(futures)

			for future in (
				tqdm(as_completed(futures), total=len(futures), desc="Extracting features")
			):
				feature_extractors[future_indices[future]] = future.result()
		features = {
			feature_name: np.array([getattr(extractor, feature_name) for extractor in feature_extractors]) 
			for feature_name in sets_included
		}


		if (features_path is not None):
			np.savez_compressed(
				features_path,
				**features, # pyright: ignore[reportArgumentType]
			)
		print(f"Feature extraction completed in {perf_counter() - started_at:.2f} seconds.")


	sampled_features = sample_dataset_feature_windows.compute(features, n_samples=60).result

	return sampled_features
	
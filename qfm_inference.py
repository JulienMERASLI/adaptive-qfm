from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import cast

import joblib
import numpy as np
from scipy import sparse
from tqdm import tqdm

from features import FeatureExtractor, _flatten_feature_by_dimension
from qfm_training import FEATURE_NAMES, build_unigram_dataset_features, compute_G1, sample_dataset_feature_windows

def infer_qfm_embedding(audio_inputs: list[np.ndarray], model_path="training/fma_small", clustering_method="kmeans", sample_rate=22050, hop_size=512, features_path: str | None = None) -> np.ndarray:
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
	n_inputs = len(audio_inputs)
	if (features_path is not None) and Path(features_path).exists():
		features = np.load(features_path)
		print(f"Features loaded from {features_path}")
	else:
		started_at = perf_counter()
		with ThreadPoolExecutor() as executor:
			futures = [
				executor.submit(
					FeatureExtractor,
					audio,
					sr=sample_rate,
					hopsize=hop_size,
				)
				for audio in audio_inputs
			]

			feature_extractors: list[None | FeatureExtractor] = [None] * len(futures)

			for i, future in enumerate(
				tqdm(as_completed(futures), total=len(futures), desc="Extracting features")
			):
				feature_extractors[i] = future.result()
		features = {
			feature_name: np.array([getattr(extractor, feature_name) for extractor in feature_extractors]) 
			for feature_name in FEATURE_NAMES
		}

		print(f"Features extracted in {perf_counter() - started_at:.2f} seconds")

		if (features_path is not None):
			np.savez_compressed(features_path, **features) # pyright: ignore[reportArgumentType]
			print(f"Features saved to {features_path}")

	g1 = compute_G1.compute(features).result

	print("G1 computed")

	sampled_features = sample_dataset_feature_windows.compute(features, n_samples=60)

	print("Windows sampled")
	scalers = joblib.load(f"{model_path}/scalers.joblib")
	pcas = joblib.load(f"{model_path}/pcas.joblib")
	clustering = joblib.load(f"{model_path}/{clustering_method}.joblib")

	scaled_features = {}

	for feature_name in FEATURE_NAMES:
		flattened = _flatten_feature_by_dimension(sampled_features.result[feature_name])
		scaled_features[feature_name] = scalers[feature_name].transform(flattened)

	print("Features scaled")

	whitened_features = {	
		feature_name: pcas[feature_name].transform(scaled_features[feature_name]) 
		for feature_name in FEATURE_NAMES
	}

	print("Features whitened")

	if (clustering_method == "kmeans"):
		cluster_assignments = {
			feature_name: clustering[feature_name].predict(whitened_features[feature_name].astype(np.float32, copy=False)) 
			for feature_name in FEATURE_NAMES
		}
	else:
		cluster_assignments = {
			feature_name: clustering[feature_name].predict_proba(whitened_features[feature_name]) 
			for feature_name in FEATURE_NAMES
		}

	print("Cluster assignments computed")

	unigrams = build_unigram_dataset_features.compute(cluster_assignments, clustering=clustering_method, windows_per_segment=60, inference=True, unigram_archive_path=f"{model_path}/{clustering_method}_unigrams.npz").result("")[1]

	if (clustering_method == "gmm"):
		for feature_name in unigrams:
			unigrams[feature_name] = sparse.csr_matrix(unigrams[feature_name])

	print("Unigrams computed")

	wmf_embeddings = {}
	for feature_name in FEATURE_NAMES:
		wmf_model = joblib.load(f"{model_path}/wmf_{feature_name}.joblib")
		new_user_ids = np.arange(
			len(wmf_model.user_factors),
			len(wmf_model.user_factors) + unigrams[feature_name].shape[0], # type: ignore
		)		
		wmf_model.partial_fit_users(new_user_ids, unigrams[feature_name])
		wmf_embeddings[feature_name] = wmf_model.user_factors[-n_inputs:]

	qf_total = cast(
		sparse.csr_matrix,
		sparse.hstack(
			[unigrams[name] for name in FEATURE_NAMES],
		),
	)
	
	qf_total_wmf_model = joblib.load(f"{model_path}/qf_total.joblib")
	new_user_ids = np.arange(
		len(qf_total_wmf_model.user_factors),
		len(qf_total_wmf_model.user_factors) + qf_total.shape[0], # type: ignore
	)		

	qf_total_wmf_model.partial_fit_users(new_user_ids, qf_total)
	wmf_embeddings["qf_total"] = qf_total_wmf_model.user_factors[-n_inputs:]

	print("WMF embeddings computed")

	postprocessing_scalers = joblib.load(f"{model_path}/postprocessing_scalers.joblib")
	postprocessing_pca = joblib.load(f"{model_path}/postprocessing_pca.joblib")

	scaled_blocks = {}
	for feature_name in wmf_embeddings:
		if (feature_name != "qf_total"):
			block = np.hstack((g1[f"{feature_name}_mean"], g1[f"{feature_name}_std"], wmf_embeddings[feature_name]))
		else:
			block = wmf_embeddings[feature_name]
		
		scaled_blocks[feature_name] = postprocessing_scalers[feature_name].transform(block)

	print("Blocks scaled")

	combined = np.hstack([scaled_blocks[feature_name] for feature_name in scaled_blocks]).astype(np.float32, copy=False)

	final_embeddings = postprocessing_pca.transform(combined)
	print("Final PCA computed")

	return final_embeddings
	
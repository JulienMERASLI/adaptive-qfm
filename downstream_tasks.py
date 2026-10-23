from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Generic, TypeVar

import joblib
import librosa
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, rand_score

from features import SEGMENT_DURATION_SECONDS
from helpers import atomic_write, chunk_mean
from qfm_inference import extract_and_infer_qfm_embeddings


T = TypeVar("T")

@dataclass
class Dataset(Generic[T]):
	X_train: list[np.ndarray]
	X_valid: list[np.ndarray] | None
	X_test: list[np.ndarray]
	y_train: list[T]
	y_valid: list[T] | None
	y_test: list[T]

N_chunks_GTZAN = (30 - SEGMENT_DURATION_SECONDS) // (SEGMENT_DURATION_SECONDS - 3) + 1  # 9 seconds clips with 3 seconds overlap

def extract_gtzan(path: str = "datasets/GTZAN", sample_rate: int = 22050, y_only=False) -> Dataset[str]:
	with open(f"{path}/train_filtered.txt", "r") as f:
		train_files = f.readlines()
	
	with open(f"{path}/valid_filtered.txt", "r") as f:
		valid_files = f.readlines()

	with open(f"{path}/test_filtered.txt", "r") as f:
		test_files = f.readlines()

	def get_genre(filename: str) -> str:
		return filename.split("/")[0]

	files = {
		"train": train_files,
		"valid": valid_files,
		"test": test_files,
	}

	X = {
		"train": [],
		"valid": [],
		"test": [],
	}

	Y = {
		"train": [],
		"valid": [],
		"test": [],
	}

	if (not y_only):
		for split in ["train", "valid", "test"]:
			for f in files[split]:
				try:
					audio, sr = librosa.load(f"{path}/Data/genres_original/{f.strip()}", sr=sample_rate)
					extract_chunks(sample_rate, get_genre(f), X[split], Y[split], audio)
				except Exception as e:
					print(f"Error loading audio file {f.strip()} for {split} split: {e}")

	return Dataset(
		X_train=X["train"],
		X_valid=X["valid"],
		X_test=X["test"],
		y_train=Y["train"],
		y_valid=Y["valid"],
		y_test=Y["test"],
	)

def extract_chunks(sample_rate, y, X, Y, audio):
	for i in range(N_chunks_GTZAN):
		X.append(audio[6*i*sample_rate:(6*i+9)*sample_rate])  # 9 seconds clips with 3 seconds overlap
		Y.append(y)

def compute_embeddings_gtzan(dataset_path: str = "datasets/GTZAN", sample_rate: int = 22050, hop_size: int = 512, qfm_path = "training/fma_small", clustering_method = "kmeans"):
	started_at = perf_counter()

	dataset = extract_gtzan(dataset_path, sample_rate)
	
	print(f"Dataset loaded in {perf_counter() - started_at:.2f} seconds")
	started_at = perf_counter()

	X_train, X_valid, X_test = dataset.X_train, dataset.X_valid, dataset.X_test
	y_train, y_valid, y_test = dataset.y_train, dataset.y_valid, dataset.y_test

	if X_valid is not None and y_valid is not None:
		X_train.extend(X_valid)
		y_train.extend(y_valid)

	dataset_name = qfm_path.split("/")[-1]

	train_embeddings = extract_and_infer_qfm_embeddings(X_train, model_path=qfm_path, clustering_method=clustering_method, sample_rate=sample_rate, hop_size=hop_size, features_path=f"training/gtzan_train_features.npz")

	np.save(f"training/gtzan_train_embeddings_{clustering_method}_{dataset_name}.npy", train_embeddings)

	print(f"Train embeddings computed in {perf_counter() - started_at:.2f} seconds")
	started_at = perf_counter()
	
	test_embeddings = extract_and_infer_qfm_embeddings(X_test, model_path=qfm_path, clustering_method=clustering_method, sample_rate=sample_rate, hop_size=hop_size, features_path=f"training/gtzan_test_features.npz")

	np.save(f"training/gtzan_test_embeddings_{clustering_method}_{dataset_name}.npy", test_embeddings)

	print(f"Test embeddings computed in {perf_counter() - started_at:.2f} seconds")

	return train_embeddings, test_embeddings, y_train, y_test



def genre_classification(dataset_path: str = "datasets/GTZAN", sample_rate: int = 22050, hop_size: int = 512, qfm_path = "training/fma_small", clustering_method = "kmeans"):
	dataset_name = qfm_path.split("/")[-1]

	if not Path(f"training/gtzan_train_embeddings_{clustering_method}_{dataset_name}.npy").exists() or not Path(f"training/gtzan_test_embeddings_{clustering_method}_{dataset_name}.npy").exists():
		train_embeddings, test_embeddings, y_train, y_test = compute_embeddings_gtzan(dataset_path, sample_rate, hop_size, qfm_path, clustering_method)
	else:
		train_embeddings = np.load(f"training/gtzan_train_embeddings_{clustering_method}_{dataset_name}.npy")
		test_embeddings = np.load(f"training/gtzan_test_embeddings_{clustering_method}_{dataset_name}.npy")
		dataset = extract_gtzan(dataset_path, sample_rate)
		y_train, y_test = dataset.y_train, dataset.y_test
		if dataset.y_valid is not None:
			y_train.extend(dataset.y_valid)

	
	started_at = perf_counter()

	model = LogisticRegression(max_iter=5000).fit(train_embeddings, y_train)

	print(f"Model trained in {perf_counter() - started_at:.2f} seconds")
	started_at = perf_counter()

	atomic_write(Path(f"training/genre_classification_model_{clustering_method}_{dataset_name}.joblib"), lambda path: joblib.dump(model, path), ".tmp.joblib")

	print(f"Model saved in {perf_counter() - started_at:.2f} seconds")
	started_at = perf_counter()
	predictions = model.predict_proba(test_embeddings)

	print(f"Predictions computed in {perf_counter() - started_at:.2f} seconds")
	accuracy = compute_accuracy_genre_classification(y_test, predictions, model)

	return predictions, accuracy
	

def compute_accuracy_genre_classification(y_test, predictions, model):
	classes = model.classes_
	y_test_original = y_test[::N_chunks_GTZAN]
	mean_predictions = chunk_mean(predictions, N_chunks_GTZAN)
	predictions_original = classes[np.argmax(mean_predictions, axis=1)]
	return accuracy_score(y_test_original, predictions_original)

def genre_clustering(dataset_path: str = "datasets/GTZAN", sample_rate: int = 22050, hop_size: int = 512, qfm_path = "training/fma_small", clustering_method = "kmeans"):
	dataset_name = qfm_path.split("/")[-1]
	if not Path(f"training/gtzan_train_embeddings_{clustering_method}_{dataset_name}.npy").exists() or not Path(f"training/gtzan_test_embeddings_{clustering_method}_{dataset_name}.npy").exists():
		train_embeddings, test_embeddings, y_train, y_test = compute_embeddings_gtzan(dataset_path, sample_rate, hop_size, qfm_path, clustering_method)
	else:
		train_embeddings = np.load(f"training/gtzan_train_embeddings_{clustering_method}_{dataset_name}.npy")
		test_embeddings = np.load(f"training/gtzan_test_embeddings_{clustering_method}_{dataset_name}.npy")
		dataset = extract_gtzan(dataset_path, sample_rate)
		y_train, y_test = dataset.y_train, dataset.y_test
		if dataset.y_valid is not None:
			y_train.extend(dataset.y_valid)
	

	if not Path(f"training/genre_clustering_model_{clustering_method}_{dataset_name}.joblib").exists():
		started_at = perf_counter()

		model = KMeans(n_clusters=len(set(y_train))).fit(train_embeddings)

		print(f"Model trained in {perf_counter() - started_at:.2f} seconds")
		started_at = perf_counter()

		atomic_write(Path(f"training/genre_clustering_model_{clustering_method}_{dataset_name}.joblib"), lambda path: joblib.dump(model, path), ".tmp.joblib")

		print(f"Model saved in {perf_counter() - started_at:.2f} seconds")

	else:
		model = joblib.load(f"training/genre_clustering_model_{clustering_method}_{dataset_name}.joblib")
		print("Model loaded")

	started_at = perf_counter()
	predictions = model.predict(test_embeddings)

	print(f"Predictions computed in {perf_counter() - started_at:.2f} seconds")

	clusterpurity, randindex = compute_metrics_genre_clustering(y_test, predictions)

	return predictions, (clusterpurity, randindex)

def compute_metrics_genre_clustering(y_test, predictions):
	# compute contingency matrix (also called confusion matrix)
	contingency_matrix = metrics.cluster.contingency_matrix(y_test, predictions)
	# compute purity
	purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)  # type: ignore

	randindex = rand_score(predictions, y_test)

	return purity, randindex



# genre_classification_predictions, genre_classification_accuracy = genre_classification(qfm_path="training/fma_small", clustering_method="kmeans")
# print(genre_classification_accuracy)

# genre_clustering_predictions, (genre_clustering_purity, genre_clustering_randindex) = genre_clustering(qfm_path="training/fma_small", clustering_method="gmm")
# print(f"Genre clustering purity: {genre_clustering_purity}, rand index: {genre_clustering_randindex}")


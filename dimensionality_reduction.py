from itertools import combinations
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr
from sklearn.linear_model import LogisticRegression
import _spsa
from tqdm import tqdm

from downstream_tasks import compute_accuracy_genre_classification, extract_gtzan
from features import FEATURE_NAMES, FeatureExtractor, _flatten_feature_by_dimension, flatten_dict
from qfm_inference import compute_wmf, extract_and_infer_qfm_embeddings, infer_qfm_embeddings, postprocessing, transform
from qfm_training import _read_unigram_archive

def T(feature_sets_kept, feature_sets_length):
	blocks = []
	J = [i for i in range(len(feature_sets_kept)) if list(feature_sets_kept.values())[i]]
	for i_prime in J:
		blocks_iprime = []
		for i in range(len(feature_sets_length)):
			size_i = list(feature_sets_length.values())[i]
			size_iprime = list(feature_sets_length.values())[i_prime]
			if i == i_prime:
				block = np.eye(size_i)
			else: 
				block = np.zeros((size_iprime, size_i))
			blocks_iprime.append(block)
		blocks.append(blocks_iprime)

	return np.block(blocks)

def build_T_tilde(params, feature_sets_kept, feature_sets_length):
	blocks_A = []
	blocks_b = []
	J = [i for i in range(len(feature_sets_kept)) if list(feature_sets_kept.values())[i]]
	start = 0
	for i in range(len(feature_sets_length)):
		blocks_A_i = []
		size_i = list(feature_sets_length.values())[i]
		for i_prime in J:
			size_iprime = list(feature_sets_length.values())[i_prime]
			if i == i_prime:
				block_A = np.eye(size_i)
			else: 
				if (i in J):
					block_A = np.zeros((size_i, size_iprime))
				else:
					end = start + size_i * size_iprime
					block_A = params[start:end].reshape((size_i, size_iprime))
					start = end
			blocks_A_i.append(block_A)
		blocks_A.append(blocks_A_i)

		if (i in J):
			block_b = np.zeros((size_i, 1))
		else:
			end = start + size_i
			block_b = params[start:end].reshape((size_i, 1))
			start = end
		blocks_b.append(block_b)

	return np.block(blocks_A), np.vstack(blocks_b)

def build_T_tilde_adjoint(GA, Gb, feature_sets_kept, feature_sets_length):

    gradients = []

    J = [
        i for i in range(len(feature_sets_kept))
        if list(feature_sets_kept.values())[i]
    ]

    row_start = 0

    for i in range(len(feature_sets_length)):

        size_i = list(feature_sets_length.values())[i]

        col_start = 0

        for i_prime in J:

            size_iprime = list(feature_sets_length.values())[i_prime]

            if i == i_prime:
                pass

            else:
                if i in J:
                    pass

                else:
                    block_grad = GA[
                        row_start:row_start + size_i,
                        col_start:col_start + size_iprime
                    ]

                    gradients.append(block_grad.ravel())

            col_start += size_iprime

        if i in J:
            pass

        else:
            block_grad_b = Gb[
                row_start:row_start + size_i, :
            ]

            gradients.append(block_grad_b.ravel())

        row_start += size_i

    return np.concatenate(gradients)


def build_transformation(A, b=None):
	if b is None:
		return lambda X, N: X @ A.T
	else:
		return lambda X, N: X @ A.T + np.ones((N, 1)) @ b.T

def compute_delta(X, X_tilde):
	deltaX = {
		feature_name: X_tilde[feature_name] - X[feature_name]
		for feature_name in X
	}
	return deltaX

def get_N_params(feature_sets_kept, feature_sets_length):
	s = len(feature_sets_kept)
	J = [i for i in range(len(feature_sets_kept)) if list(feature_sets_kept.values())[i]]
	Kprime = sum(feature_sets_length.values())
	Ktilde = sum(feature_sets_length[feature_name] for feature_name in feature_sets_kept if feature_sets_kept[feature_name])
	return Ktilde * (Kprime - Ktilde), Kprime - Ktilde # blocks for i,j in J are identity for A, and 0 for b.

def test_pseudoinverse(dataset, model, feature_sets_kept, Mi, Ki, model_type = lambda: LogisticRegression(max_iter=50000)):
	A_G = T(feature_sets_kept, Mi)
	A_G_tilde = np.transpose(A_G)
	transformation_G = build_transformation(A_G_tilde @ A_G)

	A_U = T(feature_sets_kept, Ki)
	A_U_tilde = np.transpose(A_U)
	transformation_U = build_transformation(A_U_tilde @ A_U)

	if (model == None):
		train_embeddings = extract_and_infer_qfm_embeddings(dataset.X_train, features_path=f"training/gtzan_train_features.npz", transformation_G=transformation_G, transformation_U=transformation_U)
		model = model_type()
		model.fit(train_embeddings, dataset.y_train)
	test_embeddings = extract_and_infer_qfm_embeddings(dataset.X_test, features_path=f"training/gtzan_test_features.npz", transformation_G=transformation_G, transformation_U=transformation_U)
	predictions = model.predict_proba(test_embeddings)
	
	accuracy = compute_accuracy_genre_classification(dataset.y_test, predictions, model)
	return accuracy

def solve_matrix_least_squares(A1, B1, A2, B2, Y, feature_sets_kept, feature_sets_length, x0=None):
	# solve argmin ||A1 X1 B1 + A2 X2 B2 - Y||_F

	# Dimensions:
	# A1: (m, p1), X1: (p1, q1), B1: (q1, n)
	# A2: (m, p2), X2: (p2, q2), B2: (q2, n)
	# Y:  (m, n)

	m, p1 = A1.shape
	q1, n = B1.shape
	_, p2 = A2.shape
	q2, _ = B2.shape

	dtype = np.result_type(A1, B1, A2, B2, Y)

	n1 = p1 * q1
	n2 = p2 * q2
	n_params_A, n_params_b = get_N_params(feature_sets_kept, feature_sets_length)
	nvars = n_params_A + n_params_b

	A0, b0 = build_T_tilde(np.zeros(nvars), feature_sets_kept, feature_sets_length)

	def matvec(x):
		A, b = build_T_tilde(x, feature_sets_kept, feature_sets_length)

		X1 = (A - A0).T
		X2 = (b - b0).T

		R = A1 @ X1 @ B1 + A2 @ X2 @ B2

		return R.ravel()

	def rmatvec(z):
		Z = z.reshape(m, n)

		GA = B1 @ Z.T @ A1
		Gb = B2 @ Z.T @ A2

		return build_T_tilde_adjoint(GA, Gb, feature_sets_kept, feature_sets_length)
	
	# Matrix-free linear operator
	L = LinearOperator(
		shape=(m * n, nvars),
		matvec=matvec, # type: ignore
		rmatvec=rmatvec, # type: ignore
		dtype=dtype
	)

	c = matvec(np.zeros(nvars))

	Y_adjusted = Y.ravel() - c

	# Solve min ||L x - Y||
	result = lsqr(L, Y_adjusted, x0=x0)

	x = result[0]

	X1, X2 = build_T_tilde(x, feature_sets_kept, feature_sets_length)

	return X1, X2

def optimization_G(features, feature_sets_kept, Mi, transformation_U, scalers, pca, old_T_G_tilde=None):
	if (len([feature_name for feature_name in feature_sets_kept if feature_sets_kept[feature_name]]) == len(feature_sets_kept)):
		M = sum(Mi.values())
		return np.eye(M), np.zeros((M, 1))
	started_at = perf_counter()

	G = infer_qfm_embeddings(features, to_return="G1", feature_sets_kept=feature_sets_kept)	
	
	T_G = T(feature_sets_kept, Mi)

	N = list(G.values())[0].shape[0] # pyright: ignore[reportAttributeAccessIssue]
	ones = np.ones((N, 1))

	all_feature_sets = {feature_name: True for feature_name in FEATURE_NAMES}

	U = infer_qfm_embeddings(features, to_return="unigram", feature_sets_kept=feature_sets_kept)
	U_tilde = transform(U, lambda X: transformation_U(X, N), all_feature_sets, sparse_format=True) if transformation_U is not None else U

	W = compute_wmf(U)
	W_tilde = compute_wmf(U_tilde)
	deltaW = compute_delta(W, W_tilde)

	G_kept = transform(G, lambda X: build_transformation(T_G)(X, N), feature_sets_kept, G1=True)

	indices_G = []
	indices_W = []
	start = 0
	for feature_name in FEATURE_NAMES:
		end = start + Mi[feature_name]
		indices_G.append((start, end))
		start = end + deltaW[feature_name].shape[1]
		indices_W.append((end, start))

	end = start + deltaW["qf_total"].shape[1]
	indices_W.append((start, end))

	IQR_inverse = np.linalg.inv(np.diag(np.concatenate([scalers[feature_name].scale_ for feature_name in FEATURE_NAMES] + [scalers["qf_total"].scale_])))

	R = pca.components_.T

	Rprime = IQR_inverse @ R
	R_Gprime = np.concatenate([Rprime[start:end] for start, end in indices_G])
	R_Wprime = np.concatenate([Rprime[start:end] for start, end in indices_W])
	A1 = flatten_dict(G_kept)
	B1 = R_Gprime
	A2 = ones
	B2 = R_Gprime
	Y = flatten_dict(G) @ R_Gprime - flatten_dict(deltaW) @ R_Wprime # pyright: ignore[reportArgumentType]

	T_G = T(feature_sets_kept, Mi)

	if (old_T_G_tilde is not None):
		old_A_G_tilde, old_b_G_tilde = old_T_G_tilde
		x0 = build_T_tilde_adjoint(old_A_G_tilde, old_b_G_tilde, feature_sets_kept, Mi)
	else:
		x0 = None
	A_G_tilde, b_G_tilde = solve_matrix_least_squares(A1, B1, A2, B2, Y, feature_sets_kept, Mi, x0=x0)

	print(f"Optimization for G completed in {perf_counter() - started_at:.2f} seconds.")
	return (A_G_tilde, b_G_tilde)

def optimization_U(features, feature_sets_kept, Ki, transformation_G, scalers, pca, n_iterations=100, old_T_U_tilde=None):
	N_params_A, N_params_b = get_N_params(feature_sets_kept, Ki)
	if (N_params_A + N_params_b == 0):
		return build_T_tilde([], feature_sets_kept, Ki)
	
	started_at = perf_counter()

	U = infer_qfm_embeddings(features, to_return="unigram", feature_sets_kept=feature_sets_kept)	
	W = compute_wmf(U)
	
	T_U = T(feature_sets_kept, Ki)

	N = list(W.values())[0].shape[0]

	all_feature_sets = {feature_name: True for feature_name in FEATURE_NAMES}

	G = infer_qfm_embeddings(features, to_return="G1", feature_sets_kept=feature_sets_kept)
	G_tilde = transform(G, lambda X: transformation_G(X, N), all_feature_sets, G1=True) if transformation_G is not None else G

	U_kept = transform(U, lambda X: build_transformation(T_U)(X, N), feature_sets_kept, sparse_format=True)

	def to_optimize(T_U_tilde_parameters):
		A_U_tilde, b_U_tilde = build_T_tilde(T_U_tilde_parameters, feature_sets_kept, Ki)
		U_tilde = transform(U_kept, lambda X: build_transformation(A_U_tilde, b_U_tilde)(X, N), all_feature_sets, sparse_format=True, dimensions={key: U[key].shape[1] for key in U})
		W_tilde = compute_wmf(U_tilde)

		E = postprocessing(scalers, pca, G, W)
		E_tilde = postprocessing(scalers, pca, G_tilde, W_tilde)
		deltaE = E_tilde - E
		L = np.linalg.norm(deltaE)
		return L

	if (old_T_U_tilde is not None):
		old_A_U_tilde, old_b_U_tilde = old_T_U_tilde
		x0 = build_T_tilde_adjoint(old_A_U_tilde, old_b_U_tilde, feature_sets_kept, Ki)
	else:
		x0 = np.hstack([np.zeros(N_params_A), -np.ones(N_params_b)])
	result = _spsa.minimize(to_optimize, x0, iterations=n_iterations)

	T_tilde = build_T_tilde(result, feature_sets_kept, Ki)
	print(f"Optimization for U completed in {perf_counter() - started_at:.2f} seconds.")
	return T_tilde

def optimization_G_and_U(features, feature_sets_kept, Mi, Ki, scalers, pca, n_iterations_spsa=100, n_repeats=2):
	started_at = perf_counter()
	T_G = T(feature_sets_kept, Mi)
	T_U = T(feature_sets_kept, Ki)

	A_G_tilde = T_G.T
	b_G_tilde = np.zeros((T_G.shape[1], 1))

	A_U_tilde = T_U.T
	b_U_tilde = np.zeros((T_U.shape[1], 1))

	T_U_tilde = build_transformation(A_U_tilde, b_U_tilde)
	T_G_tilde = build_transformation(A_G_tilde, b_G_tilde)

	for _ in tqdm(range(n_repeats)):
		A_G_tilde, b_G_tilde = optimization_G(features, feature_sets_kept, Mi, lambda U, N: T_U_tilde(U @ T_U.T, N), scalers, pca, old_T_G_tilde=(A_G_tilde, b_G_tilde))
		T_G_tilde = lambda G, N: G @ A_G_tilde.T + np.ones((N, 1)) @ b_G_tilde.T
		A_U_tilde, b_U_tilde = optimization_U(features, feature_sets_kept, Ki, lambda G, N: T_G_tilde(G @ T_G.T, N), scalers, pca, n_iterations=n_iterations_spsa, old_T_U_tilde=(A_U_tilde, b_U_tilde))
		T_U_tilde = lambda U, N: U @ A_U_tilde.T + np.ones((N, 1)) @ b_U_tilde.T 

	print(f"Optimization for G and U completed in {perf_counter() - started_at:.2f} seconds.")
	return T_G_tilde, T_U_tilde

def get_Mi_Ki(model_path = "training/fma_small", clustering_method = "kmeans"):
	unigram_archive_path = f"{model_path}/{clustering_method}_unigrams.npz"
	example_audio = np.random.rand(22050 * 9)
	example_features = FeatureExtractor(example_audio)
	Mi = {feature_name: 2*len(getattr(example_features, feature_name)) for feature_name in FEATURE_NAMES}
	unigram_archive = _read_unigram_archive(Path(unigram_archive_path), "kmeans")[1]
	Ki = {feature_name: len(unigram_archive[feature_name]) for feature_name in FEATURE_NAMES}

	return Mi, Ki

def compute_importances_subsets(dataset, model, Mi, Ki, fn, max_subset_size = len(FEATURE_NAMES) - 1, model_type = lambda: LogisticRegression(max_iter=50000), *args, **kwargs):
	feature_sets = {feature_name: True for feature_name in FEATURE_NAMES}
	accuracies = {}
	subsets = [list(combinations(feature_sets.keys(), r)) for r in range(1, max_subset_size + 1)]  
	subsets = [list(sublist) for g in subsets for sublist in g]  

	for subset in tqdm(subsets):
		feature_sets_kept = feature_sets.copy()
		for feature_name in subset:
			feature_sets_kept[feature_name] = False
		accuracies["_".join(subset)] = fn(dataset, model, feature_sets_kept, Mi, Ki, model_type=model_type, *args, **kwargs)

	baseline = fn(dataset, model, feature_sets, Mi, Ki)
	importances = {"_".join(subset): (baseline - accuracies["_".join(subset)]) / baseline for subset in subsets}

	return accuracies, importances

def test_first_order_approximation(importances, max_subset_size = len(FEATURE_NAMES) - 1):
	feature_sets = {feature_name: True for feature_name in FEATURE_NAMES}
	subsets = [list(combinations(feature_sets.keys(), r)) for r in range(1, max_subset_size + 1)]  
	errors = []
	absolute_errors = []
	for i in range(2, max_subset_size + 1):
		errors_i = []
		absolute_errors_i = []
		for subset in subsets[i-1]:
			subset_key = "_".join(subset)
			approx_importance = sum(importances[feature_name] for feature_name in subset)
			actual_importance = importances[subset_key]
			error = (actual_importance - approx_importance) / actual_importance
			absolute_error = abs(error)
			errors_i.append(error)
			absolute_errors_i.append(absolute_error)
		errors.append(np.mean(errors_i))
		absolute_errors.append(np.mean(absolute_errors_i))

	def compute_weighted_mean(errors):
		total_samples = sum(len(subset) for subset in subsets[1:])
		weighted_mean = sum(errors[i] * len(subsets[i+1]) for i in range(len(errors))) / total_samples
		return weighted_mean
	return {
		"errors": errors,
		"absolute_errors": absolute_errors,
		"mean_error": compute_weighted_mean(errors),
		"mean_absolute_error": compute_weighted_mean(absolute_errors),
	}



def load_model_and_features(n_samples=100):
	scalers = joblib.load(f"training/fma_small/postprocessing_scalers.joblib")
	pca = joblib.load(f"training/fma_small/postprocessing_pca.joblib")
	with np.load("training/fma_small/sampled_features.npz") as cached_features:
		indices = np.random.choice(len(list(cached_features.values())[0]), size=n_samples, replace=False)
		features = {
			feature_name: cached_features[feature_name][indices]
			for feature_name in cached_features
		}
	flattened_features = {}

	for feature_name in features:
		flattened_features[feature_name] = _flatten_feature_by_dimension(features[feature_name])

	return scalers, pca, flattened_features

def test_transformations(T_U, T_G, T_U_tilde, T_G_tilde, model, model_type, dataset):
	if (model == None):
		train_embeddings = extract_and_infer_qfm_embeddings(dataset.X_train, features_path=f"training/gtzan_train_features.npz", transformation_G=lambda G: T_G_tilde(G @ T_G.T, len(dataset.X_train)), transformation_U=lambda U: T_U_tilde(U @ T_U.T, len(dataset.X_train)))
		model = model_type()
		model.fit(train_embeddings, dataset.y_train)

	test_embeddings = extract_and_infer_qfm_embeddings(dataset.X_test, features_path=f"training/gtzan_test_features.npz", transformation_G=lambda G: T_G_tilde(G @ T_G.T, len(dataset.X_test)), transformation_U=lambda U: T_U_tilde(U @ T_U.T, len(dataset.X_test)))
	predictions = model.predict_proba(test_embeddings)
	
	accuracy = compute_accuracy_genre_classification(dataset.y_test, predictions, model)
	return accuracy


def compute_accuracy_after_optimization_G(dataset, model, feature_sets_kept, Mi, Ki, T_U_tilde=None, model_type=lambda: LogisticRegression(max_iter=50000), n_samples=100):
	scalers, pca, flattened_features = load_model_and_features(n_samples)
	T_U = T(feature_sets_kept, Ki)
	if (T_U_tilde is None):
		T_U_tilde = lambda U, N: U @ np.transpose(T_U).T

	A_G_tilde, b_G_tilde = optimization_G(flattened_features, feature_sets_kept, Mi, lambda U, N: T_U_tilde(U @ T_U.T, N), scalers, pca)


	T_G = T(feature_sets_kept, Mi)

	T_G_tilde = lambda U, N: U @ A_G_tilde.T + np.ones((N, 1)) @ b_G_tilde.T

	return test_transformations(T_U, T_G, T_U_tilde, T_G_tilde, model, model_type, dataset)

def compute_accuracy_after_optimization_U(dataset, model, feature_sets_kept, Mi, Ki, T_G_tilde=None, model_type=lambda: LogisticRegression(max_iter=50000), n_samples=100):
	scalers, pca, flattened_features = load_model_and_features(n_samples)
	T_G = T(feature_sets_kept, Mi)
	if (T_G_tilde is None):
		T_G_tilde = lambda G, N: G @ np.transpose(T_G).T

	A_U_tilde, b_U_tilde = optimization_U(flattened_features, feature_sets_kept, Ki, lambda G, N: T_G_tilde(G @ T_G.T, N), scalers, pca)


	T_U = T(feature_sets_kept, Ki)

	T_U_tilde = lambda U, N: U @ A_U_tilde.T + np.ones((N, 1)) @ b_U_tilde.T

	return test_transformations(T_U, T_G, T_U_tilde, T_G_tilde, model, model_type, dataset)

def compute_accuracy_after_optimization_G_and_U(dataset, model, feature_sets_kept, Mi, Ki, model_type=lambda: LogisticRegression(max_iter=50000), n_samples=100):
	T_G = T(feature_sets_kept, Mi)
	T_U = T(feature_sets_kept, Ki)

	scalers, pcas, flattened_features = load_model_and_features(n_samples)
	T_G_tilde, T_U_tilde = optimization_G_and_U(flattened_features, feature_sets_kept, Mi, Ki, scalers, pcas)

	return test_transformations(T_U, T_G, T_U_tilde, T_G_tilde, model, model_type, dataset)

if (__name__ == "__main__"):
	model = joblib.load(f"training/genre_classification_model_kmeans_fma_small.joblib")
	dataset = extract_gtzan()
	if dataset.X_valid is not None and dataset.y_valid is not None:
		dataset.X_train.extend(dataset.X_valid)
		dataset.y_train.extend(dataset.y_valid)
	Mi, Ki = get_Mi_Ki()
	accuracies_U, importances_U = compute_importances_subsets(dataset, None, Mi, Ki, compute_accuracy_after_optimization_G_and_U, max_subset_size=1)
	print("accuracies", accuracies_U)
	print("importances", importances_U)

from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import cast

import joblib
import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
from scipy.sparse.linalg import LinearOperator, lsmr
import scipy.special
from sklearn.linear_model import LogisticRegression
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
				row_start:row_start + size_i
			]

			gradients.append(block_grad_b.ravel())

		row_start += size_i

	return np.concatenate(gradients)


def build_transformation(A, b=None):
	if b is None:
		return lambda X, _: X @ A.T
	else:
		return lambda X, N: X @ A.T + np.ones((N, 1)) @ b.T

def compute_delta(X, X_tilde):
	deltaX = {
		feature_name: X_tilde[feature_name] - X[feature_name]
		for feature_name in X
	}
	return deltaX

def get_N_params(feature_sets_kept, feature_sets_length):
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
	result = lsmr(L, Y_adjusted, x0=x0)

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

def li_adjoint(A, Kprime, start, feature_name):
	if (feature_name == "qf_total"):
		return A
	A_Kprime = np.zeros((A.shape[0], Kprime))
	A_Kprime[:, start:start + A.shape[1]] = A
	return A_Kprime

def diag_2d_to_3d(A):
	return A[:, :, None] * np.eye(A.shape[1])[None, :, :]

def sigmoid(k, x):
	return scipy.special.expit(k * x)

def optimization_U(features, feature_sets_kept, Ki, transformation_G, scalers, pca, old_T_U_tilde=None, k_sigmoid=5, use_pi_tilde_approx=True, force_positive_parameters=False, l2_regularization=1e-2):
	N_params_A, N_params_b = get_N_params(feature_sets_kept, Ki)
	if (N_params_A + N_params_b == 0):
		return build_T_tilde([], feature_sets_kept, Ki)
	
	started_at = perf_counter()

	U = cast(dict, infer_qfm_embeddings(features, to_return="unigram", feature_sets_kept=feature_sets_kept))	
	W = compute_wmf(U)
	
	T_U = T(feature_sets_kept, Ki)

	N = list(W.values())[0].shape[0]
	ones = np.ones((N, 1))

	all_feature_sets = {feature_name: True for feature_name in FEATURE_NAMES}

	G = infer_qfm_embeddings(features, to_return="G1", feature_sets_kept=feature_sets_kept)
	G_tilde = transform(G, lambda X: transformation_G(X, N), all_feature_sets, G1=True) if transformation_G is not None else G

	U_kept = transform(U, lambda X: build_transformation(T_U)(X, N), feature_sets_kept, sparse_format=True)

	indices_W = {}
	start = 0
	for feature_name in FEATURE_NAMES:
		end = start + Mi[feature_name]
		start = end + W[feature_name].shape[1]
		indices_W[feature_name] = ((end, start))

	end = start + W["qf_total"].shape[1]
	indices_W["qf_total"] = ((start, end))

	IQR_inverse = np.linalg.inv(np.diag(np.concatenate([scalers[feature_name].scale_ for feature_name in FEATURE_NAMES] + [scalers["qf_total"].scale_])))

	wmf_models = {
		feature_name: joblib.load(f"training/fma_small/wmf_{feature_name}.joblib")
		for feature_name in FEATURE_NAMES
	}
	wmf_models["qf_total"] = joblib.load(f"training/fma_small/qf_total.joblib")
	V = {feature_name: wmf_models[feature_name].item_factors for feature_name in wmf_models}

	R = pca.components_.T

	Rprime = IQR_inverse @ R

	start_Ki = {"qf_total": 0}
	last_start_Ki = 0
	for feature_name in FEATURE_NAMES:
		start_Ki[feature_name] = last_start_Ki
		last_start_Ki += Ki[feature_name]
	Kprime = last_start_Ki

	flattened_U = cast(scipy.sparse.csr_matrix, flatten_dict(U, sparse=True))
	flattened_U_transformed = flattened_U @ T_U.T
	E = postprocessing(scalers, pca, G, W)
	def to_optimize(theta):
		started_at2 = perf_counter()
		if (force_positive_parameters):
			T_U_tilde_parameters = scipy.special.softplus(theta)
		else:
			T_U_tilde_parameters = theta
		A_U_tilde, b_U_tilde = build_T_tilde(T_U_tilde_parameters, feature_sets_kept, Ki)
		U_tilde = transform(U_kept, lambda X: build_transformation(A_U_tilde, b_U_tilde)(X, N), all_feature_sets, sparse_format=True, dimensions={key: U[key].shape[1] for key in U})
		flattened_U_tilde = cast(scipy.sparse.csr_matrix, flatten_dict(U_tilde, sparse=True))
		W_tilde = compute_wmf(U_tilde)

		E_tilde = postprocessing(scalers, pca, G_tilde, W_tilde)
		deltaE = E_tilde - E

		deltaE_RprimeT = deltaE @ Rprime.T
		gradient_A = np.zeros((Kprime, T_U.shape[0]))
		gradient_b = np.zeros(Kprime)
		for feature_name in indices_W:
			start, end = indices_W[feature_name]
			Vi = V[feature_name]
			Di = Vi.shape[1]
			alpha = wmf_models[feature_name].alpha
			lambda_ = wmf_models[feature_name]. regularization
			Wi_tilde = W_tilde[feature_name]
			if (feature_name == "qf_total"):
				Ui_tilde = flattened_U_tilde.toarray()
			else:
				Ui_tilde = U_tilde[feature_name].toarray()
			Oi_tilde = 1 + alpha * Ui_tilde
			pi_tilde = (Ui_tilde > 0).astype(int)

			X = deltaE_RprimeT[:, start:end]
			Ci_tilde = Oi_tilde[:, :, None]
			to_inverse = np.matmul(
				Vi.T[None, :, :],
				Ci_tilde * Vi[None, :, :],
			) + lambda_ * np.eye(Di)[None, :, :]
			solved_X = np.linalg.solve(to_inverse, X[:, :, None])[:, :, 0]

			cond = np.linalg.cond(to_inverse)
			if (np.max(cond) > 1e6):
				print(f"ill-conditioned matrix for feature {feature_name} with condition number {np.max(cond):.2e}")

			solved_X_ViT = solved_X @ Vi.T
			diag_Xprime = solved_X_ViT * (pi_tilde - Wi_tilde @ Vi.T)
			# X_doubleprime = Oi_tilde * solved_X_ViT
			li_star_diag_star_Xprime = li_adjoint(
				alpha * diag_Xprime,
				Kprime,
				start_Ki[feature_name],
				feature_name,
			)
			q_total = li_star_diag_star_Xprime
			if (use_pi_tilde_approx):
				pi_tilde_approx = sigmoid(k_sigmoid, Ui_tilde)
				X_doubleprime = Oi_tilde * solved_X_ViT
				sigma_derivative = k_sigmoid * pi_tilde_approx * (1 - pi_tilde_approx)
				sigma_derivative_X_doubleprime = sigma_derivative * X_doubleprime
				li_star_sigma_derivative_X_doubleprime = li_adjoint(
					sigma_derivative_X_doubleprime,
					Kprime,
					start_Ki[feature_name],
					feature_name,
				)
				q_total += li_star_sigma_derivative_X_doubleprime


			gradient_A += q_total.T @ flattened_U_transformed
			gradient_b += q_total.sum(axis=0)
		gradient = build_T_tilde_adjoint(gradient_A, gradient_b, feature_sets_kept, Ki)

		L = np.linalg.norm(deltaE)**2  / (N*E.shape[1]) + l2_regularization * np.linalg.norm(T_U_tilde_parameters)**2 / len(T_U_tilde_parameters)

		gradient = gradient / (N*E.shape[1]) +  l2_regularization * T_U_tilde_parameters / len(T_U_tilde_parameters)

		if (force_positive_parameters):
			gradient_theta = gradient * scipy.special.expit(theta)
		else:
			gradient_theta = gradient
		
		print(f"loss: {L:.6f}, time elapsed: {perf_counter() - started_at2:.2f} seconds, min: {np.min(T_U_tilde_parameters):.6f}, max: {np.max(T_U_tilde_parameters):.6f}, gradient norm: {np.linalg.norm(gradient_theta):.6f}, gradient max: {np.max(gradient_theta):.6f}")
		return L, 2*gradient_theta

	if (old_T_U_tilde is not None):
		old_A_U_tilde, old_b_U_tilde = old_T_U_tilde
		old_theta = build_T_tilde_adjoint(old_A_U_tilde, old_b_U_tilde, feature_sets_kept, Ki)
		if (force_positive_parameters):
			x0 = np.log(np.expm1(1e-4 + old_theta))  # Inverse of softplus
		else:
			x0 = old_theta
	else:
		x0 = np.zeros(N_params_A + N_params_b)

	result = scipy.optimize.minimize(to_optimize, x0, jac=True, method="L-BFGS-B")
	x = result.x

	if (force_positive_parameters):
		theta_opt = scipy.special.softplus(x)
	else:
		theta_opt = x
	T_tilde = build_T_tilde(theta_opt, feature_sets_kept, Ki)
	print(f"Optimization for U completed in {perf_counter() - started_at:.2f} seconds.")
	return T_tilde

def optimization_G_and_U(features, feature_sets_kept, Mi, Ki, scalers, pca, n_repeats=5):
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
		A_U_tilde, b_U_tilde = optimization_U(features, feature_sets_kept, Ki, lambda G, N: T_G_tilde(G @ T_G.T, N), scalers, pca, old_T_U_tilde=(A_U_tilde, b_U_tilde))
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
		rng = np.random.default_rng(42)
		indices = rng.choice(len(list(cached_features.values())[0]), size=n_samples, replace=False)
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

def compute_accuracy_after_optimization_G_and_U(dataset, model, feature_sets_kept, Mi, Ki, model_type=lambda: LogisticRegression(max_iter=50000), n_samples=10, n_repeats=3):
	T_G = T(feature_sets_kept, Mi)
	T_U = T(feature_sets_kept, Ki)

	scalers, pcas, flattened_features = load_model_and_features(n_samples)
	T_G_tilde, T_U_tilde = optimization_G_and_U(flattened_features, feature_sets_kept, Mi, Ki, scalers, pcas, n_repeats=n_repeats)

	return test_transformations(T_U, T_G, T_U_tilde, T_G_tilde, model, model_type, dataset)

if (__name__ == "__main__"):
	model = joblib.load(f"training/genre_classification_model_kmeans_fma_small.joblib")
	dataset = extract_gtzan()
	if dataset.X_valid is not None and dataset.y_valid is not None:
		dataset.X_train.extend(dataset.X_valid)
		dataset.y_train.extend(dataset.y_valid)
	Mi, Ki = get_Mi_Ki()
	accuracies_U, importances_U = compute_importances_subsets(dataset, None, Mi, Ki, compute_accuracy_after_optimization_U)
	print("accuracies", accuracies_U)
	print("importances", importances_U)

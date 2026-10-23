import inspect
from time import perf_counter

import joblib
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm

from _accumulated_local_effects import accumulated_local_effects
from downstream_tasks import compute_accuracy_genre_classification, extract_gtzan
from features import FEATURE_NAMES, _flatten_feature_by_dimension, _restore_feature_shape
from qfm_inference import extract_features, infer_qfm_embeddings

class ModelImportance:
	def __init__(self, dataset, score_func, model, dataset_name: str, qfm_path = "training/fma_small", clustering_method = "kmeans", greater_is_better: bool = True, response_method: str = "predict_proba"):
		self.model = model
		self.dataset = dataset
		self.qfm_path = qfm_path
		self.clustering_method = clustering_method
		self.response_method = response_method

		self.features_train = extract_features(dataset.X_train, sample_rate=22050, hop_size=512, features_path=f"training/{dataset_name}_train_features.npz")
		self.features_train_df = self.convert_features_to_dataframe(self.features_train)

		self.features_test = extract_features(dataset.X_test, sample_rate=22050, hop_size=512, features_path=f"training/{dataset_name}_test_features.npz")
		self.features_test_df = self.convert_features_to_dataframe(self.features_test)


		self.pipeline = Pipeline([
			("conversion", FunctionTransformer(self.convert_features_to_dict, validate=False)),
			("embedding", FunctionTransformer(lambda X: infer_qfm_embeddings(X, model_path=self.qfm_path, clustering_method=self.clustering_method), validate=False)),
			("model", self.model)
		])

		self.score_func = make_scorer(lambda y_pred, y_true: score_func(
			y_pred,
			y_true,
			**({"model": self.model}
			if "model" in inspect.signature(score_func).parameters
			else {})
		), greater_is_better=greater_is_better, response_method=response_method)

		self.original_shapes = {key: arr.shape for key, arr in self.features_test.items()}

	def convert_features_to_dataframe(self, features: dict[str, np.ndarray]) -> pd.DataFrame:
		df = pd.concat(
			{
				key: pd.DataFrame(_flatten_feature_by_dimension(arr), columns=[f"{key}_{i}" for i in range(arr.shape[1])])
				for key, arr in features.items()
			},
			axis=1
		)

		df.columns = df.columns.map(lambda t: t[1])
		return df

	def convert_features_to_dict(self, df: pd.DataFrame):
		a = {}
		for _key, cols in (
						pd.Series(df.columns, index=df.columns)
						.groupby(lambda col: col.rsplit("_", 1)[0], sort=False)
						.agg(list)
						.items()
					):
			key = str(_key).split("_")[0]
			a[key] = df[cols].to_numpy()

		return a

	def plot_importances(self, importances, n_features_to_show = 10, errors=None) -> tuple[Figure, Axes]:
		fig, ax = plt.subplots()

		sorted_idx = np.argsort(list(importances.values()))
		first_and_last_idx = np.concatenate([sorted_idx[:n_features_to_show // 2], sorted_idx[-n_features_to_show // 2:]])

		xerr = errors[first_and_last_idx] if errors is not None else None
		ax.barh(np.array(list(importances.keys()))[first_and_last_idx], np.array(list(importances.values()))[first_and_last_idx], xerr=xerr, align='center')
		ax.yaxis.set_inverted(True)
		fig.show()
		return fig, ax

	def permutation_importance(self, n_features_to_show: int = 10, n_repeats=50) -> tuple[Figure, Axes]:
		started_at = perf_counter()
		result = permutation_importance(self.pipeline, self.features_test_df, self.dataset.y_test, scoring=self.score_func, n_repeats=n_repeats, random_state=42)
		sorted_idx = result["importances_mean"].argsort()
		
		self.permutation_importance_result = result
		fig, ax = self.plot_importances({
			self.features_test_df.columns[i]: result["importances_mean"][i] for i in range(len(result["importances_mean"]))
		}, n_features_to_show)

		for i in sorted_idx[::-1]:
			if result["importances_mean"][i] - 2 * result["importances_std"][i] > 0:
				print(f"feature {self.features_test_df.columns[i]}: "
					f"{result['importances_mean'][i]:.3f}"
					f" +/- {result['importances_std'][i]:.3f}")
		print(f"Permutation importance computed in {perf_counter() - started_at:.2f} seconds")

		return fig, ax

	def flatten_dict(self, d: dict[str, np.ndarray]):
		return np.concatenate([value for value in d.values()], axis=1)


	
	def convert_3d_to_dataframe(self, x: np.ndarray) -> pd.DataFrame:
		return pd.DataFrame(
			[
				[np.array(cell) for cell in row]
				for row in x.reshape(x.shape[0], x.shape[1], -1)
			],
			columns=self.features_test_df.columns
		)

	def shap_importance(self, class_idx_to_plot: int = 0, n_samples = 100):
		started_at = perf_counter()
		def predict_fn(X):
			x = np.stack(X.ravel()).reshape(*X.shape, -1)
			reshaped = np.reshape(x, (x.shape[0], len(self.features_test_df.columns), -1))
			df = pd.DataFrame(
				_flatten_feature_by_dimension(reshaped),
				columns=self.features_test_df.columns
			)

			predictions = self.pipeline.predict_proba(df)
			return predictions

		class TimeSeriesMasker:
			def __init__(self, background_df):
				self.background = background_df.to_numpy(dtype=object)
				self.shape = self.background.shape
				self.clustering = None
	
			def __call__(self, mask, x):
				mask = np.asarray(mask, dtype=bool)
				x = np.asarray(x, dtype=object)
	
				n_background, n_features = self.background.shape
	
				# Start with the original input repeated for every background row.
				masked = np.empty((n_background, n_features), dtype=object)
	
				for j in range(n_features):
					masked[:, j] = [x[j]] * n_background
	
				# False means masked: replace those features using the
				# corresponding columns from each background row.
				for j in np.flatnonzero(~mask):
					masked[:, j] = self.background[:, j]
	
				return masked
		background_all = self.flatten_dict(self.features_train)
		flattened_test = self.flatten_dict(self.features_test)

		background_df = self.convert_3d_to_dataframe(background_all)
		test_df = self.convert_3d_to_dataframe(flattened_test)

		rng = np.random.default_rng(42)
		idx = rng.choice(
			len(background_df),
			size=min(100, len(background_df)),
			replace=False
		)

		idx_test = rng.choice(
			len(test_df),
			size=min(n_samples, len(test_df)),
			replace=False
		)

		background = background_df.iloc[idx].copy()
		test = test_df.iloc[idx_test].copy()

		masker = TimeSeriesMasker(background)

		explainer = shap.Explainer(
			predict_fn,
			masker,
			feature_names=self.features_test_df.columns.to_list()
		)

		shap_values = explainer(test, max_evals=2*len(test_df.columns) + 1) # type: ignore
		shap_values.feature_names = explainer.feature_names # pyright: ignore[reportAttributeAccessIssue]
		class_shap = shap_values[..., class_idx_to_plot] # type: ignore

		shap.plots.bar(class_shap)
		print(f"SHAP importance computed in {perf_counter() - started_at:.2f} seconds")	
		return explainer, shap_values

	def ale_importance(self, n_features_to_show: int = 10):
		started_at = perf_counter()
		results = {}

		flattened_test = self.flatten_dict(self.features_test)

		def convert(X):
			X_3d = _restore_feature_shape(X, flattened_test.shape)
			X_dict = {}
			i = 0
			for feature_name in FEATURE_NAMES:
				original_length = self.features_test[feature_name].shape[1]
				X_dict[feature_name] = _flatten_feature_by_dimension(X_3d[:, i:i+original_length, :])
				i += original_length
			return X_dict
		ale_pipeline = Pipeline([
			("conversion", FunctionTransformer(lambda X: convert(X), validate=False)),
			("pipeline", self.pipeline[1:])
		])

		for i in tqdm(range(len(self.features_test_df.columns))):
			results[self.features_test_df.columns[i]] = accumulated_local_effects(ale_pipeline, flattened_test, i, response_method=self.response_method)
		main_importances = {feature_name: np.var(results[feature_name]["effects"]) for feature_name in results}
		fig, ax = self.plot_importances(main_importances, n_features_to_show=n_features_to_show)
		print(f"ALE importance computed in {perf_counter() - started_at:.2f} seconds")
		return results, main_importances, fig, ax
		
def permutation_importance_genre_classification(model_name: str = "kmeans_fma_small"):
	model = joblib.load(f"training/genre_classification_model_{model_name}.joblib")
	dataset = extract_gtzan(y_only=True)
	importance = ModelImportance(dataset, compute_accuracy_genre_classification, model, qfm_path=f"training/{model_name.split('_', 1)[1]}", clustering_method=model_name.split('_', 1)[0], greater_is_better=True, response_method="predict_proba", dataset_name="gtzan")
	importance.permutation_importance()
	r = importance.permutation_importance_result	
	

	np.savez(f"training/permutation_importance_genre_classification_{model_name}.npz", **r, columns=importance.features_test_df.columns) # pyright: ignore[reportArgumentType]

def shap_importance_genre_classification(model_name: str = "kmeans_fma_small"):
	model = joblib.load(f"training/genre_classification_model_{model_name}.joblib")
	dataset = extract_gtzan(y_only=True)
	importance = ModelImportance(dataset, compute_accuracy_genre_classification, model, qfm_path=f"training/{model_name.split('_', 1)[1]}", clustering_method=model_name.split('_', 1)[0], greater_is_better=True, response_method="predict_proba", dataset_name="gtzan")
	explainer, shap_values = importance.shap_importance(n_samples=2)
	
	np.savez(f"training/shap_importance_genre_classification_{model_name}.npz", values=shap_values.values, base_values=shap_values.base_values, data=shap_values.data, feature_names=shap_values.feature_names)  # type: ignore

def ale_importance_genre_classification(model_name: str = "kmeans_fma_small"):
	model = joblib.load(f"training/genre_classification_model_{model_name}.joblib")
	dataset = extract_gtzan(y_only=True)
	importance = ModelImportance(dataset, compute_accuracy_genre_classification, model, qfm_path=f"training/{model_name.split('_', 1)[1]}", clustering_method=model_name.split('_', 1)[0], greater_is_better=True, response_method="predict_proba", dataset_name="gtzan")

	results, main_importances, fig, ax = importance.ale_importance()
	np.savez(f"training/ale_importance_genre_classification_{model_name}.npz", **results, main_importances=np.array(list(main_importances.values())), columns=importance.features_test_df.columns)

# permutation_importance_genre_classification()
# shap_importance_genre_classification()
ale_importance_genre_classification()
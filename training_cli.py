
import argparse
import logging

from features import extract_dataset_features
from qfm_training import build_unigram_dataset_features, compute_G1, gmm_dataset_features, kmeans_dataset_features, postprocessing_embeddings, sample_dataset_feature_windows, scale_dataset_features, train_factorization_stage, train_qfm_model, whitening_pca_dataset_features


parser = argparse.ArgumentParser(description="Run feature-processing stages.")
subparsers = parser.add_subparsers(dest="operation", required=True)

def add_dataset_and_overwrite(command_parser: argparse.ArgumentParser) -> None:
	command_parser.add_argument(
		"--dataset",
		default="fma_small",
		help="Dataset name under datasets/ and training/.",
	)
	command_parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Replace existing output files.",
	)

def add_parser(name: str, help_text: str) -> argparse.ArgumentParser:
	command_parser = subparsers.add_parser(name, help=help_text)
	add_dataset_and_overwrite(command_parser)
	return command_parser

extract_parser = add_parser("extract", "Extract features from the dataset.")
extract_parser.add_argument("--samples", type=int, default=60)
extract_parser.set_defaults(
	handler=lambda args: extract_dataset_features(
		dataset_name=args.dataset,
		n_samples=args.samples,
		overwrite=args.overwrite,
	)
)

sample_parser = add_parser("sample", "Sample feature windows.")
sample_parser.add_argument("--samples", type=int, default=60)
sample_parser.set_defaults(
	handler=lambda args: sample_dataset_feature_windows(
		dataset_name=args.dataset,
		n_samples=args.samples,
		overwrite=args.overwrite,
	)
)

scale_parser = add_parser("scale", "Scale extracted features.")
scale_parser.set_defaults(
	handler=lambda args: scale_dataset_features(
		dataset_name=args.dataset,
		overwrite=args.overwrite,
	)
)

pca_parser = add_parser("pca_whiten", "Apply whitening PCA.")
pca_parser.set_defaults(
	handler=lambda args: whitening_pca_dataset_features(
		dataset_name=args.dataset,
		overwrite=args.overwrite,
	)
)

for clustering_name, clustering_function in (
	("kmeans", kmeans_dataset_features),
	("gmm", gmm_dataset_features),
):
	clustering_parser = add_parser(
		clustering_name,
		f"Run {clustering_name.upper()} clustering.",
	)
	clustering_parser.add_argument("--k", type=int, default=512)
	clustering_parser.set_defaults(
		handler=lambda args, function=clustering_function: function(
			dataset_name=args.dataset,
			n_clusters=args.k,
			overwrite=args.overwrite,
		)
	)

for unigram_name, clustering_name in (
	("kmeans_unigram", "kmeans"),
	("gmm_unigram", "gmm"),
):
	unigram_parser = add_parser(
		unigram_name,
		f"Build {clustering_name.upper()} unigram matrices.",
	)
	unigram_parser.add_argument("--min-presence", type=float, default=0.001)
	unigram_parser.add_argument("--windows-per-segment", type=int, default=60)
	unigram_parser.set_defaults(
		handler=lambda args, clustering=clustering_name:
			build_unigram_dataset_features(
				dataset_name=args.dataset,
				clustering=clustering,
				min_presence=args.min_presence,
				windows_per_segment=args.windows_per_segment,
				overwrite=args.overwrite,
			)
	)

g1_parser = add_parser("G1", "Compute G1 statistics.")
g1_parser.set_defaults(
	handler=lambda args: compute_G1(
		dataset_name=args.dataset,
		overwrite=args.overwrite,
	)
)

factorization_parser = add_parser(
	"factorization",
	"Build unigrams and train WMF models.",
)
factorization_parser.add_argument("--clustering", choices=("kmeans", "gmm"), default="kmeans")
factorization_parser.add_argument("--alpha", type=float, default=5)
factorization_parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
factorization_parser.add_argument("--factors", type=int, default=128)
factorization_parser.add_argument("--windows-per-segment", type=int, default=60)
factorization_parser.add_argument("--min-presence", type=float, default=0.001)
factorization_parser.set_defaults(
	handler=lambda args: train_factorization_stage(
		dataset_name=args.dataset,
		clustering=args.clustering,
		alpha=args.alpha,
		lambda_=args.lambda_,
		factors_default=args.factors,
		windows_per_segment=args.windows_per_segment,
		min_presence=args.min_presence,
	)
)

postprocessing_parser = add_parser(
	"postprocessing",
	"Create final postprocessed embeddings.",
)
postprocessing_parser.set_defaults(
	handler=lambda args: postprocessing_embeddings(
		dataset_name=args.dataset,
		overwrite=args.overwrite,
	)
)

train_parser = subparsers.add_parser(
	"train",
	help="Run the complete QFM training pipeline.",
)
train_parser.add_argument(
	"--dataset",
	default="fma_small",
	help="Dataset name under datasets/ and training.",
)
train_parser.add_argument("--alpha", type=float, default=5)
train_parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
train_parser.add_argument("--factors", type=int, default=128)
train_parser.add_argument("--windows-per-segment", type=int, default=60)
train_parser.add_argument("--min-presence", type=float, default=0.001)
train_parser.add_argument("--clustering", choices=("kmeans", "gmm"), default="kmeans")
train_parser.add_argument("--sample-rate", type=int, default=22050)
train_parser.add_argument("--hop-size", type=int, default=512)
train_parser.add_argument("--max-workers", type=int, default=8)
train_parser.add_argument("--n-clusters", type=int, default=512)
train_parser.add_argument("--extract-features", action="store_true", help="Extract and sample features.", default=False)
train_parser.set_defaults(
	handler=lambda args: train_qfm_model(
		dataset_name=args.dataset,
		alpha=args.alpha,
		lambda_=args.lambda_,
		factors=args.factors,
		windows_per_segment=args.windows_per_segment,
		min_presence=args.min_presence,
		clustering=args.clustering,
		sample_rate=args.sample_rate,
		hop_size=args.hop_size,
		n_clusters=args.n_clusters,
		extract_features=args.extract_features
	)
)

args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
print(args.handler(args))
	
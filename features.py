import logging
import os
from pathlib import Path
from typing import cast
import librosa
import numpy as np
import hashlib

from threadpoolctl import threadpool_limits
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from helpers import ResumableJob, write_feature_archive

FEATURE_NAMES = ("mfcc", "cqt", "rhythmic", "spectral", "patches")
SEGMENT_DURATION_SECONDS = 9 # W in paper
SEGMENTS_PER_FILE = 30 // 9  # each file is 30 seconds long

class FeatureExtractor:
	def __init__(
		self,
		audio,
		sr=22050,
		hopsize=512,
		patch_seed=None,
		n_patches=60,
	):
		self.audio = audio
		self.sr = sr
		self.hopsize = hopsize
		self.patch_seed = patch_seed
		self.n_patches = n_patches

		self.spectrogram = librosa.stft(y=self.audio, hop_length=self.hopsize)
		self.magnitude_spectrogram = np.abs(self.spectrogram)
		self.melspectrogram = librosa.feature.melspectrogram(S=self.spectrogram, sr=self.sr, hop_length=self.hopsize, n_mels=96)
		self.log_melspectrogram = librosa.power_to_db(np.abs(self.melspectrogram), ref=np.max)

		self.feature_sets = {
			feature_name: False for feature_name in FEATURE_NAMES
		}

		(self.addMFCC()
		.addCQT()
		.addRhythmic()
		.addSpectral()
		.addPatches())


	def addMFCC(self):
		# 96 MFCC coefficients excluding the first
		mfcc = librosa.feature.mfcc(S=self.log_melspectrogram, sr=self.sr, hop_length=self.hopsize, n_mfcc=96)

		self.mfcc = mfcc[1:]
		self.feature_sets["mfcc"] = True

		return self

	def addCQT(self):
		# 84-bin log Constant-Q power spectrum. convert to log scale
		cqt = librosa.cqt(y=self.audio, sr=self.sr, hop_length=self.hopsize, n_bins=84)
		cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

		self.cqt = cqt
		self.feature_sets["cqt"] = True

		return self

	def addRhythmic(self):
		# Multi-band Onset Strength, Each 4 bands are divided by {16, 32, 64}-th bins among 96 mel bins
		bin1 = 16
		bin2 = 32
		bin3 = 64

		band1 = self.melspectrogram[:bin1, :]
		band2 = self.melspectrogram[bin1:bin2, :]
		band3 = self.melspectrogram[bin2:bin3, :]
		band4 = self.melspectrogram[bin3:, :]

		onset1 = librosa.onset.onset_strength(S=band1, sr=self.sr, hop_length=self.hopsize)
		onset2 = librosa.onset.onset_strength(S=band2, sr=self.sr, hop_length=self.hopsize)
		onset3 = librosa.onset.onset_strength(S=band3, sr=self.sr, hop_length=self.hopsize)
		onset4 = librosa.onset.onset_strength(S=band4, sr=self.sr, hop_length=self.hopsize)

		# normalized BPM estimation
		tempogram = librosa.feature.tempogram(y=self.audio, sr=self.sr, hop_length=self.hopsize)
		bpm = librosa.feature.tempo(tg=tempogram, aggregate=None)

		# Tempogram Ratio 
		tempogram_ratio = librosa.feature.tempogram_ratio(tg=tempogram, sr=self.sr, hop_length=self.hopsize)
		tempogram_ratio = np.abs(tempogram_ratio)


		self.rhythmic = np.real(
			np.vstack([onset1, onset2, onset3, onset4, tempogram_ratio, bpm])
		).astype(np.float32)
		self.feature_sets["rhythmic"] = True

		return self

	def addSpectral(self):
		# Spectral Centroid
		centroid = librosa.feature.spectral_centroid(S=self.magnitude_spectrogram, sr=self.sr, hop_length=self.hopsize)

		# Spectral Bandwidth
		bandwidth = librosa.feature.spectral_bandwidth(S=self.magnitude_spectrogram, sr=self.sr, hop_length=self.hopsize)

		# Spectral Contrast
		contrast = librosa.feature.spectral_contrast(S=self.magnitude_spectrogram, sr=self.sr, hop_length=self.hopsize)

		# Spectral Flatness
		flatness = librosa.feature.spectral_flatness(S=self.magnitude_spectrogram, hop_length=self.hopsize)

		# Spectral Rolloff
		rolloff = librosa.feature.spectral_rolloff(S=self.magnitude_spectrogram, sr=self.sr, hop_length=self.hopsize)

		# Zero-crossing Rate
		zero_crossing_rate = librosa.feature.zero_crossing_rate(y=self.audio, hop_length=self.hopsize)

		# Tonnetz
		tonnetz = librosa.feature.tonnetz(y=self.audio, sr=self.sr, hop_length=self.hopsize)
		self.feature_sets["spectral"] = True

		self.spectral = np.vstack([centroid, bandwidth, contrast, flatness, rolloff, zero_crossing_rate, tonnetz]).astype(np.float32)
		return self

	def addPatches(self):
		# (9 × 9) patches from log mel spectrogram
		n_mels, n_frames = self.log_melspectrogram.shape
		p = 9

		rng = np.random.default_rng(self.patch_seed)
		n_mel_positions = n_mels - p + 1
		n_frame_positions = n_frames - p + 1
		n_positions = n_mel_positions * n_frame_positions
		
		selected = rng.choice(n_positions, size=self.n_patches, replace=False)
		mel_starts, frame_starts = np.divmod(selected, n_frame_positions)

		offsets = np.arange(p)
		patches = self.log_melspectrogram[
			mel_starts[:, None, None] + offsets[None, :, None],
			frame_starts[:, None, None] + offsets[None, None, :]
		]
		self.patches = patches.reshape(self.n_patches, p * p).T
		self.feature_sets["patches"] = True

		return self


def _extract_file_features_impl(
	arguments: tuple[Path, int, int, int, int],
) -> tuple[Path, dict[str, np.ndarray]]:
	"""Extract all three segments from one file in a worker process."""
	audio_path, sample_rate, hop_size, patch_seed, n_samples = arguments
	audio, loaded_sample_rate = librosa.load(
		audio_path,
		sr=sample_rate,
		mono=True,
		duration=SEGMENT_DURATION_SECONDS * SEGMENTS_PER_FILE,
	)
	segment_samples = SEGMENT_DURATION_SECONDS * sample_rate
	features: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_NAMES}

	for segment_number in range(SEGMENTS_PER_FILE):
		start = segment_number * segment_samples
		segment = audio[start : start + segment_samples]
		extractor = FeatureExtractor(
			segment,
			sr=int(loaded_sample_rate),
			hopsize=hop_size,
			patch_seed=patch_seed + segment_number,
			n_patches=n_samples,
		)
		for name in FEATURE_NAMES:
			features[name].append(np.asarray(getattr(extractor, name)))

	return audio_path, {
		name: np.stack(values, axis=0) for name, values in features.items()
	}


def _extract_file_features(
	arguments: tuple[Path, int, int, int, int],
) -> tuple[Path, dict[str, np.ndarray]]:
	"""Extract one file while limiting BLAS threads in this worker only."""
	with threadpool_limits(limits=1):
		return _extract_file_features_impl(arguments)

def extract_dataset_features(
	dataset_name: str,
	*,
	sample_rate: int = 22050,
	hop_size: int = 512,
	n_samples: int = 60,
	overwrite: bool = False,
) -> Path:
	"""Extract three 9-second segments per track into one compressed archive.

	Files are processed in separate worker processes. At most a small bounded
	number of files is in flight, and temporary feature arrays are disk-backed,
	so the complete dataset is never held in memory. The final archive contains
	only the five requested arrays, with segments stored along their first axis.
	"""
	dataset_name = dataset_name if dataset_name is None else dataset_name
	project_dir = Path(__file__).resolve().parent
	input_dir = project_dir / "datasets" / dataset_name
	output_dir = project_dir / "training" / dataset_name
	output_path = output_dir / "features.npz"
	if output_path.exists() and not overwrite:
		raise FileExistsError(
			f"{output_path} already exists; pass overwrite=True to replace it"
		)
	if not input_dir.is_dir():
		raise FileNotFoundError(f"Dataset directory does not exist: {input_dir}")

	audio_paths = sorted(
		path
		for path in input_dir.rglob("*")
		if path.is_file() and path.suffix.lower() == ".mp3"
	)
	if not audio_paths:
		raise FileNotFoundError(f"No supported audio files found in {input_dir}")
	if n_samples < 1:
		raise ValueError("n_samples must be at least 1")

	output_dir.mkdir(parents=True, exist_ok=True)
	configuration = {
		"dataset_name": dataset_name,
		"sample_rate": sample_rate,
		"hop_size": hop_size,
		"n_samples": n_samples,
		"patch_seed": "blake2b-relative-path-v1",
		"paths": [str(path.relative_to(input_dir)) for path in audio_paths],
	}
	with ResumableJob(
		output_dir,
		output_path,
		".features_checkpoint",
		configuration,
		overwrite,
	) as job:
		completed = cast(list[int], job.state["completed"])
		completed_indices = {int(index) for index in completed}
		pending = {}
		next_file_number = 0
		failed = 0
		arguments = [
			(
				audio_path,
				sample_rate,
				hop_size,
				int.from_bytes(
					hashlib.blake2b(
						str(audio_path.relative_to(input_dir)).encode(),
						digest_size=8,
					).digest(),
					"little",
				),
				n_samples,
			)
			for audio_path in audio_paths
		]

		with ProcessPoolExecutor() as executor:
			while next_file_number < len(arguments) and len(pending) < cast(int, os.cpu_count()):
				if next_file_number in completed_indices:
					next_file_number += 1
					continue
				pending[
					executor.submit(_extract_file_features, arguments[next_file_number])
				] = next_file_number
				next_file_number += 1

			while pending:
				completed_futures, _ = wait(
					pending, return_when=FIRST_COMPLETED
				)
				for future in completed_futures:
					file_number = pending.pop(future)
					try:
						audio_path, extracted = future.result()
					except Exception:
						failed += 1
						logging.exception(
							"Failed to process %s; skipping file",
							audio_paths[file_number],
						)
						if next_file_number < len(arguments):
							pending[
								executor.submit(
									_extract_file_features,
									arguments[next_file_number],
								)
							] = next_file_number
							next_file_number += 1
						continue

					if not job.memmaps:
						total_segments = len(audio_paths) * SEGMENTS_PER_FILE
						for name in FEATURE_NAMES:
							feature = extracted[name]
							job.open_memmap(
								name,
								dtype=feature.dtype,
								shape=(total_segments, *feature.shape[1:]),
							)

					row_start = file_number * SEGMENTS_PER_FILE
					row_end = row_start + SEGMENTS_PER_FILE
					for name in FEATURE_NAMES:
						job.memmaps[name][row_start:row_end] = extracted[name]

					completed_indices.add(file_number)
					job.state["completed"] = sorted(completed_indices)
					job.save()
					logging.info(
						"Processed %d/%d: %s",
						len(completed_indices),
						len(audio_paths),
						audio_path,
					)

					if next_file_number < len(arguments):
						pending[
							executor.submit(
								_extract_file_features,
								arguments[next_file_number],
							)
						] = next_file_number
						next_file_number += 1

		if not completed_indices:
			raise RuntimeError("No audio files could be processed")

		successful_rows = np.concatenate(
			[
				np.arange(
					index * SEGMENTS_PER_FILE,
					(index + 1) * SEGMENTS_PER_FILE,
				)
				for index in sorted(completed_indices)
			]
		)
		completed = len(completed_indices)

		def write_archive(path: Path) -> None:
			write_feature_archive(
				path,
				job.memmaps,
				FEATURE_NAMES,
				rows=successful_rows,
			)

		job.finalize(write_archive)
		logging.info("Processed %d files (%d skipped)", completed, failed)
	return output_path


def _flatten_feature_by_dimension(data: np.ndarray) -> np.ndarray:
	"""Flatten observation axes while preserving feature dimensions."""
	if data.ndim != 3:
		raise ValueError(f"Expected a 3D feature array, got shape {data.shape}")
	return np.moveaxis(data, 1, -1).reshape(-1, data.shape[1])


def _restore_feature_shape(
	transformed: np.ndarray,
	original_shape: tuple[int, ...],
) -> np.ndarray:
	"""Restore ``(segments, dimensions, windows)`` after processing."""
	segments, dimensions, windows = original_shape
	return np.moveaxis(
		transformed.reshape(segments, windows, dimensions),
		-1,
		1,
	)
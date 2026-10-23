from time import perf_counter

import librosa
import numpy as np

from features import FEATURE_NAMES, FeatureExtractor


def measure_computation_time_feature_sets(audio_input, n_repeats=10, output_path="training/feature_sets_time.npz"):
	times = {
		feature_name: [] for feature_name in FEATURE_NAMES
	}

	for _ in range(n_repeats):
		feature_extractor = FeatureExtractor(audio_input, sets_included={
			feature_name: False for feature_name in FEATURE_NAMES
		})
		for feature_name in FEATURE_NAMES:
			started_at = perf_counter()
			feature_extractor.addMethods[feature_name]()
			times[feature_name].append(perf_counter() - started_at)

	times_mean = {
		feature_name: np.mean(times[feature_name]) for feature_name in FEATURE_NAMES
	}

	np.savez_compressed(output_path, **times_mean) # pyright: ignore[reportArgumentType]

	return times_mean

sr = 22050
audio, sr = librosa.load("audio.mp3", sr=sr)[0:9*sr] 
print(measure_computation_time_feature_sets(audio))

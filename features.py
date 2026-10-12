import librosa
import numpy as np

class FeatureExtractor:
	def __init__(self, audio, sr=22050, hopsize=512):
		self.audio = audio
		self.sr = sr
		self.hopsize = hopsize

		self.spectrogram = librosa.stft(y=self.audio, hop_length=self.hopsize)
		self.magnitude_spectrogram = np.abs(self.spectrogram)
		self.melspectrogram = librosa.feature.melspectrogram(S=self.spectrogram, sr=self.sr, hop_length=self.hopsize, n_mels=96)
		self.log_melspectrogram = librosa.power_to_db(np.abs(self.melspectrogram), ref=np.max)

		self.feature_sets = {
			"mfcc": False,
			"cqt": False,
			"rhythmic": False,
			"spectral": False,
			"patches": False
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


		self.rhythmic = np.vstack([onset1, onset2, onset3, onset4, tempogram_ratio, bpm])
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

		self.spectral = np.vstack([centroid, bandwidth, contrast, flatness, rolloff, zero_crossing_rate, tonnetz])
		return self

	def addPatches(self, n_patches=60):
		# (9 × 9) patches from log mel spectrogram
		# TODO: find n_patches (not provided by paper)
		n_mels, n_frames = self.log_melspectrogram.shape
		p = 9

		rng = np.random.default_rng()
		n_mel_positions = n_mels - p + 1
		n_frame_positions = n_frames - p + 1
		n_positions = n_mel_positions * n_frame_positions
		
		selected = rng.choice(n_positions, size=n_patches, replace=False)
		mel_starts, frame_starts = np.divmod(selected, n_frame_positions)

		offsets = np.arange(p)
		patches = self.log_melspectrogram[
			mel_starts[:, None, None] + offsets[None, :, None],
			frame_starts[:, None, None] + offsets[None, None, :]
		]
		self.patches = patches.reshape(n_patches, p * p).T
		self.feature_sets["patches"] = True

		return self

audio, sr = librosa.load("audio.mp3", sr=22050)
extractor = FeatureExtractor(audio, sr=sr) # type: ignore
print("MFCC shape:", extractor.mfcc.shape)
print("CQT shape:", extractor.cqt.shape)
print("Rhythmic shape:", extractor.rhythmic.shape)
print("Spectral shape:", extractor.spectral.shape)
print("Patches shape:", extractor.patches.shape)

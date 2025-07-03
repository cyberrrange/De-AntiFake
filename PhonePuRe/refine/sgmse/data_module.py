from os.path import join
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import h5py
import json
from sgmse.util.other import snr_scale_factor, pydub_read, align
import math

SEED = 10
np.random.seed(SEED)

def get_window(window_type, window_length):
	if window_type == 'sqrthann':
		return torch.sqrt(torch.hann_window(window_length, periodic=True))
	elif window_type == 'hann':
		return torch.hann_window(window_length, periodic=True)
	else:
		raise NotImplementedError(f"Window type {window_type} not implemented!")

class Specs(Dataset):
	def __init__(
		self, data_dir, subset, dummy, shuffle_spec, num_frames, format,
		normalize_audio=True, spec_transform=None, stft_kwargs=None, spatial_channels=1, 
		return_time=False,
		use_text = False,
		condition = "post_denoiser",
		**ignored_kwargs
	):
		self.data_dir = data_dir
		self.subset = subset
		self.format = format
		self.spatial_channels = spatial_channels
		self.return_time = return_time
		self.condition = condition
		self.noisy_files = {}
		if format in ["wsj0", "vctk"]:
			dic_correspondence_subsets = {"train": "tr", "valid": "cv", "test": "tt"}
			self.clean_files = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/clean/*.wav'))
			self.noisy_files['t0'] = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/noisy/*.wav'))
		elif format in["librispeech-clean"]:
			dic_correspondence_subsets = {"train": "train", "valid": "val", "test": "test"}
			self.clean_files = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/*.wav'))
			self.noisy_files['t0'] = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/*.wav'))
		elif format in ["voicebank", "librispeech-adv", "librispeech-white", "librispeech-demand"]:
			self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
			self.noisy_files['t0'] = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))
			print("from voicebank, get {} clean files and {} noisy files".format(len(self.clean_files), len(self.noisy_files)))
		elif format == "dns":
			self.noisy_files['t0'] = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))
			clean_dir = join(data_dir, subset) + '/clean/'
			self.clean_files = [clean_dir + 'clean_fileid_' \
				+ noisy_file.split('/')[-1].split('_fileid_')[-1] for noisy_file in self.noisy_files['t0']]
		elif format == "reverb_wsj0":
			dic_correspondence_subsets = {"train": "tr", "valid": "cv", "test": "tt"}
			self.clean_files = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/anechoic/*.wav'))
			self.noisy_files['t0'] = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/reverb/*.wav'))
		elif format == "timit":
			dic_correspondence_subsets = {"train": "tr", "valid": "cv", "test": "tt"}
			self.clean_files = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/clean/*.wav'))
			self.noisy_files['t0'] = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/noisy/*.wav'))
			self.transcriptions = sorted(glob(join(data_dir, "transcriptions", dic_correspondence_subsets[subset]) + '/*.txt'))
		if "multi" in self.condition:
			self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
			self.noisy_files = {}
			for t in range(8):
				key = f"t{t}"
				if key in self.condition:
					if t == 0:
						self.noisy_files[key] = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))
					else:
						self.noisy_files[key] = sorted(glob(join(data_dir, subset) + f'/noisy-purified-DDPM_{key}_step1/*.wav'))

		elif format in ["voicebank-diffwave-t7"]:
			self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
			self.noisy_files['t7'] = sorted(glob(join(data_dir, subset) + '/noisy-purified-DDPM_t7_step1/*.wav'))
		elif format in ["librispeech-demand-diffwave-t3", "librispeech-diffwave-t3", "voicebank-diffwave-t3"]:
			self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
			self.noisy_files['t3'] = sorted(glob(join(data_dir, subset) + '/noisy-purified-DDPM_t3_step1/*.wav'))
			print("from librispeech, get {} clean files and {} noisy files".format(len(self.clean_files), len(self.noisy_files)))
		elif format in ["librispeech-demand-diffwave-t4", "librispeech-diffwave-t4", "voicebank-diffwave-t4"]:
			self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
			self.noisy_files['t4'] = sorted(glob(join(data_dir, subset) + '/noisy-purified-DDPM_t4_step1/*.wav'))
			print("from librispeech, get {} clean files and {} noisy files".format(len(self.clean_files), len(self.noisy_files)))
		elif format in ["librispeech-demand-diffwave-t5", "librispeech-diffwave-t5", "voicebank-diffwave-t5"]:
			self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
			self.noisy_files['t5'] = sorted(glob(join(data_dir, subset) + '/noisy-purified-DDPM_t5_step1/*.wav'))
			print("from librispeech, get {} clean files and {} noisy files".format(len(self.clean_files), len(self.noisy_files)))

		self.dummy = dummy
		self.num_frames = num_frames
		self.shuffle_spec = shuffle_spec
		self.normalize_audio = normalize_audio
		self.spec_transform = spec_transform
		self.avgspec_dict = None
		self.use_text = use_text
		if self.use_text > 0:
			self.create_metadata()
		assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
		self.stft_kwargs = stft_kwargs
		self.hop_length = self.stft_kwargs["hop_length"]
		assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"
	def create_metadata(self):
		uttinfo_fpath_subsets = { \
			"train": "/path/to/data/phoneme_avg_spec_dict/libri_get_spec_dataset_train.txt", \
			"valid": "/path/to/data/phoneme_avg_spec_dict/libri_get_spec_dataset_valid.txt", \
			#"test": "/path/to/data/phoneme_avg_spec_dict/libri_get_spec_dataset_test_preprocessed.txt", \
		} 						
		uttinfo_fpath = uttinfo_fpath_subsets[self.subset]
		# uttinfo
		with open(uttinfo_fpath, 'r') as f:
			all_metadata = [line.strip() for line in f.readlines()]
			# Create a dictionary, key is clean_path, value is metadata
			self.metadata_dict = {}
			for meta in all_metadata:
				parts = meta.split('|')
				if len(parts) < 5:
					raise ValueError(f"Metadata line does not have enough parts: {meta}")
				_, phonemes_duration, phonemes_code, clean_path, _ = parts
				self.metadata_dict[os.path.basename(clean_path)] = meta
		# Reorder self._metadata according to the order of self.clean_files
		self._metadata = []
		for clean_path in self.clean_files:
			# Get filename
			filename = os.path.basename(clean_path)
			# Try to find filename match in metadata_dict
			meta = self.metadata_dict.get(filename)
			if meta is None:
				# If full path does not match, try to match only the latter part after 'p'
				meta = self.metadata_dict.get(filename.split('p')[-1])
			if meta is None:
				print(f"Warning: Metadata for clean file {clean_path} not found.")
				#raise ValueError(f"Metadata for clean file {clean_path} not found.")
			self._metadata.append(meta)
		print("metadata_length:", len(self._metadata))

		#self.avg_spec_path = "/path/to/LibriSpeech_wav/amps_avg_libri.txt"
		#self.avg_spec_path = "/path/to/LibriSpeech_wav/amps_avg_libri_spec_factor_0.33.txt"
		self.avg_spec_path = "/path/to/LibriSpeech_wav/amps_avg_libri_spec.txt"
		self.avgspec_dict = self.init_avgspec_dict()

	def init_avgspec_dict(self):
		avgspec_dict = dict()
		with open(self.avg_spec_path) as f:
			lines = [line.strip() for line in f.readlines()]
		for line in lines:
			p, value = line.split('\t')
			value = value[1:-1].split(',')
			value = np.round(np.array([float(x) for x in value]), 1)
			avgspec_dict[p] = value
		return avgspec_dict
	def _open_hdf5(self):
		self.meta_data = json.load(open(sorted(glob(join(self.data_dir, f"*.json")))[-1], "r"))
		self.prep_file = h5py.File(sorted(glob(join(self.data_dir, f"*.hdf5")))[-1], 'r')
	def get_text_spec_from_metadata(self, item_meta, filename):
		try:
			start_frames, phonemes_duration, phonemes_code, clean_path, speaker_id = item_meta.split('|')
			try:
				phonemes_code = phonemes_code.split()
				phonemes_duration = [int(dur) for dur in phonemes_duration.split()]
				text_spec = []
				for ph, dur in zip(phonemes_code, phonemes_duration):
					text_spec.append(np.repeat(
						np.expand_dims(self.avgspec_dict[ph], 1), dur, 1))
				text_spec = np.concatenate(text_spec, axis=1)

			except:
				text_spec = None
				print(f"Warning: Phonemes for clean file {filename} not found.")

		except Exception as e:
			text_spec = None
			print(f"Warning: Metadata of {filename} has wrong format: {item_meta}. Error: {e}")
		return text_spec
	

	
	def __getitem__(self, i, raw=False):
		ret = {}
		if self.use_text:
			try:
				item_meta = self._metadata[i]
				text_spec = self.get_text_spec_from_metadata(item_meta, self.clean_files[i])
			except:
				text_spec = None
				print(f"Warning: Metadata for clean file {self.clean_files[i]} not found.")

		x, sr = load(self.clean_files[i])			
		#y, sr = load(self.noisy_files[i])
		y_list = [load(self.noisy_files[key][i])[0] for key in self.noisy_files.keys()]
		#clean, sr = load(clean_path)
		#print(x.size(-1), y.size(-1), clean.size(-1))
		#min_len = min(x.size(-1), y.size(-1))
		min_len = min(x.size(-1), min(y.size(-1) for y in y_list))
		#x, y = x[..., : min_len], y[..., : min_len] 
		x = x[..., :min_len]
		y_list = [y[..., :min_len] for y in y_list]
		
		if x.ndimension() == 2 and self.spatial_channels == 1:
			#x, y = x[0].unsqueeze(0), y[0].unsqueeze(0) #Select first channel
			x = x[0].unsqueeze(0)  # Select first channel
			y_list = [y[0].unsqueeze(0) for y in y_list]  # Select first channel
		# Select channels
		assert self.spatial_channels <= x.size(0), f"You asked too many channels ({self.spatial_channels}) for the given dataset ({x.size(0)})"
		#x, y = x[: self.spatial_channels], y[: self.spatial_channels]
		x = x[:self.spatial_channels]
		y_list = [y[:self.spatial_channels] for y in y_list]
		if text_spec is not None:
			#audio_spec_length = y.shape[-1] // self.hop_length
			audio_spec_length = y_list[0].shape[-1] // self.hop_length
			if (text_spec.shape[-1] / audio_spec_length) > 1.3 or (text_spec.shape[-1] / audio_spec_length) < 0.8:
				print(f"Warning: file {self.clean_files[i]} textspec length: {text_spec.shape[-1]} and audiospec length: {audio_spec_length} maybe mismatch.")

		if raw:
			X = torch.stft(x, **self.stft_kwargs)
			#Y = torch.stft(y, **self.stft_kwargs)
			Y_list = [torch.stft(y, **self.stft_kwargs) for y in y_list]

			#X, Y = self.spec_transform(X), self.spec_transform(Y)
			X = self.spec_transform(X)
			Y_list = [self.spec_transform(Y) for Y in Y_list]
			ret["clean_wav"] = x
			#ret["noisy_wav"] = y
			ret["noisy_wav"] = y_list
			ret["clean_spec"] = X
			#ret["noisy_spec"] = Y
			ret["noisy_spec"] = Y_list
			if self.use_text:
				if text_spec is not None:
					if text_spec.shape[-1] < Y_list[0].shape[-1]: #Y.shape[-1]:
						padding_length = Y_list[0].shape[-1] - text_spec.shape[-1] # Y.shape[-1] - text_spec.shape[-1]
						text_spec = np.pad(text_spec, 
										((0, 0), (0, padding_length)), 
										mode='constant')
					elif text_spec.shape[-1] > Y_list[0].shape[-1]: #Y.shape[-1]:
						text_spec = text_spec[..., :Y_list[0].shape[-1]] #: Y.shape[-1]]
					text_spec = torch.tensor(text_spec).unsqueeze(0).float()
					ret["text_spec"] = text_spec

				else:
					ret["text_spec"] = torch.zeros_like(Y_list[0]).float() #torch.zeros_like(Y).float()
					
				#print("raw:", ret["text_spec"].shape, x.shape, y_list[0].shape, y_list[1].shape, X.shape, Y_list[0].shape, Y_list[1].shape)
			return ret

		normfac = y_list[0].abs().max() #y.abs().max()

		# formula applies for center=True
		target_len = (self.num_frames - 1) * self.hop_length
		current_len = x.size(-1)
		pad = max(target_len - current_len, 0)
		if pad == 0:
			# extract random part of the audio file
			if self.shuffle_spec:
				start = int(np.random.uniform(0, current_len-target_len))
			else:
				start = int((current_len-target_len)/2)
			x = x[..., start:start+target_len]
			#y = y[..., start:start+target_len]
			y_list = [y[..., start:start + target_len] for y in y_list]
			start_frame = start // self.hop_length
		else:
			# pad audio if the length T is smaller than num_frames
			x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
			#y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')
			y_list = [F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode='constant') for y in y_list]

		if self.normalize_audio:
			# normalize both based on noisy speech, to ensure same clean signal power in x and y.
			x = x / normfac
			#y = y / normfac
			y_list = [y / normfac for y in y_list]



		X = torch.stft(x, **self.stft_kwargs)
		#Y = torch.stft(y, **self.stft_kwargs)
		Y_list = [torch.stft(y, **self.stft_kwargs) for y in y_list]

		#X, Y = self.spec_transform(X), self.spec_transform(Y)
		X = self.spec_transform(X)
		Y_list = [self.spec_transform(Y) for Y in Y_list]
		ret["clean_wav"] = x
		ret["noisy_wav"] = y_list
		ret["clean_spec"] = X
		ret["noisy_spec"] = Y_list

		if self.use_text:
			if text_spec is not None:
				if pad == 0:
					text_spec = text_spec[..., start_frame : start_frame + self.num_frames]
					text_spec_sliced = text_spec
				else:
					# In the case of padding, calculate the number of padded frames
					pad_left_samples = pad // 2
					pad_right_samples = pad // 2 + ( pad % 2 )
					pad_left_frames = math.ceil(pad_left_samples / self.hop_length)
					pad_right_frames = math.ceil(pad_right_samples / self.hop_length)
					# Pad text_spec with 0 on both sides
					text_spec_padded = np.pad(text_spec, 
											((0, 0), (pad_left_frames, pad_right_frames)), 
											mode='constant')
					# Slice to num_frames
					text_spec_sliced = text_spec_padded[..., :self.num_frames]

				# Handle edge cases to ensure the number of frames in text_spec_sliced matches num_frames
				if text_spec_sliced.shape[-1] < self.num_frames:
					pad_width = self.num_frames - text_spec_sliced.shape[-1]
					text_spec_sliced = np.pad(text_spec_sliced, 
											((0, 0), (0, pad_width)), 
											mode='constant')
				elif text_spec_sliced.shape[-1] > self.num_frames:
					text_spec_sliced = text_spec_sliced[..., :self.num_frames]

				ret["text_spec"] = torch.tensor(text_spec_sliced).unsqueeze(0).float()
			else:
				#ret["text_spec"] = torch.zeros_like(Y).float()
				ret["text_spec"] = torch.zeros_like(Y_list[0]).float()
			#print(f"padded:{pad}", text_spec_sliced.shape, x.shape, y.shape, X.shape, Y.shape)
		#print("shape:", ret["text_spec"].shape, x.shape, y_list[0].shape, y_list[1].shape, X.shape, Y_list[0].shape, Y_list[1].shape)
		return ret



	def __len__(self):
		if self.dummy:
			# for debugging shrink the data set sizer
			return int(len(self.clean_files)/10)
		else:
			if self.format == "vctk":
				return len(self.clean_files)//2
			else:
				return len(self.clean_files)





class SpecsDataModule(pl.LightningDataModule):
	def __init__(
		self, base_dir="", format="wsj0", spatial_channels=1, batch_size=8,
		n_fft=510, hop_length=128, num_frames=256, window="hann",
		num_workers=8, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
		gpu=True, return_time=False, use_text = False, condition = "post_denoiser", **kwargs
	):
		super().__init__()
		self.base_dir = base_dir
		self.format = format
		self.spatial_channels = spatial_channels
		self.batch_size = batch_size
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.num_frames = num_frames
		self.window = get_window(window, self.n_fft)
		self.windows = {}
		self.num_workers = num_workers
		self.dummy = dummy
		self.spec_factor = spec_factor
		self.spec_abs_exponent = spec_abs_exponent
		self.gpu = gpu
		self.return_time = return_time
		self.use_text = use_text
		self.condition = condition
		self.kwargs = kwargs

	def setup(self, stage=None):
		specs_kwargs = dict(
			stft_kwargs=self.stft_kwargs, num_frames=self.num_frames, spec_transform=self.spec_fwd,
			**self.stft_kwargs, **self.kwargs
		)
		if stage == 'fit' or stage is None:
			self.train_set = Specs(self.base_dir, 'train', self.dummy, True, 
				format=self.format, spatial_channels=self.spatial_channels, 
				return_time=self.return_time, use_text = self.use_text, condition=self.condition, **specs_kwargs)
			self.valid_set = Specs(self.base_dir, 'valid', self.dummy, False, 
				format=self.format, spatial_channels=self.spatial_channels, 
				return_time=self.return_time, use_text = self.use_text, condition=self.condition, **specs_kwargs)
		if stage == 'test' or stage is None:
			self.test_set = Specs(self.base_dir, 'test', self.dummy, False, 
				format=self.format, spatial_channels=self.spatial_channels, 
				return_time=self.return_time, use_text = self.use_text, condition=self.condition, **specs_kwargs)

	def time_to_spec(self, x):
		#print("time_to_spec: ", x.shape)
		X = self.stft(x.squeeze(1))

		X= self.spec_fwd(X)

		return X.unsqueeze(1)
	def spec_fwd(self, spec):
		if self.spec_abs_exponent != 1:
			e = self.spec_abs_exponent
			spec = spec.abs()**e * torch.exp(1j * spec.angle())
		return spec * self.spec_factor

	def spec_back(self, spec):
		spec = spec / self.spec_factor
		if self.spec_abs_exponent != 1:
			e = self.spec_abs_exponent
			spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
		return spec

	@property
	def stft_kwargs(self):
		return {**self.istft_kwargs, "return_complex": True}

	@property
	def istft_kwargs(self):
		return dict(
			n_fft=self.n_fft, hop_length=self.hop_length,
			window=self.window, center=True
		)

	def _get_window(self, x):
		"""
		Retrieve an appropriate window for the given tensor x, matching the device.
		Caches the retrieved windows so that only one window tensor will be allocated per device.
		"""
		window = self.windows.get(x.device, None)
		if window is None:
			window = self.window.to(x.device)
			self.windows[x.device] = window
		return window

	def stft(self, sig):
		window = self._get_window(sig)
		return torch.stft(sig, **{**self.stft_kwargs, "window": window})

	def istft(self, spec, length=None):
		window = self._get_window(spec)
		return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

	@staticmethod
	def add_argparse_args(parser):
		parser.add_argument("--format", type=str, default="wsj0", help="File paths follow the DNS data description.")
		parser.add_argument("--base_dir", type=str, default="/path/to/mixed_dataset",
			help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, "
				"each of which contain `clean` and `noisy` subdirectories.")
		parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 32 by default.")
		parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
		parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
		parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
		parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'sqrthann' by default.")
		parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for DataLoaders. 4 by default.")
		parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
		parser.add_argument("--spec_factor", type=float, default=0.33, help="Factor to multiply complex STFT coefficients by.") ##### In Simon's current impl, this is 0.15 !
		parser.add_argument("--spec_abs_exponent", type=float, default=0.5,
			help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). "
				"1 by default; set to values < 1 to bring out quieter features.")
		parser.add_argument("--return_time", action="store_true", help="Return the waveform instead of the STFT")

		return parser

	def train_dataloader(self):
		# return DataLoader(
		return DataLoader(
			self.train_set, batch_size=self.batch_size,
			num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
		)

	def val_dataloader(self):
		# return DataLoader(
		return DataLoader(
			self.valid_set, batch_size=self.batch_size,
			num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
		)

	def test_dataloader(self):
		# return DataLoader(
		return DataLoader(
			self.test_set, batch_size=self.batch_size,
			num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
		)














class SpecsAndTranscriptions(Specs):

	def __init__(
		self, data_dir, subset, dummy, shuffle_spec, num_frames, format,
		**kwargs
	):
		super().__init__(data_dir, subset, dummy, shuffle_spec, num_frames, format, **kwargs)
		if format == "timit":
			dic_correspondence_subsets = {"train": "tr", "valid": "cv", "test": "tt"}
			self.clean_files = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/clean/*.wav'))
			self.noisy_files = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/noisy/*.wav'))
			self.transcriptions = sorted(glob(join(data_dir, "transcriptions", dic_correspondence_subsets[subset]) + '/*.txt'))
		else:
			raise NotImplementedError

	def __getitem__(self, i, raw=False):
		X, Y = super().__getitem__(i, raw=raw)
		transcription = open(self.transcriptions[i], "r").read()
		if self.format == "timit": #remove the number at the beginning
			transcription = " ".join(transcription.split(" ")[2: ])

		return X, Y, transcription

	def __len__(self):
		if self.dummy:
			return int(len(self.clean_files)/10)
		else:
			return len(self.clean_files)

class SpecsAndTranscriptionsDataModule(SpecsDataModule):

	def setup(self, stage=None):
		specs_kwargs = dict(
			stft_kwargs=self.stft_kwargs, num_frames=self.num_frames, spec_transform=self.spec_fwd,
			**self.stft_kwargs, **self.kwargs
		)
		if stage == 'fit' or stage is None:
			raise NotImplementedError
		if stage == 'test' or stage is None:
			self.test_set = SpecsAndTranscriptions(self.base_dir, 'test', self.dummy, False, 
			format=self.format, **specs_kwargs)


	@staticmethod
	def add_argparse_args(parser):
		parser.add_argument("--format", type=str, default="reverb_wsj0", choices=["wsj0", "vctk", "dns", "reverb_wsj0"], help="File paths follow the DNS data description.")
		parser.add_argument("--base-dir", type=str, default="/data/lemercier/databases/reverb_wsj0+chime/audio")
		parser.add_argument("--batch-size", type=int, default=8, help="The batch size.")
		parser.add_argument("--num-workers", type=int, default=8, help="Number of workers to use for DataLoaders.")
		parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
		return parser
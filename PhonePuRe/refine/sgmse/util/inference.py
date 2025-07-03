import sre_compile
import torch
from torchaudio import load
from sgmse.util.other import si_sdr, pad_spec, EmbeddingLoss
from pesq import pesq
from tqdm import tqdm
from pystoi import stoi
import numpy as np

# Settings
snr = 0.5
N = 50
corrector_steps = 1

# Plotting settings
MAX_VIS_SAMPLES = 10
n_fft = 512
hop_length = 128

def evaluate_model(model, num_eval_files, spec=False, audio=False, discriminative=False):

	model.eval()
	_pesq, _si_sdr, _estoi, _speaker_emb_similarity = 0., 0., 0., 0.
	if spec:
		noisy_spec_list, estimate_spec_list, clean_spec_list, denoised_spec_list = [], [], [], []
	if audio:
		noisy_audio_list, estimate_audio_list, clean_audio_list, denoised_audio_list = [], [], [], []
	embedding_loss = EmbeddingLoss()
	embedding_loss.emb_model.eval()
	for i in range(num_eval_files):
		# Load wavs
		ret = model.data_module.valid_set.__getitem__(i, raw=True) #d,t
		x = ret["clean_wav"]
		y_list = ret["noisy_wav"] #list
		y_noisy = y_list[0]
		if model.use_text:
			text_spec = ret["text_spec"].cuda()
		else:
			text_spec = None
		#norm_factor = y[0].abs().max().item()
		x_hat, y_denoised = model.enhance([y.cuda() for y in y_list], text_spec = text_spec)
		_speaker_emb_similarity += 1 - embedding_loss(x.unsqueeze(0), x_hat.unsqueeze(0), loss_type = "cos").item()
		if x_hat.ndim == 1:
			x_hat = x_hat.unsqueeze(0).cpu()
			y_denoised = y_denoised.unsqueeze(0).cpu()
		if x.ndim == 1:
			x = x.unsqueeze(0).cpu().numpy()
			x_hat = x_hat.unsqueeze(0).cpu().numpy()
			y_noisy = y_noisy.unsqueeze(0).cpu().numpy()
			y_denoised = y_denoised.unsqueeze(0).cpu().numpy()
		else: #eval only first channel
			x = x[0].unsqueeze(0).cpu().numpy()
			x_hat = x_hat[0].unsqueeze(0).cpu().numpy()
			y_noisy = y_noisy[0].unsqueeze(0).cpu().numpy()
			y_denoised = y_denoised[0].unsqueeze(0).cpu().numpy()

		_si_sdr += si_sdr(x[0], x_hat[0])
		_pesq += pesq(16000, x[0], x_hat[0], 'wb') 
		_estoi += stoi(x[0], x_hat[0], 16000, extended=True)
		
		y_noisy, x_hat, x, y_denoised = torch.from_numpy(y_noisy), torch.from_numpy(x_hat), torch.from_numpy(x), torch.from_numpy(y_denoised)
		if spec and i < MAX_VIS_SAMPLES:
			y_noisy_stft, x_hat_stft, x_stft, y_denoised_stft = model._stft(y_noisy[0]), model._stft(x_hat[0]), model._stft(x[0]), model._stft(y_denoised[0])
			noisy_spec_list.append(y_noisy_stft)
			estimate_spec_list.append(x_hat_stft)
			clean_spec_list.append(x_stft)
			denoised_spec_list.append(y_denoised_stft)

		if audio and i < MAX_VIS_SAMPLES:
			noisy_audio_list.append(y_noisy[0])
			estimate_audio_list.append(x_hat[0])
			clean_audio_list.append(x[0])
			denoised_audio_list.append(y_denoised[0])

	if spec:
		if audio:
			return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _speaker_emb_similarity/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list, denoised_spec_list], [noisy_audio_list, estimate_audio_list, clean_audio_list, denoised_audio_list]
		else:
			return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _speaker_emb_similarity/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list, denoised_spec_list], None
	elif audio and not spec:
			return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _speaker_emb_similarity/num_eval_files, None, [noisy_audio_list, estimate_audio_list, clean_audio_list, denoised_audio_list]
	else:
		return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, _speaker_emb_similarity/num_eval_files, None, None


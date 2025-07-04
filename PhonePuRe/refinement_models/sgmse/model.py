from math import ceil
import warnings

import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
#import wandb
import time
import os
import numpy as np

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.graphics import visualize_example, visualize_one
from sgmse.util.other import pad_spec, si_sdr_torch, EmbeddingLoss
VIS_EPOCHS = 5 

torch.autograd.set_detect_anomaly(True)

class ScoreModel(pl.LightningModule):
    def __init__(self,
        backbone: str = "ncsnpp", sde: str = "ouvesde",
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 3e-2, transform: str = 'none', nolog: bool = False,
        num_eval_files: int = 50, loss_type: str = 'mse', data_module_cls = None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a score-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        kwargs.update(input_channels=4)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        self.nolog = nolog

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae", "gaussian_entropy", "kristina", "sisdr", "time_mse"), help="The type of loss function to use.")
        parser.add_argument("--spatial_channels", type=int, default=1)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, err_time=None, err_mag=None):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == 'mae':
            losses = err.abs()
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        return loss

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)

    def _raw_dnn_output(self, x, t, y):
        dnn_input = torch.cat([x, y], dim=1) #b,2*d,f,t
        return self.dnn(dnn_input, t)

    def forward(self, x, t, y, **kwargs):
        score = -self._raw_dnn_output(x, t, y)
        std = self.sde._std(t, y=y)
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,)*(y.ndim - std.ndim)))
        return score

    def _step(self, batch, batch_idx):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            assert "bwe" in self.data_module.task, "Received metadata for a task which is not BWE"
            x, y, scale_factors = batch
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5 ---> problem: this cannot work for FreqOUVE, because is standard, and tries to match a score with a sigma which is not standard
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,)*(y.ndim - std.ndim)))
        sigmas = std
        perturbed_data = mean + sigmas * z
        score = self(perturbed_data, t, y)
        err = score * sigmas + z
        loss = self._loss(err)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_est, si_sdr_est, estoi_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print('__________________________________________________________________')
            
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True)
            self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True)
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True)

            if audio is not None:
                y_list, x_hat_list, x_list = audio
                for idx, (y, x_hat, x) in enumerate(zip(y_list, x_hat_list, x_list)):
                    if self.current_epoch == 0:
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))).unsqueeze(0), sample_rate=sr, global_step=self.global_step)
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(x)).unsqueeze(0), sample_rate=sr, global_step=self.global_step)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))).unsqueeze(0), sample_rate=sr, global_step=self.global_step)

            if spec is not None:
                figures = []
                y_stft_list, x_hat_stft_list, x_stft_list = spec
                for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list)):
                    figures.append(
                        visualize_example(
                        torch.abs(y_stft), 
                        torch.abs(x_hat_stft), 
                        torch.abs(x_stft), return_fig=True))
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures)

        return loss

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, scale_factor=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=50, corrector_steps=1, snr=0.5, timeit=False,
        scale_factor = None, return_stft=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y, N=N,
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                scale_factor=scale_factor,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y, N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()

        if return_stft:
            return sample.squeeze(), Y.squeeze(), T_orig, norm_factor

        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu()
        end = time.time()
        if timeit:
            sr = 16000
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat









class DiscriminativeModel(ScoreModel):

    def forward(self, y):
        if self.dnn.FORCE_STFT_OUT:
            y = self._istft(self._backward_transform(y.clone().squeeze(1)))
        t = torch.ones(y.shape[0], device=y.device)
        x_hat = self.dnn(y, t)
        return x_hat

    def _loss(self, x, xhat):
        if self.dnn.FORCE_STFT_OUT:
            x = self._istft(self._backward_transform(x.clone().squeeze(1)))

        if self.loss_type == 'mse':
            losses = torch.square((x - xhat).abs())
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == 'mae':
            losses = (x - xhat).abs()
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        elif self.loss_type == "sisdr":
            loss = - torch.mean(torch.stack([si_sdr_torch(x[i], xhat[i]) for i in range(x.size(0))]))
        return loss

    def _step(self, batch, batch_idx):
        X, Y = batch
        Xhat = self(Y)
        loss = self._loss(X, Xhat)
        return loss

    def enhance(self, y, **ignored_kwargs):
        with torch.no_grad():
            norm_factor = y.abs().max().item()
            T_orig = y.size(1)

            if self.data_module.return_time:
                Y = torch.unsqueeze((y/norm_factor).cuda(), 0) #1,D=1,T
            else:
                Y = torch.unsqueeze(self._forward_transform(self._stft((y/norm_factor).cuda())), 0) #1,D,F,T
                Y = pad_spec(Y)
            X_hat = self(Y)
            if self.dnn.FORCE_STFT_OUT:
                X_hat = self._forward_transform(self._stft(X_hat)).unsqueeze(1)

            if self.data_module.return_time:
                x_hat = X_hat.squeeze()
            else:
                x_hat = self.to_audio(X_hat.squeeze(), T_orig)

            return (x_hat * norm_factor).squeeze()
                    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx, discriminative=True)
    

















class StochasticRegenerationModel(pl.LightningModule):
    def __init__(self,
        backbone_denoiser: str, backbone_score: str, sde: str,
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 3e-2, nolog: bool = False, num_eval_files: int = 50,
        loss_type_denoiser: str = "none", loss_type_score: str = 'mse', data_module_cls = None, 
        mode = "regen-joint-training", condition = "both",
        use_text = False,
        **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a score-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        super().__init__()
        # Initialize Backbone DNN
        kwargs_denoiser = kwargs
        kwargs_denoiser.update(input_channels=2)
        kwargs_denoiser.update(discriminative=True)
        self.denoiser_net = BackboneRegistry.get_by_name(backbone_denoiser)(**kwargs) if backbone_denoiser != "none" else None
        self.use_text = use_text
        self.condition = condition
        in_channels= (6 if condition == "both" else 4)
        if self.use_text:
            in_channels += 2
        if "multi" in self.condition:
            for t in range(8):
                key = f"t{t}"
                if key in self.condition:
                    in_channels += 2
            in_channels -= 2
        if backbone_score == "wavenet":
            in_channels = in_channels // 2
        kwargs.update(input_channels=in_channels)
        kwargs_denoiser.update(discriminative=False)
        self.score_net = BackboneRegistry.get_by_name(backbone_score)(**kwargs) if backbone_score != "none" else None

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        self.t_eps = t_eps
        # get loss type
        self.loss_type_denoiser = loss_type_denoiser
        self.loss_type_score = loss_type_score
        if "weighting_denoiser_to_score" in kwargs.keys():
            self.weighting_denoiser_to_score = kwargs["weighting_denoiser_to_score"]
        else:
            self.weighting_denoiser_to_score = .5
        # if loss_type is emb, load the embedding model

        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False

        self.mode = mode
        self.configure_losses()

        self.num_eval_files = num_eval_files
        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0, use_text = self.use_text, condition = self.condition)
        #self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        self._reduce_op = lambda *args, **kwargs: 0.5 * torch.mean(*args, **kwargs)
        self.nolog = nolog

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type_denoiser", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_type_score", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--weighting_denoiser_to_score", type=float, default=0.5, help="a, as in L = a * L_denoiser + (1-a) * .")
        parser.add_argument("--weighting_emb_to_denoiser", type=float, default=0.5, help="a, as in L = a * L_emb + (1-a) * L_denoiser.")
        parser.add_argument("--spatial_channels", type=int, default=1)
        
        return parser

    def configure_losses(self):
        # Score Loss
        if "mse" in self.loss_type_score:
            self.loss_fn_score = lambda err: self._reduce_op(torch.square(torch.abs(err)))
        elif "mae" in self.loss_type_score :
            self.loss_fn_score = lambda err: self._reduce_op(torch.abs(err))
        elif self.loss_type_score == "none":
            raise NotImplementedError
            self.loss_fn_score = None
        else:
            raise NotImplementedError
        
        # Denoiser Loss
        if "mse" in self.loss_type_denoiser:
            self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.square(torch.abs(x - y)))
        elif "mae" in self.loss_type_denoiser:
            self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.abs(x - y))
        elif self.loss_type_denoiser == "none":
            self.loss_fn_denoiser = None
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def load_denoiser_model(self, checkpoint):
        self.denoiser_net = DiscriminativeModel.load_from_checkpoint(checkpoint).dnn
        if "freeze-denoiser" in self.mode:
            for param in self.denoiser_net.parameters():
                param.requires_grad = False

    def load_score_model(self, checkpoint):
        self.score_net = ScoreModel.load_from_checkpoint(checkpoint).dnn

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err, x, y_denoised, x_wav = None, y_denoised_wav = None):
        loss_score = self.loss_fn_score(err) if self.loss_type_score != "none" else None
        if self.loss_type_denoiser != "none" and self.denoiser_net is not None:
            if "time" in self.loss_type_denoiser:
                loss_denoiser = self.loss_fn_denoiser(y_denoised_wav, x_wav)
            else:
                loss_denoiser = self.loss_fn_denoiser(y_denoised, x)
        else:
            loss_denoiser = None
        if loss_score is not None:
            if loss_denoiser is not None:
                loss = self.weighting_denoiser_to_score * loss_denoiser + (1 - self.weighting_denoiser_to_score) * loss_score
            else:
                loss = loss_score
        else:
            loss = loss_denoiser
        #print('loss:', loss, 'loss_score:', loss_score, 'loss_denoiser:', loss_denoiser)
        return loss, loss_score, loss_denoiser

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)

    def forward_score(self, x, t, score_conditioning, sde_input, **kwargs):
        #print(x.shape, score_conditioning[0].shape)
        dnn_input = torch.cat([x] + score_conditioning, dim=1) #b,n_input*d,f,t

        #print(dnn_input.shape, t.shape)
        if self.score_net.domain == "time":
            score = -self.score_net(dnn_input.squeeze(2), t.unsqueeze(1)).unsqueeze(2)
        elif self.score_net.domain == "TF":
            score = -self.score_net(dnn_input, t)
        std = self.sde._std(t, y=sde_input)
        if std.ndim < sde_input.ndim:
            std = std.view(*std.size(), *((1,)*(sde_input.ndim - std.ndim)))
        return score

    def forward_denoiser(self, y, **kwargs):
        if self.denoiser_net is None:
            return y
        x_hat = self.denoiser_net(y)
        return x_hat
    def embedding_loss(self, x_wav, y_denoised_wav):
        x_emb = self.emb_model.encode_batch(x_wav.squeeze(1))
        y_denoised_emb = self.emb_model.encode_batch(y_denoised_wav.squeeze(1))
        #cos_sim = torch.nn.functional.cosine_similarity(x_emb, y_denoised_emb, dim=1)
        #return 1 - cos_sim.mean()
        return torch.nn.functional.mse_loss(x_emb, y_denoised_emb)

    def _step(self, batch, batch_idx):
        if self.use_text:
            text_spec = batch["text_spec"]
        else:
            text_spec = None
        if self.data_module.return_time:
            x_wav = batch["clean_wav"]
            y_wav_list = batch["noisy_wav"]
            x = batch["clean_spec"]
            y_list = batch["noisy_spec"] #list of noisy specs
            y_noisy = y_list[0]
            y_input = y_list[-1]
            y_noisy_wav = y_wav_list[0]
            y_input_wav = y_wav_list[-1]

            # Denoising step
            if self.denoiser_net is None:
                y_denoised = y_input
                y_denoised_wav = y_input_wav
            elif self.denoiser_net.domain == "time":
                with torch.set_grad_enabled("freeze-denoiser" not in self.mode):
                    y_denoised_wav = self.forward_denoiser(y_noisy_wav)
                    y_denoised = self.data_module.time_to_spec(y_denoised_wav)
            elif self.denoiser_net.domain == "TF":
                with torch.set_grad_enabled("freeze-denoiser" not in self.mode):
                    y_denoised = self.forward_denoiser(y_noisy)
                    y_denoised_wav = self.to_audio(y_denoised.squeeze(1), length=x_wav.shape[-1])
            else:
                raise NotImplementedError
        else:
            x, y = batch
            x_wav = self.to_audio(x.squeeze(1))
            y_wav = [self.to_audio(y.squeeze(1)) for y in y]
            #print("x:", x.shape, "y:", y.shape)
            # Denoising step
            if self.denoiser_net is not None:
                with torch.set_grad_enabled("freeze-denoiser" not in self.mode):
                    y_denoised = [self.forward_denoiser(y) for y in y]
                    y_denoised_wav = [self.to_audio(y_denoised.squeeze(1), length=x_wav.shape[-1]) for y_denoised in y_denoised]
            else:
                y_denoised = y
                y_denoised_wav = y_wav



        # Score step
        #print("x:", x.shape, "y_denoised:", y_denoised.shape, "y:", y.shape)
        if self.score_net.domain == "time":
            sde_target = x_wav.unsqueeze(1)
            sde_input = y_denoised_wav.unsqueeze(1)
        elif self.score_net.domain == "TF":
            sde_target = x
            sde_input = y_denoised
        # Forward process
        t = torch.rand(sde_target.shape[0], device=sde_target.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(sde_target, t, sde_input)
        z = torch.randn_like(sde_target)  # i.i.d. normal distributed with var=0.5
        if std.ndim < sde_input.ndim:
            std = std.view(*std.size(), *((1,)*(sde_input.ndim - std.ndim)))
        sigmas = std
        #print(mean.shape, sigmas.shape, z.shape, (sigmas * z).shape)
        perturbed_data = mean + sigmas * z

        # Score estimation
        if self.condition == "noisy":
            score_conditioning = [y] if self.score_net.domain == "TF" else [y_wav.unsqueeze(1)]
        elif self.condition == "post_denoiser":
            score_conditioning = [y_denoised] if self.score_net.domain == "TF" else [y_denoised_wav.unsqueeze(1)]
        elif self.condition == "both":
            score_conditioning = [y, y_denoised] if self.score_net.domain == "TF" else [y_wav.unsqueeze(1), y_denoised_wav.unsqueeze(1)]
        elif "multi" in self.condition:
            score_conditioning = y_list if self.score_net.domain == "TF" else [y_wav.unsqueeze(1) for y_wav in y_wav_list]
        if self.use_text:
            score_conditioning += [text_spec]
        #print(score_condition.shape for score_condition in score_conditioning)
        #print("pertubed_data:", perturbed_data.shape, "t:", t.shape, "score_conditioning[0]:", score_conditioning[0].shape, "sde_input:", sde_input.shape)
        #print("score_conditioning:", score_conditioning[0].shape, score_conditioning[1].shape, sde_input.shape, text_spec.shape)
        score = self.forward_score(perturbed_data, t, score_conditioning, sde_input)
        err = score * sigmas + z
        

        loss, loss_score, loss_denoiser = self._loss(err, x, y_denoised, x_wav.squeeze(1), y_denoised_wav)

        return loss, loss_score, loss_denoiser

    def training_step(self, batch, batch_idx):
        loss, loss_score, loss_denoiser = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        self.log('train_loss_score', loss_score, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_denoiser is not None:
            self.log('train_loss_denoiser', loss_denoiser, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss, loss_score, loss_denoiser = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        self.log('valid_loss_score', loss_score, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_denoiser is not None:
            self.log('valid_loss_denoiser', loss_denoiser, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_est, si_sdr_est, estoi_est, spkemb_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print(f"Speaker Embedding Cosine Similarity at epoch {self.current_epoch} : {spkemb_est:.2f}")
            print('__________________________________________________________________')
            
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True)
            self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True)
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True)
            self.log('ValidationSECS', spkemb_est, on_step=False, on_epoch=True)

            if audio is not None:
                y_list, x_hat_list, x_list, y_denoised_list = audio
                for idx, (y, x_hat, x, y_denoised) in enumerate(zip(y_list, x_hat_list, x_list, y_denoised_list)):
                    if self.current_epoch == 0:
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))).unsqueeze(1), sample_rate=sr, global_step=self.global_step)
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(x)).unsqueeze(1), sample_rate=sr, global_step=self.global_step)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))).unsqueeze(1), sample_rate=sr, global_step=self.global_step)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Denoised/{idx}", (y_denoised / torch.max(torch.abs(y_denoised))).unsqueeze(1), sample_rate=sr, global_step=self.global_step)

            if spec is not None:
                figures = []
                y_stft_list, x_hat_stft_list, x_stft_list, y_denoised_stft_list = spec
                for idx, (y_stft, x_hat_stft, x_stft, y_denoised_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list, y_denoised_stft_list)):
                    figures.append(
                        visualize_example(
                        torch.abs(y_stft), 
                        torch.abs(x_hat_stft), 
                        torch.abs(x_stft), 
                        torch.abs(y_denoised_stft),
                        return_fig=True))
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures)

        return loss

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, scale_factor=None, conditioning=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self.forward_score, y=y, conditioning=conditioning, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self.forward_score, y=y_mini, conditioning=conditioning, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self.forward_score, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self.forward_score, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y_list, sampler_type="pc", predictor="reverse_diffusion",
        corrector="none", N=30, corrector_steps=1, snr=0.5, timeit=False,
        scale_factor = None, return_stft=False, denoiser_only=False, text_spec = None,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        y_noisy = y_list[0]
        y_input = y_list[-1]
        T_orig = y_noisy.size(1)
        norm_factor = y_noisy.abs().max().item()
        y_list = [y / norm_factor for y in y_list]
        Y_list = [torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0) for y in y_list]
        Y_list = [pad_spec(Y) for Y in Y_list]
        Y_noisy = Y_list[0]
        Y_input = Y_list[-1]
        # first one is the original noisy one; and the last one is the score input
        if self.use_text:
            if text_spec is not None:
                text_spec = pad_spec(text_spec.unsqueeze(0).cuda())
            else:
                print("Warning: use_text is True but text_spec not found.")
                text_spec = torch.zeros_like(Y_list[0]).float().cuda()

        with torch.no_grad():

            if self.denoiser_net is not None:
                if self.denoiser_net.domain == "time":
                    y_denoised = self.forward_denoiser(y_noisy.unsqueeze(1).cuda()).squeeze(1)
                    Y_denoised = torch.unsqueeze(self._forward_transform(self._stft(y_denoised.cuda())), 0)
                    Y_denoised = pad_spec(Y_denoised)
                else:
                    Y_denoised = self.forward_denoiser(Y_noisy)
            else:
                Y_denoised = Y_input
                y_denoised = y_input

            if self.score_net is not None and not denoiser_only:
                # Conditioning
                if self.condition == "noisy":
                    score_conditioning = [Y_noisy] if self.score_net.domain == "TF" else [y_noisy.unsqueeze(1)]
                elif self.condition == "post_denoiser":
                    score_conditioning = [Y_denoised] if self.score_net.domain == "TF" else [y_denoised.unsqueeze(1)]
                elif self.condition == "both":
                    score_conditioning = [Y_noisy, Y_denoised] if self.score_net.domain == "TF" else [y_noisy.unsqueeze(1), y_denoised.unsqueeze(1)]
                elif "multi" in self.condition:
                    score_conditioning = Y_list if self.score_net.domain == "TF" else [y.unsqueeze(1) for y in y_list]
                else:
                    raise NotImplementedError(f"Don't know the conditioning you have wished for: {self.condition}")
                if self.use_text:
                    score_conditioning += [text_spec]
                # Reverse process
                sample_input = Y_denoised if self.score_net.domain == "TF" else y_denoised.unsqueeze(1)
                #print("sample_input:", sample_input.shape, "conditional:", score_conditioning[0].shape)
                if sample_input.ndim < 4:
                    sample_input = sample_input.unsqueeze(1)
                    score_conditioning = [c.unsqueeze(1) for c in score_conditioning]
                if sampler_type == "pc":
                    sampler = self.get_pc_sampler(predictor, corrector, sample_input, N=N,
                        corrector_steps=corrector_steps, snr=snr, intermediate=False,
                        scale_factor=scale_factor, conditioning=score_conditioning,
                        **kwargs)
                elif sampler_type == "ode":
                    sampler = self.get_ode_sampler(sample_input, N=N, 
                        conditioning=score_conditioning, 
                        **kwargs)
                else:
                    print("{} is not a valid sampler type!".format(sampler_type))
                sample, nfe = sampler()
            else:
                sample = sample_input

            if return_stft:
                return sample.squeeze(), [Y.squeeze() for Y in Y_list], T_orig, norm_factor
        if self.score_net.domain == "time":
            x_hat = sample.squeeze()
        else:
            x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze()
        y_denoised = y_denoised.squeeze()
        end = time.time()
        if timeit:
            sr = 16000
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat, y_denoised

class NoScoreRegenerationModel(pl.LightningModule):
    def __init__(self,
        backbone_denoiser: str, backbone_score: str, sde: str,
        lr: float = 1e-4, ema_decay: float = 0.999,
        t_eps: float = 3e-2, nolog: bool = False, num_eval_files: int = 50,
        loss_type_denoiser: str = "none", loss_type_score: str = 'mse', data_module_cls = None, 
        mode = "regen-joint-training", condition = "both",
        **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: The underlying backbone DNN that serves as a score-based model.
                Must have an output dimensionality equal to the input dimensionality.
            sde: The SDE to use for the diffusion.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            reduce_mean: If `True`, average the loss across data dimensions.
                Otherwise sum the loss across data dimensions.
        """
        super().__init__()
        # Initialize Backbone DNN
        kwargs_denoiser = kwargs
        kwargs_denoiser.update(input_channels=2)
        kwargs_denoiser.update(discriminative=True)
        self.denoiser_net = BackboneRegistry.get_by_name(backbone_denoiser)(**kwargs) if backbone_denoiser != "none" else None

        #kwargs.update(input_channels=(6 if condition == "both" else 4))
        #kwargs_denoiser.update(discriminative=True)
        # for discriminative model as the second "score" model
        self.score_net = BackboneRegistry.get_by_name(backbone_score)(**kwargs) if backbone_score != "none" else None

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        self.t_eps = t_eps
        # get loss type
        self.loss_type_denoiser = loss_type_denoiser
        self.loss_type_score = loss_type_score
        if "weighting_denoiser_to_score" in kwargs.keys():
            self.weighting_denoiser_to_score = kwargs["weighting_denoiser_to_score"]
        else:
            self.weighting_denoiser_to_score = .5
        # if loss_type is emb, load the embedding model
        if 'emb' in self.loss_type_denoiser or 'emb' in self.loss_type_score:
            self.weighting_emb_to_score = kwargs.get('weighting_emb_to_score', 0.5)
            self.embedding_loss = EmbeddingLoss()
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False

        self.condition = condition
        self.mode = mode
        self.configure_losses()

        self.num_eval_files = num_eval_files
        self.save_hyperparameters(ignore=['nolog'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        #self._reduce_op = lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        self._reduce_op = lambda *args, **kwargs: 0.5 * torch.mean(*args, **kwargs)
        self.nolog = nolog

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type_denoiser", type=str, default="time-mse", help="The type of loss function to use.")
        parser.add_argument("--loss_type_score", type=str, default="TF-mse", help="The type of loss function to use.")
        #parser.add_argument("--loss_type_denoiser", type=str, default="mse", choices=("none", "mse", "mae", "sisdr", "mse_cplx+mag", "mse_time+mag", "emb"), help="The type of loss function to use.")
        #parser.add_argument("--loss_type_score", type=str, default="mse", choices=("none", "mse", "mae", "emb"), help="The type of loss function to use.")
        parser.add_argument("--weighting_denoiser_to_score", type=float, default=0.5, help="a, as in L = a * L_denoiser + (1-a) * .")
        parser.add_argument("--weighting_emb_to_score", type=float, default=0.5, help="a, as in L = a * L_emb + (1-a) * L_score.")
        #parser.add_argument("--condition", default="post_denoiser", choices=["noisy", "post_denoiser", "both"])
        parser.add_argument("--spatial_channels", type=int, default=1)
        
        return parser

    def configure_losses(self):
        # Score Loss
        '''if self.loss_type_score == "mse":
            self.loss_fn_score = lambda err: self._reduce_op(torch.square(torch.abs(err)))
        elif self.loss_type_score == "mae":
            self.loss_fn_score = lambda err: self._reduce_op(torch.abs(err))
        elif self.loss_type_score == "none":
            raise NotImplementedError
            self.loss_fn_score = None
        else:
            raise NotImplementedError'''
        if "mse" in self.loss_type_score:
            self.loss_fn_score = lambda x, y: self._reduce_op(torch.square(torch.abs(x - y)))
        elif "mae" in self.loss_type_score:
            self.loss_fn_score = lambda x, y: self._reduce_op(torch.abs(x - y))
        elif self.loss_type_score == "none":
            self.loss_fn_score = None
        else:
            raise NotImplementedError
        # Denoiser Loss
        if "mse" in self.loss_type_denoiser:
            self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.square(torch.abs(x - y)))
        elif "mae" in self.loss_type_denoiser:
            self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.abs(x - y))
        elif self.loss_type_denoiser == "none":
            self.loss_fn_denoiser = None
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    def load_denoiser_model(self, checkpoint):
        self.denoiser_net = DiscriminativeModel.load_from_checkpoint(checkpoint).dnn
        if "freeze-denoiser" in self.mode:
            for param in self.denoiser_net.parameters():
                param.requires_grad = False

    def load_score_model(self, checkpoint):
        self.score_net = DiscriminativeModel.load_from_checkpoint(checkpoint).dnn

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, x, y_denoised, y_estimated, x_wav = None, y_denoised_wav = None, y_estimated_wav = None):
        #print(x.shape, y_denoised.shape, y_estimated.shape)
        # Initialize loss to 0
        loss = 0.
        if self.loss_type_score != "none" :
            if "time" in self.loss_type_score:
                loss_score = self.loss_fn_score(y_estimated_wav, x_wav)
                #print("loss_score:", loss_score.item())
            elif "TF" in self.loss_type_score:
                loss_score = self.loss_fn_score(y_estimated, x)
            if "emb" in self.loss_type_score:
                #print('x.shape:', x_wav.shape,'y.shape:',  y_estimated_wav.shape)
                loss_emb = self.embedding_loss(x_wav, y_estimated_wav)
            else:
                loss_emb = None


        else:
            loss_score = None
            loss_emb = None

        if self.loss_type_denoiser != "none" and self.denoiser_net is not None:
            if "time" in self.loss_type_denoiser:
                loss_denoiser = self.loss_fn_denoiser(y_denoised_wav, x_wav)
            elif "TF" in self.loss_type_denoiser:
                loss_denoiser = self.loss_fn_denoiser(y_denoised, x)
            else:
                raise NotImplementedError
        else:
            loss_denoiser = None
        # Aggregate the different losses based on the available values
        if loss_score is not None:
            loss += loss_score

        if loss_emb is not None:
            loss += self.weighting_emb_to_score * loss_emb

        if loss_denoiser is not None:
            loss += self.weighting_denoiser_to_score * loss_denoiser
        #print('loss:', loss.item(), 'loss_score:', loss_score.item(), 'loss_denoiser:', loss_denoiser.item(), 'loss_emb:', loss_emb.item())
        return loss, loss_score, loss_denoiser, loss_emb

    def _weighted_mean(self, x, w):
        return torch.mean(x * w)

    def forward_score(self, y_denoised, **kwargs):
        #print(x.shape, score_conditioning[0].shape)
        '''dnn_input = torch.cat([x] + score_conditioning, dim=1) #b,n_input*d,f,t
        score = -self.score_net(dnn_input, t)
        std = self.sde._std(t, y=sde_input)
        if std.ndim < sde_input.ndim:
            std = std.view(*std.size(), *((1,)*(sde_input.ndim - std.ndim)))
        return score'''
        y_estimated = self.score_net(y_denoised)
        return y_estimated

    def forward_denoiser(self, y, **kwargs):
        x_hat = self.denoiser_net(y)
        return x_hat

    def _step(self, batch, batch_idx):
        if self.data_module.return_time:
            x_wav, y_wav, x, y = batch

            # Denoising step
            if self.denoiser_net is None:
                y_denoised = y
                y_denoised_wav = y_wav
            elif self.denoiser_net.domain == "time":
                with torch.set_grad_enabled("freeze-denoiser" not in self.mode):
                    y_denoised_wav = self.forward_denoiser(y_wav)
                    y_denoised = self.data_module.time_to_spec(y_denoised_wav)
            elif self.denoiser_net.domain == "TF":
                with torch.set_grad_enabled("freeze-denoiser" not in self.mode):
                    y_denoised = self.forward_denoiser(y)
                    y_denoised_wav = self.to_audio(y_denoised.squeeze(1), length=x_wav.shape[-1])
            else:
                raise NotImplementedError
        else:
            x, y = batch
            x_wav = self.to_audio(x.squeeze(1))
            y_wav = self.to_audio(y.squeeze(1))
            #print("x:", x.shape, "y:", y.shape)
            # Denoising step
            if self.denoiser_net is not None:
                with torch.set_grad_enabled("freeze-denoiser" not in self.mode):
                    y_denoised = self.forward_denoiser(y)
                    y_denoised_wav = self.to_audio(y_denoised.squeeze(1), length=x_wav.shape[-1])
            else:
                y_denoised = y
                y_denoised_wav = y_wav


        # Score estimation
        if self.condition == "noisy":
            conditioning = y
            conditioning_wav = y_wav
        elif self.condition == "post_denoiser":
            conditioning = y_denoised
            conditioning_wav = y_denoised_wav
        elif self.condition == "both":
            conditioning = (y + y_denoised) / 2
            conditioning_wav = (y_wav + y_denoised_wav) / 2
        else:
            raise NotImplementedError(f"Don't know the conditioning you have wished for: {self.condition}")
        #print("pertubed_data:", perturbed_data.shape, "t:", t.shape, "score_conditioning[0]:", score_conditioning[0].shape, "sde_input:", sde_input.shape)
        if self.score_net.domain == "time":
            y_estimated_wav = self.forward_score(conditioning_wav)
            y_estimated = self.data_module.time_to_spec(y_estimated_wav)
        else:
            y_estimated = self.forward_score(conditioning)
            y_estimated_wav = self.to_audio(y_estimated.squeeze(1), length=x_wav.shape[-1])

        '''loss, loss_score, loss_denoiser = self._loss(err, y_denoised, x, x_wav.squeeze(1), y_estimated_wav)'''
        loss, loss_score, loss_denoiser, loss_emb = self._loss(x, y_denoised, y_estimated, x_wav.squeeze(1), y_denoised_wav, y_estimated_wav)
        return loss, loss_score, loss_denoiser, loss_emb

    def training_step(self, batch, batch_idx):
        loss, loss_score, loss_denoiser, loss_emb = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        self.log('train_loss_score', loss_score, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_denoiser is not None:
            self.log('train_loss_denoiser', loss_denoiser, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_emb is not None:
            self.log('train_loss_emb', loss_emb, on_step=True, on_epoch=True, batch_size=self.data_module.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, discriminative=False, sr=16000):
        loss, loss_score, loss_denoiser, loss_emb = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        self.log('valid_loss_score', loss_score, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_denoiser is not None:
            self.log('valid_loss_denoiser', loss_denoiser, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        if loss_emb is not None:
            self.log('valid_loss_emb', loss_emb, on_step=False, on_epoch=True, batch_size=self.data_module.batch_size)
        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq_est, si_sdr_est, estoi_est, spkemb_est, spec, audio = evaluate_model(self, self.num_eval_files, spec=not self.current_epoch%VIS_EPOCHS, audio=not self.current_epoch%VIS_EPOCHS, discriminative=discriminative)
            print(f"PESQ at epoch {self.current_epoch} : {pesq_est:.2f}")
            print(f"SISDR at epoch {self.current_epoch} : {si_sdr_est:.1f}")
            print(f"ESTOI at epoch {self.current_epoch} : {estoi_est:.2f}")
            print(f"Speaker Embedding Cosine Similarity at epoch {self.current_epoch} : {spkemb_est:.2f}")
            print('__________________________________________________________________')
            
            self.log('ValidationPESQ', pesq_est, on_step=False, on_epoch=True)
            self.log('ValidationSISDR', si_sdr_est, on_step=False, on_epoch=True)
            self.log('ValidationESTOI', estoi_est, on_step=False, on_epoch=True)
            self.log('ValidationSECS', spkemb_est, on_step=False, on_epoch=True)

            if audio is not None:
                y_list, x_hat_list, x_list, y_denoised_list = audio
                for idx, (y, x_hat, x, y_denoised) in enumerate(zip(y_list, x_hat_list, x_list, y_denoised_list)):
                    if self.current_epoch == 0:
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y / torch.max(torch.abs(y))).unsqueeze(1), sample_rate=sr, global_step=self.global_step)
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x / torch.max(x)).unsqueeze(1), sample_rate=sr, global_step=self.global_step)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat / torch.max(torch.abs(x_hat))).unsqueeze(1), sample_rate=sr, global_step=self.global_step)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Denoised/{idx}", (y_denoised / torch.max(torch.abs(y_denoised))).unsqueeze(1), sample_rate=sr, global_step=self.global_step)

            if spec is not None:
                figures = []
                y_stft_list, x_hat_stft_list, x_stft_list, y_denoised_stft_list = spec
                for idx, (y_stft, x_hat_stft, x_stft, y_denoised_stft) in enumerate(zip(y_stft_list, x_hat_stft_list, x_stft_list, y_denoised_stft_list)):
                    figures.append(
                        visualize_example(
                        torch.abs(y_stft), 
                        torch.abs(x_hat_stft), 
                        torch.abs(x_stft), 
                        torch.abs(y_denoised_stft),
                        return_fig=True))
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}/Spec", figures)

        return loss

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, scale_factor=None, conditioning=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self.forward_score, y=y, conditioning=conditioning, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self.forward_score, y=y_mini, conditioning=conditioning, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)
    def to_spec(self, audio):
        return self._forward_transform(self._stft(audio))
    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="none", N=30, corrector_steps=1, snr=0.5, timeit=False,
        scale_factor = None, return_stft=False, denoiser_only=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        with torch.no_grad():

            if self.denoiser_net is not None:
                if self.denoiser_net.domain == "time":
                    y_denoised = self.forward_denoiser(y.unsqueeze(1).cuda()).squeeze(1)
                    Y_denoised = torch.unsqueeze(self._forward_transform(self._stft(y_denoised.cuda())), 0)
                    Y_denoised = pad_spec(Y_denoised)
                elif self.denoiser_net.domain == "TF":
                    Y_denoised = self.forward_denoiser(Y)
                    y_denoised = self.to_audio(Y_denoised.squeeze(1), length=T_orig)
                else:
                    raise NotImplementedError
            else:
                Y_denoised = Y
                y_denoised = y

            if self.score_net is not None and not denoiser_only:
                # Conditioning
                if self.condition == "noisy":
                    score_conditioning = Y
                    score_conditioning_wav = y
                elif self.condition == "post_denoiser":
                    score_conditioning = Y_denoised
                    score_conditioning_wav = y_denoised
                elif self.condition == "both":
                    score_conditioning = (Y + Y_denoised) / 2
                    score_conditioning_wav = (y.cuda() + y_denoised) / 2
                else:
                    raise NotImplementedError(f"Don't know the conditioning you have wished for: {self.condition}")
                #print(score_conditioning.shape)
                if self.score_net.domain == "time":
                    x_hat = self.forward_score(score_conditioning_wav.unsqueeze(1)).squeeze(1)
                    sample = self.data_module.time_to_spec(x_hat)

                else:
                    sample = self.forward_score(score_conditioning)
                    x_hat = self.to_audio(sample.squeeze(), T_orig)

            if return_stft:
                return sample.squeeze(), Y.squeeze(), T_orig, norm_factor


        x_hat = x_hat * norm_factor
        y_denoised = y_denoised * norm_factor
        # new norm factor, for the volume of enhanced is too low
        # new_norm_factor = x_hat.abs().max().item()
        # x_hat = x_hat / new_norm_factor
        x_hat = x_hat.squeeze().cpu()
        y_denoised = y_denoised.squeeze().cpu()
        end = time.time()
        if timeit:
            sr = 16000
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, None, rtf
        else:
            return x_hat, y_denoised
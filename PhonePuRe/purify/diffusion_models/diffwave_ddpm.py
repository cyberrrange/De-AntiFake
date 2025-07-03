import argparse
import json

import torch
from .DiffWave_Unconditional.dataset import load_Qualcomm_keyword
from .DiffWave_Unconditional.WaveNet import WaveNet_Speech_Commands
from .DiffWave_Unconditional.util import calc_diffusion_hyperparams

import numpy as np
from typing import Union
import torchaudio
import csv

    
class DiffWave(torch.nn.Module):

    def __init__(self, 
                model: WaveNet_Speech_Commands, 
                diffusion_hyperparams: dict,
                reverse_timestep: int=200, 
                grad_enable=True,
                defense_method='None',
                ):
        super().__init__()

        '''
            model: input (x_t, t), output epsilon_theta at timestep t
        '''

        self.model = model
        self.diffusion_hyperparams = diffusion_hyperparams
        self.reverse_timestep = reverse_timestep
        self.freeze = False
        self.grad_enable = grad_enable
        self.defense_method = defense_method

    def forward(self, waveforms: Union[torch.Tensor, np.ndarray]):

        if isinstance(waveforms, np.ndarray): 
            waveforms = torch.from_numpy(waveforms)

        output = waveforms

        if self.defense_method == 'DualPure' or self.defense_method == 'DiffNoise':
            output = self._diffusion(waveforms)
        elif self.defense_method == 'AudioPure' or self.defense_method == 'DDPM' :
            output = self._diffusion(waveforms)
            output = self._reverse(output)
        elif self.defense_method == 'DiffRev':
            output = self._reverse(output)
        elif self.defense_method == 'OneShot':
            output = self._diffusion(waveforms)
            output = self.one_shot_denoise(output)
        
        return output

    def _diffusion(self, x_0: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 
        
        '''convert np.array to torch.tensor'''
        if isinstance(x_0, np.ndarray): 
            x_0 = torch.from_numpy(x_0)

        T, Alpha, Alpha_bar, Sigma = self.diffusion_hyperparams["T"], \
                                    self.diffusion_hyperparams["Alpha"], \
                                    self.diffusion_hyperparams["Alpha_bar"], \
                                    self.diffusion_hyperparams["Sigma"] 
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert x_0.ndim == 3

        '''noising'''
        z = torch.normal(0, 1, size=x_0.shape).cuda()
        x_t = torch.sqrt(Alpha_bar[self.reverse_timestep-1]).cuda() * x_0 + torch.sqrt(1-Alpha_bar[self.reverse_timestep-1]).cuda() * z
        return x_t

    def _reverse(self, x_t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 

        '''convert np.array to torch.tensor'''
        if isinstance(x_t, np.ndarray): 
            x_t = torch.from_numpy(x_t)

        T, Alpha, Alpha_bar, Sigma = self.diffusion_hyperparams["T"], \
                                    self.diffusion_hyperparams["Alpha"], \
                                    self.diffusion_hyperparams["Alpha_bar"], \
                                    self.diffusion_hyperparams["Sigma"] 
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T
        assert x_t.ndim == 3

        '''denoising'''
        x_t_rev = x_t.clone()
        for t in range(self.reverse_timestep-1, -1, -1):

            epsilon_theta_t, mu_theta_t, sigma_thata_t = self.compute_coefficients(x_t_rev, t)

            if t > 0:
                x_t_rev = mu_theta_t + sigma_thata_t * torch.normal(0, 1, size=x_t_rev.shape).cuda()
            else:
                x_t_rev = mu_theta_t
        return x_t_rev
    
    def fast_reverse(self, x_t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor: 

        '''convert np.array to torch.tensor'''
        if isinstance(x_t, np.ndarray): 
            x_t = torch.from_numpy(x_t)

        T, Alpha, Alpha_bar, Sigma = self.diffusion_hyperparams["T"], \
                                    self.diffusion_hyperparams["Alpha"], \
                                    self.diffusion_hyperparams["Alpha_bar"], \
                                    self.diffusion_hyperparams["Sigma"]

        K = 3
        S = torch.linspace(1, self.reverse_timestep, K)
        S = torch.round(S).int() - 1
        Beta_new, Beta_tilde_new = torch.zeros(size=(K,)), torch.zeros(size=(K,))

        for i in range(K):
            if i > 0:
                Beta_new[i] =  1 - Alpha_bar[S[i]] / Alpha_bar[S[i-1]]
                Beta_tilde_new[i] = (1 - Alpha_bar[S[i-1]]) / (1 - Alpha_bar[S[i]]) * Beta_new[i]
            else:
                Beta_new[i] =  1 - Alpha_bar[S[i]]
                Beta_tilde_new[i] = 0
        Alpha_new = 1 - Beta_new
        Alpha_bar_new = torch.cumprod(Alpha_new, dim=0)

        x_St = x_t
        for t in range(K-1, -1, -1):

            real_t = S[t]
            eps_St = self.model((x_St, real_t * torch.ones((x_St.shape[0], 1)).cuda()))
            mu_St = (x_St - (1 - Alpha_new[t]) / torch.sqrt(1 - Alpha_bar_new[t]) * eps_St) / torch.sqrt(Alpha_new[t])
            sigma_St = Beta_tilde_new[t]
            x_St = mu_St + sigma_St * torch.normal(0, 1, size=x_St.shape).cuda()

        return x_St

    def compute_coefficients(self, x_t: Union[torch.Tensor, np.ndarray], t: int):

        '''
            a single reverse step
            compute coefficients at timestep t+1
            t: in [0, T-1]
            return: eps_theta(x_t+1, t+1), mu_theta(x_t+1, t+1) and sigma_theta(x_t+1, t+1)
        '''

        Alpha, Alpha_bar, Sigma = self.diffusion_hyperparams["Alpha"], \
                                self.diffusion_hyperparams["Alpha_bar"], \
                                self.diffusion_hyperparams["Sigma"] 


        diffusion_steps = t * torch.ones((x_t.shape[0], 1)).cuda()
        epsilon_theta = self.model((x_t, diffusion_steps))
        mu_theta = (x_t - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
        sigma_theta = Sigma[t]

        # sigma_theta = self.diffusion_hyperparams["Beta"][t].sqrt()
        
        return epsilon_theta, mu_theta, sigma_theta

    @torch.no_grad()
    def compute_eps_t(self, x_t: Union[torch.Tensor, np.ndarray], t):

        diffusion_steps = t * torch.ones((x_t.shape[0], 1)).cuda()
        epsilon_theta = self.model((x_t, diffusion_steps))

        return epsilon_theta

    def one_shot_denoise(self, x_t: Union[torch.Tensor, np.ndarray]):

        t = self.reverse_timestep - 1
        diffusion_steps = t * torch.ones((x_t.shape[0], 1)).cuda()
        epsilon_theta = self.model((x_t, diffusion_steps))

        pred_x_0 = self._predict_x0_from_eps(x_t, t, epsilon_theta)

        return pred_x_0
    
    def two_shot_denoise(self, x_t: Union[torch.Tensor, np.ndarray]):

        t = self.reverse_timestep - 1
        diffusion_steps = t * torch.ones((x_t.shape[0], 1)).cuda()
        epsilon_theta = self.model((x_t, diffusion_steps))

        pred_x_1 = self._predict_x1_from_eps(x_t, t, epsilon_theta)
        pred_x_0 = self._predict_x0_from_x1(pred_x_1)

        return pred_x_0

    def _predict_x0_from_eps(self, x_t, t, eps):

        assert x_t.shape == eps.shape

        Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]

        sqrt_recip_alphas_bar = (1 / Alpha_bar).sqrt()
        sqrt_recipm1_alphas_bar = (1 / Alpha_bar - 1).sqrt()
        pred_x_0 = self._extract_into_tensor(sqrt_recip_alphas_bar, t, x_t.shape) * x_t - self._extract_into_tensor(sqrt_recipm1_alphas_bar, t, x_t.shape) * eps

        return pred_x_0

    def _predict_x1_from_eps(self, x_t, t, eps):

        Alpha = self.diffusion_hyperparams["Alpha"]
        Alpha_bar = self.diffusion_hyperparams["Alpha_bar"]
        Beta = self.diffusion_hyperparams["Beta"]

        mu = (Alpha_bar[t] / Alpha[0]).sqrt()
        sigma = (1 - Alpha_bar[t] - (Alpha_bar[t] / Alpha[0]) * Beta[0] ** 2).sqrt()

        pred_x_1 = (x_t - sigma * eps) / mu

        return pred_x_1
    
    def _predict_x0_from_x1(self, x_1):

        _, mu_0, _ = self.compute_coefficients(x_1, 0)

        pred_x_0 = mu_0

        return pred_x_0

    def _extract_into_tensor(self, arr_or_func, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        :param arr: the 1-D numpy array or a func.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if callable(arr_or_func):
            res = arr_or_func(timesteps).float()
        else:
            if isinstance(arr_or_func, torch.Tensor):
                res = arr_or_func.cuda()[timesteps].float()
            elif isinstance(arr_or_func, np.ndarray):
                res = torch.from_numpy(arr_or_func).cuda()[timesteps].float()
            else:
                raise TypeError('Unsupported data type {} in arr_or_func'.format(type(arr_or_func)))
        
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

def create_diffwave_model(args, model_path, config_path, reverse_timestep=25):

    with open(config_path) as f:
        data = f.read()
    cfg = json.loads(data)

    wavenet_config = cfg["wavenet_config"]      # to define wavenet
    diffusion_config = cfg["diffusion_config"]    # basic hyperparameters
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    WaveNet_model = WaveNet_Speech_Commands(**wavenet_config).cuda()
    checkpoint = torch.load(model_path)
    WaveNet_model.load_state_dict(checkpoint['model_state_dict'])

    Denoiser = DiffWave(model=WaveNet_model, diffusion_hyperparams=diffusion_hyperparams, reverse_timestep=reverse_timestep, defense_method=args.defense)

    return Denoiser

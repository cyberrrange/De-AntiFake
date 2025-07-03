import abc

import torch
import numpy as np

from sgmse.util.registry import Registry
import math

PredictorRegistry = Registry("Predictor")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")

def get_nearest_indices(timesteps, t):
    # Ensure t is float and on the same device as timesteps
    t = t.float()
    # timesteps: 1D tensor sorted in decreasing order
    # t: tensor of any shape
    # sort timesteps in increasing order
    timesteps_sorted, _ = torch.sort(timesteps, descending=False)
    # Flatten t for searchsorted
    t_flat = t.view(-1)
    # use searchsorted with right=True
    indices = torch.searchsorted(timesteps_sorted, t_flat, right=True)
    # handle edge cases
    indices = torch.clamp(indices, 0, len(timesteps_sorted) - 1)
    # compute distances
    dist_right = torch.abs(t_flat - timesteps_sorted[indices])
    # for indices > 0, compute dist_left
    mask = indices > 0
    dist_left = torch.abs(t_flat - timesteps_sorted[indices - 1])
    # where dist_left < dist_right, set nearest_indices to indices - 1
    nearest_indices = indices.clone()
    nearest_indices[mask & (dist_left < dist_right)] -= 1
    # Reshape nearest_indices back to t's original shape
    nearest_indices = nearest_indices.view(t.shape)
    return nearest_indices.long()



@PredictorRegistry.register('euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args, **kwargs):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        f, g = self.rsde.sde(x, t, *args, **kwargs)
        x_mean = x + f * dt
        if g.ndim < x.ndim:
            g = g.view( *g.size(), *((1,)*(x.ndim - g.ndim)) )
        x = x_mean + g * np.sqrt(-dt) * z
        return x, x_mean



@PredictorRegistry.register('reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow=probability_flow)

    def update_fn(self, x, t, *args, **kwargs):
        f, g = self.rsde.discretize(x, t, *args, **kwargs)
        z = torch.randn_like(x)
        x_mean = x - f
        if g.ndim < x.ndim:
            g = g.view( *g.size(), *((1,)*(x.ndim - g.ndim)) )
        x = x_mean + g * z
        return x, x_mean


@PredictorRegistry.register('none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, t, *args, **kwargs):
        return x, x

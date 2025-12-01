
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    v = v.to(device)
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        


    def forward(self, x_0, labels):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='mean')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))


    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in tqdm(reversed(range(self.T))):
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   

class DDIMSampler(nn.Module):
    def __init__(
        self,
        model, beta_1, beta_T, T, w = 0.,
        sample_steps: int = 25,
    ):
        super().__init__()
        self.model = model
        self.num_train_timesteps = T
        self.betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        print(self.alphas_cumprod.shape)
        self.timesteps = torch.linspace(T - 1, 0, sample_steps).long()


    @torch.no_grad()
    def forward(
        self,
        x_T, labels,
        # unet: UNet2DModel,
        # batch_size: int,
        # in_channels: int,
        # sample_size: int,
        eta: float = 0.0,
    ):
        x_t = x_T
        alphas = self.alphas.to(x_T.device)
        alphas_cumprod = self.alphas_cumprod.to(x_T.device)
        timesteps = self.timesteps.to(x_T.device)
        # x_ = torch.randn((batch_size, in_channels, sample_size, sample_size), device=unet.device)
        for t, tau in tqdm(list(zip(timesteps[:-1], timesteps[1:])), desc='Sampling'):
            # print(t, tau)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * t
            # print(x_t.shape, t.shape, labels.shape)
            pred_noise = self.model(x_t, t, labels)

            # sigma_t
            if not math.isclose(eta, 0.0):
                one_minus_alpha_prod_tau = 1.0 - alphas_cumprod[tau]
                one_minus_alpha_prod_t = 1.0 - alphas_cumprod[t]
                one_minus_alpha_t = 1.0 - alphas[t]
                sigma_t = eta * (one_minus_alpha_prod_tau * one_minus_alpha_t / one_minus_alpha_prod_t) ** 0.5
            else:
                sigma_t = torch.zeros_like(alphas[0])

            # first term of x_tau
            alphas_cumprod_tau = alphas_cumprod[tau]
            sqrt_alphas_cumprod_tau = alphas_cumprod_tau ** 0.5
            alphas_cumprod_t = alphas_cumprod[t]
            sqrt_alphas_cumprod_t = alphas_cumprod_t ** 0.5
            sqrt_one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod_t) ** 0.5
            sqrt_one_minus_alphas_cumprod_t_expanded = sqrt_one_minus_alphas_cumprod_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            sqrt_alphas_cumprod_t_expanded = sqrt_alphas_cumprod_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            first_term = sqrt_alphas_cumprod_tau * (x_t - sqrt_one_minus_alphas_cumprod_t_expanded * pred_noise) / sqrt_alphas_cumprod_t_expanded

            # second term of x_tau
            coeff = (1.0 - alphas_cumprod_tau - sigma_t ** 2) ** 0.5
            second_term = coeff * pred_noise

            epsilon = torch.randn_like(x_t)
            x_t = first_term + second_term + sigma_t * epsilon
        # x_t = (x_t / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return x_t
    
import math
import torch
import numpy as np

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_alphas_and_betas(beta_schedule='linear'):
    ts_cnt = 1000   # timestep count
    beta_start, beta_end = 0.0001, 0.02
    if beta_schedule == "cosine":
        # cosine scheduler is from the following paper:
        # ICML. 2021. Alex Nichol. Improved Denoising Diffusion Probabilistic Models
        # In this option, it composes alphas_accum firstly, then alphas and betas.
        cos_0 = math.cos(0.008 / 1.008 * math.pi / 2) ** 2
        alphas_accum = []  # alpha accumulate product array
        for i in range(ts_cnt):
            t = i / ts_cnt
            ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            ac /= cos_0
            alphas_accum.append(ac)
        alphas_accum = torch.Tensor(alphas_accum).float()
        divisor = torch.cat([torch.ones(1), alphas_accum[:-1]], dim=0)
        alphas = torch.div(alphas_accum, divisor)
        betas = 1 - alphas
    else:
        betas = get_beta_schedule(beta_schedule, beta_start, beta_end, ts_cnt)
        betas = torch.from_numpy(betas).float()
        alphas = 1.0 - betas
        alphas_accum = alphas.cumprod(dim=0)
    return alphas, alphas_accum, betas

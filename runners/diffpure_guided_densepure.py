import os
import random

import torch
import torchvision.utils as tvu

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
import math
import numpy as np


class GuidedDiffusion(torch.nn.Module):
    def __init__(self, args, config, device=None, model_dir='pretrained'):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.reverse_state = None
        self.reverse_state_cuda = None

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(self.config.model))
        print(f'model_config: {model_config}')
        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load("/mnt/home/diffusion-ars/imagenet/256x256_diffusion_uncond.pt"))
        model.requires_grad_(False).eval().to(self.device)

        if model_config['use_fp16']:
            model.convert_to_fp16()

        self.model = model
        self.model.eval()
        self.diffusion = diffusion
        self.betas = diffusion.betas
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        sigma = self.args.sigma

        a = 1/(1+(sigma*2)**2)
        self.scale = a**0.5

        sigma = sigma*2
        T = self.args.t_total
        for t in range(len(self.sqrt_recipm1_alphas_cumprod)-1):
            if self.sqrt_recipm1_alphas_cumprod[t]<sigma and self.sqrt_recipm1_alphas_cumprod[t+1]>=sigma:
                if sigma - self.sqrt_recipm1_alphas_cumprod[t] > self.sqrt_recipm1_alphas_cumprod[t+1] - sigma:
                    self.t = t+1
                    break
                else:
                    self.t = t
                    break
            self.t = len(diffusion.alphas_cumprod)-1

        print(f"jump to step {self.t}")
        print(f"sigma is {sigma}")

    def image_editing_sample(self, img=None, bs_id=0, tag=None, sigma=0.0):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        with torch.no_grad():
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim
            x0 = img

            x0 = self.scale*(img)
            t = self.t

            model_kwargs={"img" :  img}

            if self.args.use_clustering:
                x0 = x0.unsqueeze(1).repeat(1,self.args.clustering_batch,1,1,1).view(batch_size*self.args.clustering_batch,3,256,256)
            self.model.eval()

            if self.args.use_one_step:
                # one step denoise
                t = torch.tensor([round(t)] * x0.shape[0], device=self.device)
                out = self.diffusion.p_sample(
                    self.model,
                    x0,
                    t+self.args.t_plus,
                    clip_denoised=True,
                )

                x0 = out["pred_xstart"]

            elif self.args.use_t_steps:
                #save random state
                if self.args.save_predictions:
                    global_seed_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        global_cuda_state = torch.cuda.random.get_rng_state_all()

                    if self.reverse_state==None:
                        torch.manual_seed(self.args.reverse_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(self.args.reverse_seed)
                    else:
                        torch.random.set_rng_state(self.reverse_state)
                        if torch.cuda.is_available():
                            torch.cuda.random.set_rng_state_all(self.reverse_state_cuda)

                # t steps denoise
                inter = t/self.args.num_t_steps
                indices_t_steps = [round(t-i*inter) for i in range(self.args.num_t_steps)]
                
                for i in range(len(indices_t_steps)):
                    t = torch.tensor([len(indices_t_steps)-i-1] * x0.shape[0], device=self.device)
                    real_t = torch.tensor([indices_t_steps[i]] * x0.shape[0], device=self.device)
                    # print(f" at i={i}, t is {t[0].item()}, real_t is {real_t[0].item()}, step is {len(indices_t_steps)-i}")

                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            x0,
                            t,
                            clip_denoised=True,
                            cond_fn = self.cond_fn,
                            model_kwargs = model_kwargs,
                            indices_t_steps = indices_t_steps.copy(),
                            T = self.args.t_total,
                            step = len(indices_t_steps)-i,
                            real_t = real_t
                        )
                        x0 = out["sample"]

                #load random state
                if self.args.save_predictions:
                    self.reverse_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        self.reverse_state_cuda = torch.cuda.random.get_rng_state_all()

                    torch.random.set_rng_state(global_seed_state)
                    if torch.cuda.is_available():
                        torch.cuda.random.set_rng_state_all(global_cuda_state)

            else:
                #save random state
                if self.args.save_predictions:
                    global_seed_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        global_cuda_state = torch.cuda.random.get_rng_state_all()

                    if self.reverse_state==None:
                        torch.manual_seed(self.args.reverse_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(self.args.reverse_seed)
                    else:
                        torch.random.set_rng_state(self.reverse_state)
                        if torch.cuda.is_available():
                            torch.cuda.random.set_rng_state_all(self.reverse_state_cuda)

                # full steps denoise
                indices = list(range(round(t)))[::-1]
                for i in indices:
                    t = torch.tensor([i] * x0.shape[0], device=self.device)
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            x0,
                            t,
                            clip_denoised=True,
                        )
                        x0 = out["sample"]

                #load random state
                if self.args.save_predictions:
                    self.reverse_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        self.reverse_state_cuda = torch.cuda.random.get_rng_state_all()

                    torch.random.set_rng_state(global_seed_state)
                    if torch.cuda.is_available():
                        torch.cuda.random.set_rng_state_all(global_cuda_state)

            return x0
    
    def cond_fn(self, x, t, **kwargs):
        # scale = 2 * torch.ones(10).cuda()
        scale = 0
        # print(f"scale is {scale}")
        var = kwargs["var"]
        sqrt_alpha = kwargs["sqrt_alpha"]
        sqrt_alpha_t_minus_one = kwargs["sqrt_alpha_t_minus_one"]
        mean_t_minus_one = kwargs["mu_t"]

        rescaled_original_img = kwargs["img"]
        # print(f"x is {x.min():.3f}, {x.max():.3f}")
        # print(f"x shape is {x.shape}")
        # print(f"img is {rescaled_original_img.min()}, {rescaled_original_img.max()}")
        # print(f"img shape is {rescaled_original_img.shape}")

        guide = sqrt_alpha_t_minus_one * rescaled_original_img - mean_t_minus_one
        guide = guide  * scale if t[0]!= 0 else torch.zeros_like(x)

        # print(t[0].item())
        # breakpoint()
        # print(f"variance is {var.min().item():.3f}, {var.max().item():.3f}")
        # print(f"guide value is {guide.min().item():.3f}, {guide.max().item():.3f}\n")
        return guide

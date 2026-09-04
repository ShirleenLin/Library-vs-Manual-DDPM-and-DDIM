import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def Get_schedule_value(a, t, x_shape): #Applied to ᾱₜ, √ᾱₜ, √(1 − ᾱₜ), 1/√αₜ, βₜ/√(1 − ᾱₜ), √βₜ, √β̃ₜ, 1/√ᾱₜ, √(1/ᾱₜ − 1)
    b, *_ = t.shape
    out = a.gather(-1, t)  # a[t] # Look up the schedule value at t  VS Rectified Flow doesn't have schedule lookup.
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # t:(b,) to αₜ: (b, 1, 1, 1)


class Diffusion(nn.Module):
    def __init__(self, model, img_size, img_channels, betas, alphas_bar=None, loss_type="l2", dtype=torch.float32):
        super().__init__()
        self.model = model
        self.step = 0
        self.num_timesteps = len(betas)
        self.x_size = img_size
        self.x_channels = img_channels
        loss_map = {"l1": F.l1_loss,"l2": F.mse_loss,"huber": F.huber_loss, "smooth_l1": F.smooth_l1_loss}
        self.loss_fn = loss_map.get(loss_type, F.mse_loss)
        if loss_type not in loss_map:
            print(f"Loss type {loss_type} not found - Defaulting to MSE. Valid options are: {list(loss_map.keys())}")

        #if to_float32 --> calculate constants in 32-bit
        to_dtype = partial(torch.tensor, dtype=dtype)
        #Beta
        self.register_buffer("betas", to_dtype(betas))
        self.register_buffer("sigma", to_dtype(np.sqrt(betas)))
        #Alpha
        alphas = 1 - betas
        self.register_buffer("alphas", to_dtype(alphas))
        #Alpha bar
        if alphas_bar is not None:
            self.register_buffer("alphas_bar", to_dtype(alphas_bar))
        else:
            self.register_buffer("alphas_bar", to_dtype(np.cumprod(alphas)))
        #Beta tilde
        beta_bar = betas * (1 - np.concatenate(([1.0], alphas_bar[:-1]))) / (1 - alphas_bar)
        self.register_buffer("sigma_tilde", to_dtype(np.sqrt(beta_bar)))  #DDPM: σₜ choice; DDIM: Add η·√β~ₜ·z
        #DDPM & DDIM Forward Process (Alpha bar)
        self.register_buffer("sqrt_alphas_bar", to_dtype(np.sqrt(alphas_bar))) #xₜ = √αˉₜ·x₀ + √(1-αˉₜ)·ϵ
        self.register_buffer("sqrt_1m_alphas_bar", to_dtype(np.sqrt(1 - alphas_bar)))
        #DDPM Sampling coefficients  (Alpha and Alpha bar)
        self.register_buffer("rsqrt_alphas", to_dtype(np.sqrt(1/alphas))) #xₜ₋₁ = 1/√αₜ(xₜ - βₜ/√(1-αˉₜ)·ϵ) + σₜ·z
        self.register_buffer("ddpm_eps_coef", to_dtype(betas / np.sqrt(1 - alphas_bar)))
        #DDIM Sampling coefficients (Alpha bar)
        self.register_buffer('rsqrt_alphas_bar', to_dtype(np.sqrt(1/np.maximum(alphas_bar, 1e-12)))) #x₀ = 1/√αˉₜ·xₜ - √(1/αˉₜ-1)·ϵ
        self.register_buffer('ddim_eps_coef', to_dtype(np.sqrt(1/alphas_bar - 1)))


    def generate_or_reconstruct(self, x, batch_size, device, strength=0.5, num_inference_steps=50, scheduler=None,generator=None, times=None):

        if times=='diffusers' and scheduler is not None: #Initially imported Diffusers timestep results for comparison -> later use downloaded source and print inside to trace execution
          times = scheduler.set_timesteps(num_inference_steps, device=device)
          scheduler.config.timestep_spacing == "leading"  #default
          times = scheduler.timesteps.clone()
        else: #times=='linspace':
          times = torch.linspace(self.num_timesteps-1, 0, steps = num_inference_steps).long().tolist()

        if x is None: # Generate a new picture
            x = torch.randn(batch_size, self.x_channels, *self.x_size, device=device, generator=generator)
        else: # Reconstruct a local picture
            x = x.to(device)
            start_idx = num_inference_steps - int(num_inference_steps*strength)
            t_batch = torch.full((batch_size,), times[start_idx], device=device, dtype=torch.long)
            noise = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype)
            x = self.x0_to_xt_add_noise(x, t_batch, noise = noise)
            times = times[start_idx:]
        return x, times

    def predict_noise_with_guidance(self, x, t, cond=None, uncond=None, guidance_scale=1.0):
        if cond is None:
            return self.model(x, t, None)
        if uncond is None or guidance_scale == 1.0:
            return self.model(x, t, cond)
        eps_cond = self.model(x, t, cond)
        eps_uncond = self.model(x, t, uncond)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


    @torch.no_grad()  #No graph stored for .backward() for gradient computing → saves memory
    def Reverse_DDPM(self, batch_size, device, x=None, strength=0.5, num_inference_steps=50, scheduler=None, generator=None, tilde=False, cond=None, uncond=None, guidance_scale=1.0):
        x, times = self.generate_or_reconstruct(x, batch_size, device, strength, num_inference_steps, scheduler=scheduler, generator=generator)
        sigma = self.sigma_tilde if tilde==True else self.sigma # or self.sigma_tilde
        for t in times[:-1]:  # t==999 handles x1000 → x999
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            #Denoise
            pred_noise = self.predict_noise_with_guidance(x, t_batch, cond, uncond, guidance_scale)
            denoise_term = Get_schedule_value(self.ddpm_eps_coef, t_batch, x.shape) * pred_noise # βₜ/√(1-αˉₜ)
            scale = Get_schedule_value(self.rsqrt_alphas, t_batch, x.shape) # 1/√αₜ
            x = scale * (x - denoise_term) #(1/√αₜ)·(xₜ​ - (βₜ/√(1-αˉₜ))·ϵθ​)
            if t > 0: #Add noise | t==0 handles x1 → x0, no added noise
                noise_term = Get_schedule_value(sigma, t_batch, x.shape) * torch.randn_like(x, generator=generator)
                x += noise_term # σₜ·z

            #Debug#########
            val_denoise = denoise_term.abs().mean().item()
            val_scaled = (scale * denoise_term).abs().mean().item()
            val_noise = noise_term.abs().mean().item()
            print(f"Step {t} | Denoise: {val_denoise:.4f} | Scale*Denoise: {val_scaled:.4f} | noise_term +: {val_noise :.4f}")
            #Debug#########


            # --- DDPM CHECK: Mean & Max of Latent ---  F16 can only represent numbers up to 65,504
            avg, mx = x.abs().mean().item(), x.max().item()
            print(f"DDPM Inference: {num_inference_steps}; Step {t} | ABS Mean: {avg:.4f} | Max: {mx:.4f}")
            if torch.isnan(x).any():
                print(f"!!! STEP {t}: NAN DETECTED (Math Exploded) !!!")
        return x.cpu().detach()


    @torch.no_grad()
    def Reverse_DDIM(self, batch_size, device, x=None, strength=0.5, num_inference_steps=50, scheduler=None, generator=None, times=None, eta = 0, cond=None, uncond=None, guidance_scale=1.0):
        x, times = self.generate_or_reconstruct(x, batch_size, device, strength, num_inference_steps, generator=generator)
        for t, t_next in zip(times[:-1], times[1:]):
            t_batch = torch.full((batch_size,), t, device = device, dtype = torch.long)
            pred_noise, x0 = self.xt_to_x0_pred_noise(x,t_batch, cond=cond, uncond=uncond, guidance_scale=guidance_scale, clip_x0=False, rederive_pred_noise=False) #DDIM paper clip_x0 = False


            alpha_bar = Get_schedule_value(self.alphas_bar, t_batch, x.shape)
            alpha_bar_next = Get_schedule_value(self.alphas_bar, torch.full_like(t_batch, t_next), x.shape)

            ddim_sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_next))
            #xₜ₋ₙ = √α-ₜ₋₁ · x₀ +√(1 - α-ₜ₋₁ - σₜ ^2)·ϵ + σₜ·z
            c = (1 - alpha_next - ddim_sigma ** 2).clamp(min=0) # making sure >= 0
            new_noise = torch.randn_like(x, generator=generator) if eta > 0 else torch.zeros_like(x)
            x = alpha_next.sqrt() * x0  + torch.sqrt(c) * pred_noise + ddim_sigma * new_noise

            # --- DDIM Checks Mean of x0 and Latent ---
            print(f"Step {t}->{t_next} | x0 Mean: {x0.abs().mean().item():.4f} | Latent Mean: {x.abs().mean().item():.4f}")
            if x.abs().mean() < 1e-5:
                print("!!! WARNING: Signal vanishing (Black Image likely) !!!")
        return x.cpu().detach()


    def x0_to_xt_add_noise(self, x, t, noise):  #xₜ = √αˉₜ·x₀ + √(1-αˉₜ)·ϵ
        return Get_schedule_value(self.sqrt_alphas_bar, t, x.shape) * x + Get_schedule_value(self.sqrt_1m_alphas_bar, t, x.shape) * noise

    def xt_to_x0_pred_noise(self, x, t,  clip_x0 = False, rederive_pred_noise = False, cond=None, uncond=None, guidance_scale=1.0):
        pred_noise = self.predict_noise_with_guidance(x, t, cond, uncond, guidance_scale)
        #x₀ = 1/√αˉₜ·xₜ - √(1/αˉₜ-1)·ϵ
        x0 = Get_schedule_value(self.rsqrt_alphas_bar,t,x.shape)*x - Get_schedule_value(self.ddim_eps_coef,t,x.shape)*pred_noise
        if clip_x0: x0 = x0.clamp(-1., 1.)  # Same with torch.clamp(x0, -1., 1.) - x0 is tensor
        if clip_x0 and rederive_pred_noise:
          #x₀ = 1/√αˉₜ·xₜ - √(1/αˉₜ-1)·ϵ  --> ϵ = (1/√αˉₜ·xₜ - x₀)/√(1/αˉₜ-1), which only differs if x₀ is clipped
          pred_noise = (Get_schedule_value(self.rsqrt_alphas_bar,t,x.shape) * x - x0) /Get_schedule_value(self.ddim_eps_coef, t, x.shape)
        return pred_noise, x0


    def compute_noise_loss(self, x, t, cond=None):
        true_noise = torch.randn_like(x)
        xt = self.x0_to_xt_add_noise(x, t, true_noise)
        pred_noise = self.model(xt, t, cond) # If using a manual NN without a 'cond' parameter, call: self.model(xt, t)
        return self.loss_fn(pred_noise, true_noise)

    def forward(self, x, cond=None):  #input image as x
        b, c, h, w = x.shape
        if h != self.x_size[0] or w != self.x_size[1]:
            raise ValueError(f"Size error: input size {(h, w)} vs expected {(self.x_size[0], self.x_size[1])}")
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device) #torch.randint(low, high) doesn't include high value
        return self.compute_noise_loss(x, t, cond)


# ##################### Diffuers ###########################


#     def manual_vs_Mirror(self, scheduler):
#       device = self.alphas_bar.device
#       def report_diff(name, manual_tensor, library_tensor, threshold=1e-12):
#           diffs = torch.abs(manual_tensor - library_tensor)
#           mismatches = torch.sum(diffs > threshold).item()
#           print(f"Total Steps: {len(manual_tensor)} | {name.upper()} Mismatches (>{threshold}): {mismatches}")
#           if mismatches > 0:
#               print(f"Max Diff: {diffs.max().item():.2e} | Mean Diff: {diffs.mean().item():.2e}")
#               print(f"Manual {name} Range: {manual_tensor[0].item():.6f} to {manual_tensor[-1].item():.6f}")
#               print(f"Library {name} Range: {library_tensor[0].item():.6f} to {library_tensor[-1].item():.6f}")
#           else:
#               print(f"PERFECT MATCH: All {name} values are identical.")

#           #Alpha Comparison
#           sched_alphas_bar = scheduler.alphas_cumprod.to(device)
#           report_diff("Alphas_bar Comparison", self.alphas_bar, sched_alphas_bar)
#           #Sigma Comparison
#           lib_vars = torch.tensor([scheduler._get_variance(i) for i in range(len(scheduler))], device=device)
#           lib_sigmas = lib_vars.sqrt()
#           report_diff("Sigma_tilde Comparison", self.sigma_tilde, lib_sigmas)
#           #Config Check
#           print(f"Library clip_sample: {scheduler.config.clip_sample}; Offset: {scheduler.config.steps_offset}; Spacing: {scheduler.config.timestep_spacing}")
#           print(f"product_type:{scheduler.config.prediction_type}; beta_schedule: {scheduler.config.beta_schedule}; variance_type:{scheduler.config.variance_type}")



# ###### Outside the Class ####################

#     def cosine_schedule(T, s=0.008):
#         t = np.arange(T + 1)
#         alphas_bar = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
#         alphas_bar = alphas_bar / alphas_bar[0]
#         betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
#         return np.clip(betas, a_min=0, a_max=0.999)  #In the end, only use beta

#     def linear_schedule(T, low, high):
#         return np.linspace(low, high, T)   #total T points



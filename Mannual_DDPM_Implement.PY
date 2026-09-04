import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from functools import partial
from Manual_Pytorch_DDPM import Diffusion  #Manual_Pytorch_DDPM.py
from diffusers import StableDiffusionPipeline  #Prompt → Text Tokenizer → Text Encoder → UNet → Scheduler (Manual_Pytorch_DDPM) → VAE → Image
# anime.png → VAE Encoder → x₀ → Add Noise: xₜ; Prompt: "masterpiece, high quality" → Tokenizer: ["masterpiece", ",", "high", "quality"] → Token IDs: [...] → Text Encoder → Text Embeddings: [[...]] → UNet(xₜ, t, embeddings) → Manual DDPM/DDIM: t=500→499→...→0 (UNet runs at each t) → VAE Decoder → Output Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Meina/MeinaMix_V11"
dtype = torch.float32
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)


#1) Get Unet from MeinaMix
class MeinaUNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, x, t, cond):
        return self.unet(x, t, encoder_hidden_states=cond, return_dict=False)[0]

#2) Get VAE from MeinaMix
def image_to_latents(path, pipe):  #pipe.vae is a CNN (Autoencoder).
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, (512, 512), Image.Resampling.LANCZOS)
    img_tensor = T.ToTensor()(img).unsqueeze(0).to(device, dtype=dtype)
    img_tensor = 2.0 * img_tensor - 1.0 # Scale to [-1, 1]
    with torch.no_grad():
        latents = pipe.vae.encode(img_tensor).latent_dist.sample()  # Trained VAE Encoder from MeinaMix  -> Mean & log sigma^2 -> sample
        return latents * pipe.vae.config.scaling_factor

#3) Get tokenizer and text_encoder from MeinaMix
def get_embeds(prompt):  #pipe.text_encoder is a Neural Network (Transformer).
    tokens = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
    return pipe.text_encoder(tokens)[0]


manual = Diffusion(model=MeinaUNet(pipe.unet), img_size=(64, 64), img_channels=4, betas=pipe.scheduler.betas.cpu().numpy(), alphas_bar = pipe.scheduler.alphas_cumprod.cpu().numpy(),dtype=dtype).to(device)



# TWO: Comparison Implementation
init_x = image_to_latents("anime_selfie.png", pipe)

shared_kwargs = {"batch_size": 1,
    "device": device,
    "x": init_x.clone(),
    "cond": get_embeds("masterpiece, high quality"), #masterpiece, high quality, anime style, girl, wearing headphones, consistent face
    "uncond": get_embeds("deformed, low quality, blurry"), #deformed, low quality
    "guidance_scale": 6.0,
    "strength": 0.45,
    "generator":torch.Generator(device=device).manual_seed(42)
    }


samplers = {
    "DDPM_seed42_pic1": partial(manual.Reverse_DDPM, **shared_kwargs, ),
    "DDPM_tilde_seed42_pic1": partial(manual.Reverse_DDPM, **shared_kwargs, tilde=True),
    "DDIM_eta0.5_seed42_pic1": partial(manual.Reverse_DDIM, **shared_kwargs, eta=0.5),
    "DDIM_no_eta_seed42_pic1": partial(manual.Reverse_DDIM, **shared_kwargs, eta=0)
    }


for name, sampler in samplers.items():
    for steps in [50, 100, 500, 1000]:

        print(f"RUNNING: {name} | INFERENCE STEPS: {steps}")

        with torch.autocast(device_type="cuda", dtype=dtype):
            output_latents = sampler(num_inference_steps=steps)  #for mirroring...

        with torch.no_grad():
            latents = output_latents.to(device) / pipe.vae.config.scaling_factor
            image = pipe.vae.decode(latents.float()).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = (image * 255).permute(0, 2, 3, 1).cpu().numpy().astype("uint8")

            filename = f"Manual_{name}_{steps}.png"
            Image.fromarray(image[0]).save(filename)
            print(f"Saved: {filename}")




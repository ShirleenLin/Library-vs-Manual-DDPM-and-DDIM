

import torch
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, DDPMScheduler
from PIL import Image, ImageOps

def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (512, 512), Image.Resampling.LANCZOS)
    return img

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("Meina/MeinaMix_V11", torch_dtype=dtype).to(device)
img = prepare_image('anime_selfie.png')

schedulers = {
    "NO_SAFETY_library_eta0.5_DDIM_seed42": DDIMScheduler.from_config(pipe.scheduler.config),
    "NO_SAFETY_library_eta0.5_DDPM_seed42": DDPMScheduler.from_config(pipe.scheduler.config)
    }

for t in [50, 100, 500, 1000]:
    for name, sched in schedulers.items():
        print(f"Testing {name} at {t} steps.../n product_type:{pipe.scheduler.config.prediction_type} /n beta_schedule: {pipe.scheduler.config.beta_schedule}")
        pipe.scheduler = sched
        pipe.safety_checker = None
        reconstructed_image = pipe(
            prompt="masterpiece, high quality",
            negative_prompt="deformed, low quality, blurry",
            image=img,
            num_inference_steps=t,
            strength=0.45,
            guidance_scale=6.0,
            eta=0.5,
            generator=torch.Generator(device=device).manual_seed(42)
        ).images[0]
        reconstructed_image.save(f"{name}_{t}.png")
print("Tests complete.")

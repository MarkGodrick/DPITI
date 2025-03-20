from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import os

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, device_map="balanced")

kwargs = {"num_inference_steps":4,"guidance_scale":0.5}

PATH = os.path.join("tmp",*[f"{key}={value}" for key, value in kwargs.items()])

images = pipe(prompt="A description of bedroom",width=512,height=512,num_images_per_prompt=10,**kwargs).images

os.makedirs(PATH,exist_ok=True)

for idx,img in enumerate(images):
    img.save(os.path.join(PATH,f"{idx:04}.png"))
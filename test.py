from diffusers import StableDiffusionXLPipeline
import pandas as pd
import numpy as np

np.random.seed(42)
file_path = "results/camelyon17/openai/gpt-4o-mini/pe/meta-llama/epsilon=1.0/iteration=10/sdxl-turbo/trial1/synthetic_text/000000009.csv"
num_samples = 30

file = pd.read_csv(file_path)
pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo").to("cuda")

indices = np.random.choice(len(file),num_samples,replace=False)

for idx in indices:
    images = pipeline(file["text"][idx], num_inference_steps=4, guidance_scale=0.0).images
    img = images[0]
    img.save(f"images/Camelyon17/{idx:04d}.png")


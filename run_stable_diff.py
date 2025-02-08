import os
import numpy as np
import torch
import pandas as pd
import argparse
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline
from tqdm import tqdm



def main(args):
    # read the latest PE results
    filenames = os.listdir(os.path.join(args.input,"synthetic_text"))
    files = [int(filename.split(".")[0]) for filename in filenames]
    target_file = filenames[np.argmax(files)]

    df = pd.read_csv(os.path.join(args.input,"synthetic_text",target_file))
    text_data = list(df['text'])

    # generate images
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    
    images = []

    for text in tqdm(text_data):
        if not isinstance(text,str):
            text = str(text)
        elif not text:
            text = ""
        image = pipe(text, height=512, width=512).images[0]
        images.append(np.array(image))

    images = np.array(images)
    np.savez(os.path.join(args.output,"images_0"),images)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input",type=str,default="results/text/LSUN_huggingface/part_0")
    parser.add_argument("--output",type=str,default="results/image/LSUN_huggingface/part_0")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    main(args)
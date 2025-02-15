import os
import numpy as np
import torch
import pandas as pd
import argparse
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline
from tqdm import tqdm

batch_size = 1
MAX_NUM = 40000

def main(args):
    # read the latest PE results
    # filenames = os.listdir(os.path.join(args.input,"synthetic_text"))
    # files = [int(filename.split(".")[0]) for filename in filenames]
    # target_file = filenames[np.argmax(files)]

    df = pd.read_csv(os.path.join(args.input)).astype(str)
    
    # idx = 3
    # text_data = list(df['text'])[idx*3500:(idx+1)*3500]

    # partition = 8
    # idx = 7
    # text_data = list(df['text'])[idx:MAX_NUM:partition]

    text_data = list(df['text'])


    # generate images
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    
    images = []
    batch_num = (len(text_data)+batch_size-1)//batch_size

    for batch_idx in tqdm(range(batch_num)):
        images.extend(pipe(text_data[batch_idx*batch_size:(batch_idx+1)*batch_size]).images)

    images = np.array(images)
    # np.savez(os.path.join(args.output,f"images_{idx}"),images)
    np.savez(os.path.join(args.output,f"caption10240_images_0"),images)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input",type=str,default="lsun/bedroom_train/Salesforce/blip-image-captioning-large/caption_0.csv")
    parser.add_argument("--output",type=str,default="results/image/LSUN_huggingface/original")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    main(args)
import os
import re
import sys
import numpy as np
import torch
import pandas as pd
import argparse
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionPipeline
from tqdm import tqdm
from logger import execution_logger, setup_logging
from pe.data import Data
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

batch_size = 16
MAX_NUM = 40000
MIN_LEN = 100
np.random.seed(42)


def main(args):
    
    execution_logger.info("Loading Caption data...")

    pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'

    # df = pd.read_csv(os.path.join(args.input)).astype(str)

    # text_data = list(df['text'])

    # text_data = [text.strip() for text in text_data if len(text)>MIN_LEN]

    idx = 0
    data = Data()
    data.load_checkpoint(args.input)
    text_data = data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME:idx})
    text_data = list(text_data.data_frame["PE.TEXT"])
    text_data = [str(item) for item in text_data]

    if args.filter:
        matches = [re.search(pattern,text,re.DOTALL) for text in text_data]

        text_list = [match.group(2).strip() for match in matches]
    else:
        text_list = text_data

    if args.num_samples is not None and args.num_samples<len(text_list):
        indices = np.random.choice(len(text_list),args.num_samples,replace=False)
        text_list = [text_list[idx] for idx in indices]

    execution_logger.info("Loading success. Now loading diffusion model...")

    # generate images
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    
    execution_logger.info("Loading success. Start sampling...")

    images = []
    batch_num = (len(text_data)+batch_size-1)//batch_size

    for batch_idx in tqdm(range(batch_num)):
        # input: a list of text string
        # output: a list of PIL.Image.Image, each dtype=np.uint8, shape=(1024,1024,3)
        images.extend(pipe(text_list[batch_idx*batch_size:(batch_idx+1)*batch_size],num_inference_steps=4,guidance_scale=0.0).images)

    execution_logger.info("Sampling process accomplished. Saving data...")

    images = np.array(images)

    # file_name = args.input.split('/')[-1]
    np.savez(args.output,images)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input",type=str,default="lsun/bedroom_train/Salesforce/blip-image-captioning-large/caption_0.csv")
    parser.add_argument("--output",type=str,default="results/image/LSUN_huggingface/original")
    parser.add_argument("--model",type=str,default="stabilityai/sdxl-turbo")
    parser.add_argument("--num_samples",type=int,default=None)
    parser.add_argument("--filter",type=bool,default=True)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    
    setup_logging(log_file=os.path.join(os.path.dirname(args.output),"log_step4.txt"))
    execution_logger.info("\nExecuting {}...\ninput: {}\noutput: {}\nmodel: {}".format(sys.argv[0],args.input,args.output,args.model))

    main(args)
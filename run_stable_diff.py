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

batch_size = 4
MAX_NUM = 40000
MIN_LEN = 100
np.random.seed(42)


def main(args):
    
    execution_logger.info("Loading Caption data...")

    pattern = r'([^:]*:|^)(.*?)(?=\.$|$)'

    # df = pd.read_csv(os.path.join(args.input)).astype(str)

    # text_data = list(df['text'])

    # text_data = [text.strip() for text in text_data if len(text)>MIN_LEN]

    idx = -1
    data = Data()
    data.load_checkpoint(args.input)
    data = data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME:idx})
    execution_logger.info(f"filtered data length is: {len(data.data_frame)}")
    text_data = list(data.data_frame["PE.TEXT"])
    labels = list(data.data_frame["PE.LABEL_ID"])
    text_data = [str(item) for item in text_data]

    if args.filter:
        filtered_texts = []
        filtered_labels = []

        for text, label in zip(text_data, labels):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                core_text = match.group(2).strip()
                if len(core_text) > MIN_LEN:
                    filtered_texts.append(core_text)
                    filtered_labels.append(label)

        text_list = filtered_texts
        labels = filtered_labels
    else:
        # 过滤长度，保持对齐
        text_label_pairs = [(t, l) for t, l in zip(text_data, labels) if len(t.strip()) > MIN_LEN]
        text_list = [t.strip() for t, _ in text_label_pairs]
        labels = [l for _, l in text_label_pairs]

    if args.num_samples is not None and args.num_samples < len(text_list):
        indices = np.random.choice(len(text_list), args.num_samples, replace=False)
        text_list = [text_list[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    execution_logger.info("Loading success. Now loading diffusion model...")

    # generate images
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    
    execution_logger.info("Loading success. Start sampling...")

    images = []
    all_labels = []
    batch_num = (len(text_data)+batch_size-1)//batch_size

    for batch_idx in tqdm(range(batch_num)):

        batch_prompt = text_list[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_labels = labels[batch_idx*batch_size:(batch_idx+1)*batch_size]

        if len(batch_prompt)==0:
            break

        batch_images = pipe(batch_prompt,num_inference_steps=4,guidance_scale=0.0).images
        images.extend(batch_images)
        all_labels.extend(batch_labels)

        if args.save_every is not None and (batch_idx + 1) % args.save_every == 0:
            execution_logger.info(f"Saving checkpoint at batch {batch_idx + 1}...")
            checkpoint_path = f"{args.output}_temp_save.npz"
            np.savez(checkpoint_path, images=np.array(images), labels=np.array(all_labels))

    execution_logger.info("Sampling process accomplished. Saving data...")

    print(f"total number of images:{len(images)}")
    print(f"total number of labels:{len(labels)}")

    # file_name = args.input.split('/')[-1]
    np.savez(args.output,images = np.array(images),labels = np.array(all_labels))

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input",type=str,default="lsun/bedroom_train/Salesforce/blip-image-captioning-large/caption_0.csv")
    parser.add_argument("--output",type=str,default="results/image/LSUN_huggingface/original")
    parser.add_argument("--model",type=str,default="stabilityai/sdxl-turbo")
    parser.add_argument("--num_samples",type=int,default=None)
    parser.add_argument("--save_every", type=int, default=None, help="Save every N batches (optional)")
    parser.add_argument("--filter",type=bool,default=True)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    
    setup_logging(log_file=os.path.join(os.path.dirname(args.output),"log_temp.txt"))
    execution_logger.info("\nExecuting {}...\ninput: {}\noutput: {}\nmodel: {}".format(sys.argv[0],args.input,args.output,args.model))

    main(args)
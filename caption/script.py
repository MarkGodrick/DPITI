import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,Subset
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from captioner import Openai_captioner, Huggingface_captioner, Gemini_captioner, Qwen_captioner
from logger import execution_logger, setup_logging

from caption.dataset import *

IMAGE_SIZE = 256                 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16

np.random.seed(42)
dataset_dict = {
    "lsun":lsun,
    "cat":cat,
    "europeart":europeart,
    "mmcelebahq":ImageFolderDataset,
    "wingit":ImageFolderDataset
}

captioner_dict = {
    "openai":Openai_captioner,
    "huggingface":Huggingface_captioner,
    "gemini":Gemini_captioner,
    "qwen":Qwen_captioner
}

def main(args, config):

    os.makedirs(args.output, exist_ok=True)

    setup_logging(log_file=os.path.join(args.output,"log.txt"))
    
    execution_logger.info("\nExcuting {}...\ncaptioner: {}\noutput: {}\n".format(sys.argv[0],args.captioner,args.output))

    execution_logger.info(f"Loading Dataset...")

    dataset = dataset_dict.get(args.dataset)(**config['dataset'].get(args.dataset, {}))
    
    # indices = np.random.choice(len(dataset),len(dataset),replace=False)
    # dataset = Subset(dataset,indices[:10240])
    if not dataset:
        raise ValueError("Captioner: dataset not recognized.")

    execution_logger.info(f"Loading Success. Loading Captioner...")

    captioner = captioner_dict.get(args.captioner)(config['captioner'].get(args.captioner,{}),os.path.join(args.output,"temp_save.csv"))
    if not dataset:
        raise ValueError("Captioner: captioner not recognized.")

    execution_logger.info("Loading Success. Start Captioning...")

    captions = captioner(dataset)

    df = pd.DataFrame(captions,columns=['text'])
    # df.to_csv(os.path.join(output,"caption.csv"),index=False)

    execution_logger.info("Captions are generated successfully. Saving data as file {}".format(os.path.join(args.output,f"caption.csv")))

    df.to_csv(os.path.join(args.output,f"caption.csv"),index=False)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--captioner',type=str,choices=['huggingface','openai','gemini','qwen'],default='huggingface')
    parser.add_argument('--dataset',type=str,choices=["lsun","cat","wingit","europeart","mmcelebahq"],default="lsun")
    parser.add_argument('--output',type=str,default="results")

    args = parser.parse_args()

    with open("caption/config.json",'r',encoding='utf-8') as f:
        config = json.load(f)

    main(args, config)
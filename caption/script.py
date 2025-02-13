import os
import json
import torch
import argparse
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset,Subset
from torchvision.datasets import LSUN
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from captioner import Openai_captioner, Huggingface_captioner

IMAGE_SIZE = 256                 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 16


class lsun(Dataset):
    def __init__(self, config):
        self.dataset = LSUN(root=config['dataset_path'],
                            classes=[config['lsun_class']],
                            transform=transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.CenterCrop(IMAGE_SIZE)]))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        images, _ = self.dataset[index]
        return images



def main(args, config):

    dataset = lsun(config) 

    idx = 0
    span = (len(dataset)+6-1)//6
    # span = 128
    dataset = Subset(dataset,indices=list(range(idx*span,(idx+1)*span)))

    if args.captioner=="huggingface":
        captioner = Huggingface_captioner(config["captioner"]["huggingface"])
    elif args.captioner=="openai":
        captioner = Openai_captioner(config["captioner"]["openai"])
    else:
        raise ValueError("Captioner type not recognized.")

    captions = captioner(dataset)


    save_path = os.path.join("lsun",config['lsun_class'],config["captioner"]["huggingface"]['hf_model']["model"] if args.captioner=="huggingface" else config["captioner"]["openai"]["openai_run"]["model"])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.DataFrame(captions,columns=['text'])
    # df.to_csv(os.path.join(save_path,"caption.csv"),index=False)
    df.to_csv(os.path.join(save_path,f"caption_{idx}.csv"),index=False)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--captioner',type=str,choices=['huggingface','openai'],default='huggingface')
    parser.add_argument('--output',type=str,default="lsun")

    args = parser.parse_args()

    with open("caption/config.json",'r',encoding='utf-8') as f:
        config = json.load(f)

    main(args, config)
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
from captioner import Openai_captioner, Huggingface_captioner, Gemini_captioner, Qwen_captioner

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

    idx = 1
    # span = (len(dataset)+6-1)//6
    span = 10240
    dataset = Subset(dataset,indices=list(range(idx*span,(idx+1)*span)))

    if args.captioner=="huggingface":
        captioner = Huggingface_captioner(config["captioner"]["huggingface"])
    elif args.captioner=="openai":
        captioner = Openai_captioner(config["captioner"]["openai"])
    elif args.captioner=="gemini":
        captioner = Gemini_captioner(config["captioner"]["gemini"])
    elif args.captioner=="qwen":
        captioner = Qwen_captioner(config["captioner"]["qwen"])
    else:
        raise ValueError("Captioner type not recognized.")

    captions = captioner(dataset)


    df = pd.DataFrame(captions,columns=['text'])
    # df.to_csv(os.path.join(save_path,"caption.csv"),index=False)
    df.to_csv(os.path.join(args.save_path,f"caption{span}_part{idx}.csv"),index=False)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--captioner',type=str,choices=['huggingface','openai','gemini','qwen'],default='huggingface')
    parser.add_argument('--output',type=str,default="lsun")

    args = parser.parse_args()

    with open("caption/config.json",'r',encoding='utf-8') as f:
        config = json.load(f)

    if args.captioner=="openai":
        model_name = config["captioner"]["openai"]["openai_run"]["model"]
    elif args.captioner=="huggingface":
        model_name = config["captioner"]["huggingface"]['hf_model']["model"]
    elif args.captioner=="gemini":
        model_name = config["captioner"]["gemini"]["model"]
    elif args.captioner=="qwen":
        model_name = config["captioner"]["qwen"]["qwen_run"]["model"]
    else:
        raise ValueError()

    save_path = os.path.join("lsun",config['lsun_class'],args.captioner,model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.save_path = save_path

    main(args, config)
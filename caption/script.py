import os
import json
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets import LSUN
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

IMAGE_SIZE = 256                 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 16
captioner_name = "Salesforce/blip-image-captioning-large"



class lsun(Dataset):
    def __init__(self, config):
        self.dataset = LSUN(root=config['dataset_path'],
                            classes=[config['lsun_class']],
                            transform=transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.CenterCrop(IMAGE_SIZE)]))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index][0]

def main():

    with open("caption/config.json",'r',encoding='utf-8') as f:
        config = json.load(f)

    dataset = lsun(config)

    # idx = 5
    # span = len(dataset)//6
    # dataset = Subset(dataset,indices=list(range(idx*span,(idx+1)*span)))

    captions = []

    captioner = pipeline(
        "image-to-text", 
        model = captioner_name,
        device = DEVICE
        )

    for caption in tqdm(captioner(dataset,batch_size=batch_size, num_workers=8, max_new_tokens=500),total=len(dataset)):
        captions.append(caption[0]['generated_text'])

    save_path = os.path.join("lsun",config['lsun_class'],config['hf_model'])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.DataFrame(captions,columns=['text'])
    df.to_csv(os.path.join(save_path,f"caption.csv"),index=False)
    # df.to_csv(os.path.join(save_path,f"caption_{idx}.csv"),index=False)
    
    

if __name__ == "__main__":
    main()
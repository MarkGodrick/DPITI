import os
import zipfile
import requests
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.datasets import LSUN
from torchvision import transforms
from datasets import load_dataset
from wilds import get_dataset
from collections import defaultdict
from torch.utils.data import Dataset,Subset

IMAGE_SIZE = 256

def download(url: str, fname: str, chunk_size=1024):
    """
    From:
    https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)



class lsun(Dataset):
    def __init__(self, root_dir="data", classes=["bedroom_train"]):
        self.dataset = LSUN(root=root_dir,
                            classes=classes,
                            transform=transforms.Compose(
                                [transforms.Resize(IMAGE_SIZE),
                                 transforms.CenterCrop(IMAGE_SIZE)]))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index][0],self.dataset[index][1]
    


class cat(Dataset):
    """The Cat dataset."""

    #: The URL of the dataset
    URL = "https://www.kaggle.com/api/v1/datasets/download/fjxmlzn/cat-cookie-doudou"

    def __init__(self, root_dir="data", res=512):
        """Constructor.

        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 512
        :type res: int, optional
        """
        self._zip_path = os.path.join(root_dir, "cat-cookie-doudou.zip")
        self.resolution = res
        self._download()
        self.images, self.labels = self._read_data()

    def _download(self):
        """Download the dataset if it does not exist."""
        if not os.path.exists(self._zip_path):
            os.makedirs(os.path.dirname(self._zip_path), exist_ok=True)
            download(url=self.URL, fname=self._zip_path)

    def _read_data(self):
        """Read the data from the zip file."""
        data = []
        labels = []
        transform = transforms.Compose([transforms.Resize(self.resolution),transforms.CenterCrop(self.resolution)])

        with zipfile.ZipFile(self._zip_path) as z:
            for name in tqdm(z.namelist(), desc="Reading zip file"):
                with z.open(name) as f:
                    image = Image.open(f)
                    data.append(transform(image))
                    labels.append(name.split('/')[0])
        return data, labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index],self.labels[index]
    
    

class camelyon17(Dataset):
    def __init__(self, root_dir="dataset", split = "train", res = 64):
        dataset = get_dataset(dataset="camelyon17", download=True, root_dir=root_dir)
        self.dataset = dataset.get_subset(split)
        self.transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label, _ = self.dataset[index]
        image = self.transform(image)
        label = int(label)
        return image, label
    

class waveui(Dataset):
    def __init__(self, split="train", res = 256):
        self.transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        self.dataset = load_dataset("agentsea/wave-ui",split=split)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.transform(self.dataset[int(index)]['image'].convert("RGB")),0



class lex10k(Dataset):
    def __init__(self, split="train", res = 256):
        self.transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        self.dataset = load_dataset("X-ART/LeX-10K",split=split)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.transform(self.dataset[int(index)]['image'].convert("RGB")),0
    


class europeart(Dataset):
    def __init__(self, split="train", res = 256):
        self.transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        self.dataset = load_dataset("biglam/european_art",split=split)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.transform(self.dataset[int(index)]['image'].convert("RGB")),0

  
class imagenet100(Dataset):
    def __init__(self, split="train", res = 256):
        self.transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        self.dataset = load_dataset("ilee0022/ImageNet100",split=split)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.transform(self.dataset[int(index)]['image'].convert("RGB")),self.dataset[int(index)]["label"]

class omni(Dataset):
    def __init__(self, split="train", res = 256):
        self.transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        self.dataset = load_dataset("OmniGen2/OmniContext",split=split)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.transform(self.dataset[int(index)]['input_images'][0]),0


class celeba(Dataset):
    def __init__(self, target_label, folder = "datasets/celeba", split="train", res = 256, ratio = 0.95):
        self.label = target_label
        self.split = split
        self.ratio = ratio
        # attr process
        with open(os.path.join(folder,"list_attr_celeba.txt"), 'r') as f:
            lines = f.readlines()
            num_images = int(lines[0])
            attributes = lines[1].split()
            # Store the attributes for each image in a dictionary
            image_attributes = {}
            for i in range(num_images):
                image_id, *attr_values = lines[i+2].split()
                image_attributes[image_id] = dict(zip(attributes, attr_values))

        # image process
        self.transform = transforms.Compose([
            transforms.Resize(res),
            transforms.CenterCrop(res)
        ])

        self.images = sorted([
            (os.path.join(root, file),image_attributes[file])
            for root, _, files in os.walk(os.path.join(folder,"img_align_celeba"))
            for file in files
            if file.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        # train test split
        self.train_indices = np.random.choice(len(self.images),int(self.ratio*len(self.images)),replace=False)
        self.test_indices = [idx for idx in range(len(self.images)) if idx not in self.train_indices]

    def __len__(self):
        return len(self.train_indices) if self.split=="train" else len(self.test_indices)
    
    def __getitem__(self, index):
        # get index
        idx = self.train_indices[index] if self.split=="train" else self.test_indices[index]
        # get image and label
        item = self.images[idx]
        
        img = Image.open(item[0]).convert('RGB')
        img = self.transform(img)
        label = 1 if int(item[1][self.label])>0 else 0

        return img, label


class ImageFolderDataset(Dataset):
    def __init__(self, folder, res = 256):
        self.folder = folder
        self.transform = transforms.Compose([
            transforms.Resize(res),transforms.CenterCrop(res)
        ])

        # Collect all .png files, sorted (important for sequential order)
        self.images = sorted(
            [f for f in os.listdir(folder) if f.endswith((".jpg", ".png", ".jpeg"))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder, self.images[idx])
        image = Image.open(image_path).convert("RGB")  # or "L" for grayscale

        if self.transform:
            image = self.transform(image)

        return image,0

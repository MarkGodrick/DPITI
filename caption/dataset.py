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
        images, _ = self.dataset[index]
        return images
    


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
        self.images = self._read_data()

    def _download(self):
        """Download the dataset if it does not exist."""
        if not os.path.exists(self._zip_path):
            os.makedirs(os.path.dirname(self._zip_path), exist_ok=True)
            download(url=self.URL, fname=self._zip_path)

    def _read_data(self):
        """Read the data from the zip file."""
        data = []
        transform = transforms.Compose([transforms.Resize(self.resolution),transforms.CenterCrop(self.resolution)])

        with zipfile.ZipFile(self._zip_path) as z:
            for name in tqdm(z.namelist(), desc="Reading zip file"):
                with z.open(name) as f:
                    image = Image.open(f)
                    data.append(transform(image))
        return data
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]
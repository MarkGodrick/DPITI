import pandas as pd
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import LSUN
import os

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME

np.random.seed(42)

class LSUN_bedroom(Data):
    """The LSUN bedroom dataset."""

    def __init__(self, split="train", root_dir="dataset/lsun", res=256, max_length = 300000):
        """Constructor.

        :param split: The split of the dataset. It should be either "train", "val", or "test", defaults to "train"
        :type split: str, optional
        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 64
        :type res: int, optional
        :raises ValueError: If the split is invalid
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        save_path = os.path.join(root_dir,"preprocessed",split)
        if os.path.exists(os.path.join(root_dir,"preprocessed",split)):
            super().__init__()
            self.load_checkpoint(save_path)
            return
        
        transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        dataset = LSUN(root=root_dir,classes=[f"bedroom_{split}"],transform=transform)
        total_length = min(len(dataset),max_length)
        indices = np.random.choice(len(dataset),total_length,replace=False)

        images = []
        labels = []
        for i in tqdm(range(total_length)):
            image, label = dataset[indices[i]]
            images.append(np.array(transform(image)))
            labels.append(label)
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: labels,
            }
        )
        metadata = {"label_info": [{"name": "bedroom"}]}
        super().__init__(data_frame=data_frame, metadata=metadata)

        self.save_checkpoint(save_path)
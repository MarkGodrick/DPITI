import pandas as pd
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import LSUN
import os
from PIL import Image
from omegaconf import OmegaConf


from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.logging import execution_logger

from datasets import load_dataset

np.random.seed(42)

class LSUN_bedroom(Data):
    """The LSUN bedroom dataset."""

    def __init__(self, split="train", root_dir="datasets", res=256, max_length = 300000):
        """Constructor.

        :param split: The split of the dataset. It should be either "train", "val", or "test", defaults to "train"
        :type split: str, optional
        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 64
        :type res: int, optional
        :param max_length: The maximum length of the dataset, will random sample min(max_length,len(dataset)) samples
        :type max_length: int, optional
        :raises ValueError: If the split is invalid
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        save_path = os.path.join(root_dir,"preprocessed",split)
        if os.path.exists(os.path.join(root_dir,"preprocessed",split)):
            execution_logger.info("Processed dataset detected. Loading preprocessed dataset.")
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
    

    def __len__(self):
        return len(self.data_frame)
    

    def __getitem__(self, index):
        return self.data_frame[IMAGE_DATA_COLUMN_NAME][index],self.data_frame[LABEL_ID_COLUMN_NAME][index]
    

class waveui(Data):
    """The waveui dataset."""

    def __init__(self, split="train", root_dir="datasets", res=256, max_length = 300000):
        """Constructor.

        :param split: The split of the dataset. It should be either "train", "val", or "test", defaults to "train"
        :type split: str, optional
        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 64
        :type res: int, optional
        :param max_length: The maximum length of the dataset, will random sample min(max_length,len(dataset)) samples
        :type max_length: int, optional
        :raises ValueError: If the split is invalid
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        save_path = os.path.join(root_dir,"preprocessed",split)
        if os.path.exists(os.path.join(root_dir,"preprocessed",split)):
            execution_logger.info("Processed dataset detected. Loading preprocessed dataset.")
            super().__init__()
            self.load_checkpoint(save_path)
            return
        
        transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        dataset = load_dataset("agentsea/wave-ui",split=split)
        total_length = min(len(dataset),max_length)
        indices = np.random.choice(len(dataset),total_length,replace=False)

        images = []
        labels = []
        for i in tqdm(range(total_length)):
            image = dataset[int(indices[i])]["image"]
            images.append(np.array(transform(image)))
            labels.append(0)
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: labels,
            }
        )
        metadata = {"label_info": [{"name": "screenshot"}]}
        super().__init__(data_frame=data_frame, metadata=metadata)

        self.save_checkpoint(save_path)
    

    def __len__(self):
        return len(self.data_frame)
    

    def __getitem__(self, index):
        return self.data_frame[IMAGE_DATA_COLUMN_NAME][index],self.data_frame[LABEL_ID_COLUMN_NAME][index]
    





class lex10k(Data):
    """The waveui dataset."""

    def __init__(self, split="train", root_dir="datasets", res=512, max_length = 300000):
        """Constructor.

        :param split: The split of the dataset. It should be either "train", "val", or "test", defaults to "train"
        :type split: str, optional
        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 64
        :type res: int, optional
        :param max_length: The maximum length of the dataset, will random sample min(max_length,len(dataset)) samples
        :type max_length: int, optional
        :raises ValueError: If the split is invalid
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        save_path = os.path.join(root_dir,"preprocessed",split)
        if os.path.exists(os.path.join(root_dir,"preprocessed",split)):
            execution_logger.info("Processed dataset detected. Loading preprocessed dataset.")
            super().__init__()
            self.load_checkpoint(save_path)
            return
        
        transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        dataset = load_dataset("X-ART/LeX-10K",split=split)
        total_length = min(len(dataset),max_length)
        indices = np.random.choice(len(dataset),total_length,replace=False)

        images = []
        labels = []
        for i in tqdm(range(total_length)):
            image = dataset[int(indices[i])]["image"]
            images.append(np.array(transform(image)))
            labels.append(0)
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: labels,
            }
        )
        metadata = {"label_info": [{"name": "generated image"}]}
        super().__init__(data_frame=data_frame, metadata=metadata)

        self.save_checkpoint(save_path)
    

    def __len__(self):
        return len(self.data_frame)
    

    def __getitem__(self, index):
        return self.data_frame[IMAGE_DATA_COLUMN_NAME][index],self.data_frame[LABEL_ID_COLUMN_NAME][index]



class europeart(Data):
    """The waveui dataset."""

    def __init__(self, split="train", root_dir="datasets", res=512, max_length = 300000):
        """Constructor.

        :param split: The split of the dataset. It should be either "train", "val", or "test", defaults to "train"
        :type split: str, optional
        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 64
        :type res: int, optional
        :param max_length: The maximum length of the dataset, will random sample min(max_length,len(dataset)) samples
        :type max_length: int, optional
        :raises ValueError: If the split is invalid
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        save_path = os.path.join(root_dir,"preprocessed",split)
        if os.path.exists(os.path.join(root_dir,"preprocessed",split)):
            execution_logger.info("Processed dataset detected. Loading preprocessed dataset.")
            super().__init__()
            self.load_checkpoint(save_path)
            return
        
        transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        dataset = load_dataset("biglam/european_art",split=split)
        total_length = min(len(dataset),max_length)
        indices = np.random.choice(len(dataset),total_length,replace=False)

        images = []
        labels = []
        for i in tqdm(range(total_length)):
            image = dataset[int(indices[i])]["image"]
            if image.mode!="RGB":
                image = image.convert("RGB")
            images.append(np.array(transform(image)))
            labels.append(0)
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: labels,
            }
        )
        metadata = {"label_info": [{"name": "European artwork"}]}
        super().__init__(data_frame=data_frame, metadata=metadata)

        self.save_checkpoint(save_path)
    

    def __len__(self):
        return len(self.data_frame)
    

    def __getitem__(self, index):
        return self.data_frame[IMAGE_DATA_COLUMN_NAME][index],self.data_frame[LABEL_ID_COLUMN_NAME][index]


class imagenet100(Data):
    """The ImageNet100 dataset."""

    def __init__(self, split="train", root_dir="datasets", res=512, max_length = 300000):
        """Constructor.

        :param split: The split of the dataset. It should be either "train", "val", or "test", defaults to "train"
        :type split: str, optional
        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 64
        :type res: int, optional
        :param max_length: The maximum length of the dataset, will random sample min(max_length,len(dataset)) samples
        :type max_length: int, optional
        :raises ValueError: If the split is invalid
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        
        save_path = os.path.join(root_dir,"preprocessed",split)
        if os.path.exists(os.path.join(root_dir,"preprocessed",split)):
            execution_logger.info("Processed dataset detected. Loading preprocessed dataset.")
            super().__init__()
            self.load_checkpoint(save_path)
            return
        
        transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        dataset = load_dataset("ilee0022/ImageNet100",split=split)
        total_length = min(len(dataset),max_length)
        indices = np.random.choice(len(dataset),total_length,replace=False)

        images = []
        labels = []
        label_info = {}
        for i in tqdm(range(total_length)):
            image = dataset[int(indices[i])]["image"]
            label = dataset[int(indices[i])]["label"]
            if image.mode!="RGB":
                image = image.convert("RGB")
            images.append(np.array(transform(image)))
            labels.append(label)
            if label not in label_info:
                label_info[label] = dataset[int(indices[i])]["text"]
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: labels,
            }
        )
        metadata = {"label_info": [{"name": value} for value in label_info.values()]}
        super().__init__(data_frame=data_frame, metadata=metadata)

        self.save_checkpoint(save_path)
    

    def __len__(self):
        return len(self.data_frame)
    

    def __getitem__(self, index):
        return self.data_frame[IMAGE_DATA_COLUMN_NAME][index],self.data_frame[LABEL_ID_COLUMN_NAME][index]




class celeba(Data):
    def __init__(self, target_label, folder = "datasets/celeba", split="train", res = 256, ratio = 0.95):
        split = split
        ratio = ratio
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
        transform = transforms.Compose([
            transforms.Resize(res),
            transforms.CenterCrop(res)
        ])

        items = sorted([
            (os.path.join(root, file),image_attributes[file])
            for root, _, files in os.walk(os.path.join(folder,"img_align_celeba"))
            for file in files
            if file.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        # train test split
        train_indices = np.random.choice(len(items),int(ratio*len(items)),replace=False)
        test_indices = [idx for idx in range(len(items)) if idx not in train_indices]
        self.target_indices = train_indices if split=="train" else test_indices
        # prepare for Data init
        images = [transform(Image.open(items[int(idx)][0]).convert('RGB')) for idx in self.target_indices]
        labels = [1 if int(items[int(idx)][1][target_label])>0 else 0 for idx in self.target_indices]
        
        data_frame = pd.DataFrame({
            IMAGE_DATA_COLUMN_NAME:images,
            LABEL_ID_COLUMN_NAME:labels
        })
        metadata = {"label_info":[{"name": "_".join(["Not",target_label])}, {"name": target_label}]}
        super().__init__(data_frame=data_frame,metadata=metadata)

        self.save_checkpoint(os.path.join(folder,"preprocessed",target_label,split))

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        return self.data_frame[IMAGE_DATA_COLUMN_NAME][index],self.data_frame[LABEL_ID_COLUMN_NAME][index]


class ImageFolderDataset(Data):
    """The ImageFolderDataset dataset."""

    def __init__(self, split="train", root_dir="datasets", res=256, max_length = 307200, label = False):
        """Constructor.

        :param split: The split of the dataset. It should be either "train", "val", or "test", defaults to "train"
        :type split: str, optional
        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 64
        :type res: int, optional
        :param max_length: The maximum length of the dataset, will random sample min(max_length,len(dataset)) samples
        :type max_length: int, optional
        :raises ValueError: If the split is invalid
        """
        load_path = os.path.join(root_dir,split)
        save_path = os.path.join(root_dir,"preprocessed",split)
        if os.path.exists(os.path.join(root_dir,"preprocessed",split)):
            execution_logger.info("Processed dataset detected. Loading preprocessed dataset.")
            super().__init__()
            self.load_checkpoint(save_path)
            return
        
        transform = transforms.Compose([transforms.Resize(res),transforms.CenterCrop(res)])
        self.images = sorted(
            [f for f in os.listdir(load_path) if f.endswith((".jpg", ".png", ".jpeg"))]
        )
        total_length = min(len(self.images),max_length)
        indices = np.random.choice(len(self.images),total_length,replace=False)

        if label:
            label_conf = OmegaConf.load(os.path.join(root_dir,"label.yaml"))
        else:
            label_conf = {
                "label_info": [{"name":"image"}],
                "image_label": [0]*len(self.images)
            }

        images = []
        labels = []
        for i in tqdm(range(total_length)):
            image = Image.open(os.path.join(load_path,self.images[int(indices[i])])).convert("RGB")
            images.append(np.array(transform(image)))
            labels.append(label_conf["image_label"][i])
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: labels,
            }
        )
        metadata = {"label_info":label_conf["label_info"]}
        super().__init__(data_frame=data_frame, metadata=metadata)

        self.save_checkpoint(save_path)
    

    def __len__(self):
        return len(self.data_frame)
    

    def __getitem__(self, index):
        return self.data_frame[IMAGE_DATA_COLUMN_NAME][index],self.data_frame[LABEL_ID_COLUMN_NAME][index]

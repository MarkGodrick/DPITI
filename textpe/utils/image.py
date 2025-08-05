import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL.Image import Image

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.logging import execution_logger

from cleanfid.inception_torchscript import InceptionV3W
from cleanfid.resize import build_resizer
from cleanfid.resize import make_resizer
import tempfile
import os

IMAGE_SIZE = 256
np.random.seed(42)

def data_from_dataset(dataset, length = float("inf"), save_path = "datasets/embedding", label_dict = None, batch_size = 1, random_shuffle = True)->Data:
    
    pe_data = Data()
    total_length = min(length, len(dataset))
    execution_logger.info(f"The length of dataset is {total_length}, and random shuffle is {random_shuffle}")

    os.makedirs(os.path.join(save_path,f"length_{total_length:08}"),exist_ok=True)
    if pe_data.load_checkpoint(os.path.join(save_path,f"length_{total_length:08}")):
        execution_logger.info("Preprocessed data detected, loading preprocessed data.")
        return pe_data
    
    execution_logger.info("No preprocessed data detected. Computing the embeddings of the dataset...")

    if random_shuffle:
        indices = np.random.choice(len(dataset),total_length,replace=False)
    else:
        indices = np.arange(total_length)

    inception = InceptionV3W(path="/data/whx/models", download=True, resize_inside=False).to("cuda")
    resizer = build_resizer("clean")

    all_embeddings = []
    all_labels = []
    for batch_ptr in tqdm(range(0,total_length,batch_size)):
        sample_batch = []
        label_batch = []
        for idx in range(batch_ptr,min(total_length,batch_ptr+batch_size)):
            sample,label = dataset[indices[idx]]
            if isinstance(sample,torch.Tensor):
                sample = sample.numpy()
            elif isinstance(sample,Image):
                sample = np.array(sample)
            if sample.dtype!=np.uint8:
                sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
            if sample.shape==(3,IMAGE_SIZE,IMAGE_SIZE):
                sample = np.transpose(sample,(1,2,0))
            sample = resizer(sample)
            sample_batch.append(sample)
            label_batch.append(label)

        samples = np.array(sample_batch).transpose(0,3,1,2)
        assert samples.shape[1]==3
        assert samples.dtype==np.float32
        embeddings = inception(torch.from_numpy(samples).to("cuda"))
        all_labels.extend(label_batch)
        all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings,dim=0)
    all_embeddings = all_embeddings.cpu().detach().numpy()
    
    execution_logger.info(f"embedding shape:{all_embeddings.shape}")
    data_frame = pd.DataFrame({
        IMAGE_DATA_COLUMN_NAME : list(all_embeddings),
        LABEL_ID_COLUMN_NAME : list(all_labels)
    })
    
    metadata = {
        "label_columns":[],
        "text_column":"text",
        "label_info":[{"name":"","column_values":{}}]
                }
    if label_dict:
        label_set = set(all_labels)
        metadata = {
            "label_columns":["label"] if label_dict else [],
            "text_column":"text",
            "label_info":[{"name": f"label: {label_dict[i].lower()}","column_values":{"label": label_dict[i].lower()}} for i in range(len(label_set))]
            }
    execution_logger.info("embedding computation complete. Saving computed data.")

    pe_data = Data(data_frame=data_frame,metadata=metadata)
    pe_data.save_checkpoint(os.path.join(save_path,f"length_{total_length:08}"))

    return pe_data

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

def data_from_dataset(dataset, length = 20000)->Data:
    all_samples = []
    all_labels = []

    execution_logger.info("transforming dataset to data object...")
    execution_logger.info(f"The length of dataset is {min(length, len(dataset))}")
    for idx in tqdm(range(min(len(dataset),length))):
        sample,label = dataset[idx]
        if isinstance(sample,torch.Tensor):
            sample = sample.numpy()
        elif isinstance(sample,Image):
            sample = np.array(sample)
        all_samples.append(sample)
        all_labels.append(label)
    all_samples = np.array(all_samples)
    if all_samples.dtype!=np.uint8:
        execution_logger.info(f"all_samples.shape:{all_samples.shape}")
        all_samples = np.around(np.clip(all_samples * 255, a_min=0, a_max=255)).astype(np.uint8)
        all_samples = np.transpose(all_samples, (0, 2, 3, 1))
    data_frame = pd.DataFrame({
        IMAGE_DATA_COLUMN_NAME : list(all_samples),
        LABEL_ID_COLUMN_NAME : list(all_labels)
    })
    metadata = {"label_info":[{"name":"None"}]}

    execution_logger.info("dataset transformation succeed.")

    return Data(data_frame=data_frame,metadata=metadata)



def emb_from_data(data: Data, batch_size = 16) -> np.ndarray:

    execution_logger.info("computing the embeddings of the data...")
    _temp_folder = tempfile.TemporaryDirectory()
    _inception = InceptionV3W(path=_temp_folder.name, download=True, resize_inside=False).to("cuda")
    _resizer = build_resizer("clean")

    images = data.data_frame[IMAGE_DATA_COLUMN_NAME].values
    images = np.stack(images,axis=0)

    execution_logger.info(f"images.shape:{images.shape}")

    if images.shape[3] == 1:
        images = np.repeat(images, 3, axis=3)
    embeddings = []
    for i in tqdm(range(0, len(images), batch_size)):
        transformed_x = []
        for j in range(i, min(i + batch_size, len(images))):
            image = _resizer(images[j])
            transformed_x.append(image)
        transformed_x = np.stack(transformed_x, axis=0).transpose((0, 3, 1, 2))
        embeddings.append(_inception(torch.from_numpy(transformed_x).to("cuda")))
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu().detach().numpy()

    execution_logger.info("embedding computation complete.")

    return embeddings
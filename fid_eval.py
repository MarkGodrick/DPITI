import pandas as pd
import os
import sys
import numpy as np
from torchvision.datasets import LSUN
from torchvision import transforms
from textpe.utils.image import data_from_dataset
from textpe.utils.callbacks import _ComputeFID
from textpe.utils.embedding import T2I_embedding
from pe.embedding.image import Inception
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME
from pe.data import Data
from pe.logger import CSVPrint
from pe.logging import setup_logging, execution_logger


PATH = "/data/whx/textDP/lsun/bedroom_train/openai/gpt-4o-mini/pe/meta-llama/noise_multiplier=0/image_voting_lsun_hist/deserted/few_shot_01"
IMAGE_SIZE = 256
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.CenterCrop(IMAGE_SIZE),transforms.ToTensor()])

setup_logging(log_file=os.path.join(PATH,"logs","log.txt"))

execution_logger.info("Preparing private dataset...")

dataset = LSUN("dataset/lsun",classes=['bedroom_train'],transform=transform)
data = data_from_dataset(dataset)

execution_logger.info("Private dataset preparation complete.")

embedding_priv = Inception(res=256, batch_size=16)
embedding_syn = T2I_embedding(model="stabilityai/sdxl-turbo")

execution_logger.info("Preparing ComputeFID object...")

compute_fid = _ComputeFID(priv_data=data,embedding_priv=embedding_priv,embedding_syn=embedding_syn)
compute_fid_vote = _ComputeFID(priv_data=data,embedding_priv=embedding_priv,embedding_syn=embedding_syn,filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
compute_fid_variation = _ComputeFID(priv_data=data,embedding_priv=embedding_priv,embedding_syn=embedding_syn,filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: 0})

execution_logger.info("ComputeFID object complete.")

csv_print = CSVPrint(output_folder=os.path.join(PATH,"logs"))

syn_data = Data()
ckpt_ptr = 3
callbacks = [compute_fid, compute_fid_vote, compute_fid_variation]
loggers = [csv_print]

while os.path.exists(os.path.join(PATH,"checkpoint",f"{ckpt_ptr:09}")):
    ckpt_path = os.path.join(PATH,"checkpoint",f"{ckpt_ptr:09}")
    syn_data.load_checkpoint(ckpt_path)
    metric_items = []
    for callback in callbacks:
        metric_items.extend(callback(syn_data) or [])
    for logger in loggers:
        logger.log(iteration=syn_data.metadata.iteration,metric_items=metric_items)


    ckpt_ptr+=1
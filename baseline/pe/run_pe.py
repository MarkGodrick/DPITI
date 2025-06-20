from baseline.pe.utils.dataset import *
from baseline.pe.utils.api_image import StableDiffusion
from baseline.pe.utils.callbacks import _ComputeFID
from baseline.pe.utils.embedding import Inception
from baseline.pe.utils.histogram import NNhistogram
from pe.logging import setup_logging, execution_logger
from pe.runner import PE
from pe.population import PEPopulation
# from pe.api.image import StableDiffusion
# from pe.embedding.image import Inception
from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import SampleImages
# from pe.callback import ComputeFID
from pe.logger import ImageFile
from pe.logger import CSVPrint
from pe.logger import LogPrint
from pe.data import Data
from textpe.utils.image import data_from_dataset
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.data.image import Cat, Camelyon17
from pe.api.image import ImprovedDiffusion270M
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME
from pe.dp.gaussian import get_noise_multiplier

import os
import json
import argparse
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True
IMAGE_SIZE = 256
ITERATIONS = 10
NUM_OF_PRIV_DATASET = 300000

dataset_dict = {
    "lsun":LSUN_bedroom,
    "cat":Cat,
    "europeart":europeart,
    "waveui":waveui,
    "wingit": ImageFolderDataset,
    "spritefright":ImageFolderDataset,
    "imagenet100":imagenet100
}

def main(args, config):
    setup_logging(log_file=os.path.join(args.output, "log.txt"))

    data = dataset_dict.get(args.dataset)(**config["dataset"].get(args.dataset))
    
    embed_from_dataset = data_from_dataset(data,length=NUM_OF_PRIV_DATASET,save_path=os.path.join("datasets",args.dataset,"embedding"))

    delta = 1.0/len(data)/np.log(len(data))
    noise_multiplier = get_noise_multiplier(epsilon=1.0,num_steps=10,delta=delta)

    if args.api=="StableDiffusion":
        api = StableDiffusion(
            prompt=config["prompt"].get(args.dataset),
            variation_degrees=[1,1,0.75,0.75,0.75,0.75,0.5,0.5,0.5,0.5],
        )
    elif args.api == "ImprovedDiffusion":
        api = ImprovedDiffusion270M(
            variation_degrees=[0] * 2 + [1] * 2 + [2] * 2 + [3] * 1 + list(range(7, 10)),
            timestep_respacing=["ddim10"] * 7 + ["40"] * 3,
        )
    else:
        raise ValueError()
    

    embedding = Inception(res=IMAGE_SIZE, batch_size=32)
    histogram = NNhistogram(
        embedding=embedding,
        mode="L2",
        lookahead_degree=4,
        api=api,
        priv_data_emb=embed_from_dataset
    )
    population = PEPopulation(
        api=api, 
        histogram_threshold=noise_multiplier*1.2,
        )

    save_checkpoints = SaveCheckpoints(os.path.join(args.output, "checkpoint"))
    sample_images = SampleImages()
    compute_fid = _ComputeFID(priv_data=embed_from_dataset, embedding=embedding)

    image_file = ImageFile(output_folder=args.output)
    csv_print = CSVPrint(output_folder=args.output)
    log_print = LogPrint()


    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, sample_images, compute_fid],
        loggers=[image_file, csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[10000] * ITERATIONS,
        delta=delta,
        epsilon=10.0,
        # noise_multiplier=2 * np.sqrt(2),
        checkpoint_path=os.path.join(args.output, "checkpoint"),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,choices=['lsun','cat','waveui','wingit','europeart','spritefright','imagenet100'],default='lsun')
    parser.add_argument("--api",type=str,choices=["StableDiffusion","ImprovedDiffusion"],default="StableDiffusion")
    parser.add_argument("--output",type=str,default="results/baseline/pe")

    args = parser.parse_args()

    with open("baseline/pe/config.json","r") as f:
        config = json.load(f)

    main(args, config)
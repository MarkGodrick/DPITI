from baseline.pe.utils.dataset import LSUN_bedroom
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
from torchvision.datasets import LSUN
from torchvision import transforms


import pandas as pd
import os
import numpy as np

pd.options.mode.copy_on_write = True
IMAGE_SIZE = 256
ITERATIONS = 20
NUM_OF_PRIV_DATASET = 300000
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.CenterCrop(IMAGE_SIZE)])



if __name__ == "__main__":
    exp_folder = "lsun/bedroom_train/baseline/pe/sdxl-turbo/iterations=20/lookahead_degree=4"

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    data = LSUN_bedroom(res=IMAGE_SIZE,max_length=NUM_OF_PRIV_DATASET)
    execution_logger.info("private dataset loaded successfully.")
    
    execution_logger.info("Loading embeddings of private dataset.")
    embed_from_dataset = Data()
    if not (os.path.exists(f"dataset/lsun/embedding/length_{NUM_OF_PRIV_DATASET:08}") and embed_from_dataset.load_checkpoint(f"dataset/lsun/embedding/length_{NUM_OF_PRIV_DATASET:08}")):
        embed_from_dataset = data_from_dataset(LSUN(root="dataset/lsun",classes=["bedroom_train"],transform=transform),length=NUM_OF_PRIV_DATASET)
    execution_logger.info("Computed embeddings of the private dataset loaded successfully.")


    api = StableDiffusion(
        prompt={"bedroom":"A photo of bedroom"},
        variation_degrees=[1,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.25,0.25],
    )
    embedding = Inception(res=IMAGE_SIZE, batch_size=32)
    histogram = NNhistogram(
        embedding=embedding,
        mode="L2",
        lookahead_degree=4,
        api=api,
        priv_data_emb=embed_from_dataset
    )
    population = PEPopulation(api=api, histogram_threshold=5)

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    sample_images = SampleImages()
    compute_fid = _ComputeFID(priv_data=embed_from_dataset, embedding=embedding)

    image_file = ImageFile(output_folder=exp_folder)
    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    delta = 1.0/NUM_OF_PRIV_DATASET/np.log(NUM_OF_PRIV_DATASET)

    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, sample_images, compute_fid],
        loggers=[image_file, csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[2000] * ITERATIONS,
        delta=delta,
        noise_multiplier=2 * np.sqrt(2),
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )

from baseline.utils.lsun_bedroom import LSUN_bedroom
# from baseline.utils.api_image import StableDiffusion
from baseline.utils.callbacks import _ComputeFID
from baseline.utils.embedding import Inception
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation
from pe.api.image import StableDiffusion
# from pe.embedding.image import Inception
from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import SampleImages
from pe.callback import ComputeFID
from pe.logger import ImageFile
from pe.logger import CSVPrint
from pe.logger import LogPrint

import pandas as pd
import os
import numpy as np

pd.options.mode.copy_on_write = True
IMAGE_SIZE = 256
ITERATIONS = 18

if __name__ == "__main__":
    exp_folder = "lsun/bedroom/baseline/pe/stable-diffusion-v1-4/lookahead_degree=4/threshold=5"

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    data = LSUN_bedroom(res=IMAGE_SIZE)
    api = StableDiffusion(
        prompt={"bedroom":"A photo of bedroom"},
        variation_degrees=list(np.arange(1.0, 0.9, -0.02)) + list(np.arange(0.88, 0.36, -0.04)),
    )
    embedding = Inception(res=IMAGE_SIZE, batch_size=32)
    histogram = NearestNeighbors(
        embedding=embedding,
        mode="L2",
        lookahead_degree=4,
        api=api,
    )
    population = PEPopulation(api=api, histogram_threshold=5)

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    sample_images = SampleImages()
    compute_fid = _ComputeFID(priv_data=data, embedding=embedding)

    image_file = ImageFile(output_folder=exp_folder)
    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, sample_images, compute_fid],
        loggers=[image_file, csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[2000] * ITERATIONS,
        delta=3e-6,
        noise_multiplier=2 * np.sqrt(2),
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )

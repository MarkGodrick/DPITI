from dotenv import load_dotenv
import argparse
import json
from textpe.utils.text import text
from textpe.utils.image import data_from_dataset
from pe.logging import setup_logging, execution_logger
from pe.runner import PE
from pe.population import PEPopulation
from pe.api.text import LLMAugPE
from pe.llm import OpenAILLM, HuggingfaceLLM
from pe.embedding.text import SentenceTransformer
from pe.embedding.image import Inception
from textpe.utils.embedding import *
from pe.histogram import NearestNeighbors
from textpe.utils.histogram import ImageVotingNN
from pe.callback import SaveCheckpoints
from pe.callback import ComputeFID
from textpe.utils.dataset import *
from textpe.utils.callbacks import _ComputeFID
from pe.callback import SaveTextToCSV
from pe.logger import CSVPrint
from pe.logger import LogPrint
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

import pandas as pd
import os
import sys
import numpy as np


pd.options.mode.copy_on_write = True
IMAGE_SIZE = 256

dataset_dict = {
    "lsun":lsun,
    "cat":cat,
    "camelyon17":camelyon17,
    "waveui":waveui,
    "lex10k":lex10k,
    "europeart":europeart,
    "mmcelebahq":ImageFolderDataset
}

def main(args, config):
    
    exp_folder = args.output
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    load_dotenv()

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    execution_logger.info("\nExecuting {}...\ninput: {}\npe llm: {}\noutput: {}".format(sys.argv[0],args.data,args.llm,args.output))


    data = text(root_dir=args.data)
    dataset = dataset_dict.get(args.dataset)(**config['dataset'].get(args.dataset,{}))
    data_from_lsun = data_from_dataset(dataset,length=300000,save_path=os.path.join("datasets",args.dataset,"embedding"))

    if args.llm=='huggingface':
        llm = HuggingfaceLLM(**config["model"]["Huggingface"])
    elif args.llm=='openai':
        llm = OpenAILLM(**config["model"]["OpenAI"])
    else:
        raise ValueError("llm argument not recognized.")
    
    api = LLMAugPE(
        llm=llm,
        random_api_prompt_file=os.path.join(current_folder, config["api_prompt"][args.dataset]['random']),
        variation_api_prompt_file=os.path.join(current_folder, config["api_prompt"][args.dataset]['variation']),
        min_word_count=25,
        word_count_std=36,
        blank_probabilities=0.5
    )

    embedding_syn = hfpipe_embedding(model="stabilityai/sdxl-turbo")
    # embedding_syn = dpldm_embedding(config_path="DPLDM/configs/latent-diffusion/txt2img-1p4B-eval.yaml", ckpt_path="textpe/dpldm-models/text2img-large/model.ckpt")

    histogram = ImageVotingNN(
        embedding=embedding_syn,
        mode="L2",
        lookahead_degree=0,
        priv_dataset=data_from_lsun
    )
    population = PEPopulation(
        api=api, initial_variation_api_fold=6, next_variation_api_fold=6, keep_selected=True, selection_mode="rank"
    )

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    compute_fid_vote = _ComputeFID(priv_data=data_from_lsun, embedding=embedding_syn, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
    compute_fid_variation = _ComputeFID(priv_data=data_from_lsun, embedding=embedding_syn, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: 0})
    save_text_to_csv = SaveTextToCSV(output_folder=os.path.join(exp_folder, "synthetic_text"))

    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    num_private_samples = len(data.data_frame)
    delta = 1.0 / num_private_samples / np.log(num_private_samples)

    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, save_text_to_csv, compute_fid_vote, compute_fid_variation],
        loggers=[csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[2000] * 10,
        delta=delta,
        epsilon=1.0,
        # noise_multiplier=0,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output',type=str,default="results/text")
    parser.add_argument('--data',type=str,default="lsun/bedroom_train")
    parser.add_argument('--llm',type=str,choices=['openai','huggingface'],default='huggingface')
    parser.add_argument('--dataset',type=str,choices=['lsun','cat','camelyon17','waveui','lex10k','europeart','mmcelebahq'],default='lsun')

    args = parser.parse_args()

    with open("textpe/config.json",'r',encoding='utf-8') as f:
        config = json.load(f)

    main(args, config)
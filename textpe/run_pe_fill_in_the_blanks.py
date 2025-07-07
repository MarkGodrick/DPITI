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
from textpe.utils.llm import *
from textpe.utils.callbacks import _ComputeFID
from pe.callback import SaveTextToCSV
from pe.logger import CSVPrint
from pe.logger import LogPrint
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

import pandas as pd
import os
import sys
import numpy as np
from omegaconf import OmegaConf

from utils.embedding_multigpu import *

pd.options.mode.copy_on_write = True
IMAGE_SIZE = 256

dataset_dict = {
    "lsun":lsun,
    "cat":cat,
    "camelyon17":camelyon17,
    "waveui":waveui,
    "lex10k":lex10k,
    "europeart":europeart,
    "imagenet100":imagenet100,
    "mmcelebahq":ImageFolderDataset,
    "wingit":ImageFolderDataset,
    "spritefright":ImageFolderDataset,
    "omni":omni
}

llm_dict = {
    "openai":OpenAILLM,
    "huggingface":HuggingfaceLLM,
    "qwen":QwenAILLM
}

embedding_dict = {
    "huggingface": hfpipe_embedding,
    "dpldm": dpldm_embedding,
    "infinity":infinity_embedding
}


def main(args, config):
    
    exp_folder = args.output
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    load_dotenv()

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    execution_logger.info(f"Command Line Arguments:{sys.argv}")

    OmegaConf.save(config,os.path.join(exp_folder,"config.yaml"))

    data = text(root_dir=args.data)
    dataset = dataset_dict.get(args.dataset)(**config['dataset'].get(args.dataset,{}))
    embeded_data = data_from_dataset(dataset,length=300000,save_path=os.path.join("datasets",args.dataset,"embedding"))

    llm = llm_dict.get(args.llm)(**config["model"].get(args.llm))
    
    api = LLMAugPE(
        llm=llm,
        random_api_prompt_file=os.path.join(current_folder, config["api_prompt"][args.dataset]['random']),
        variation_api_prompt_file=os.path.join(current_folder, config["api_prompt"][args.dataset]['variation']),
        min_word_count=25,
        word_count_std=36,
        blank_probabilities=config.running.blank_probabilities
    )

    embedding_syn = embedding_dict.get(args.embedding, None)(**config.embedding.get(args.embedding))

    if args.voting == "image":
        histogram = ImageVotingNN(
            embedding=embedding_syn,
            mode="L2",
            lookahead_degree=config.running.lookahead_degree,
            priv_dataset=embeded_data,
            api = api
        )
    elif args.voting == "text":
        histogram = NearestNeighbors(
            embedding=embedding_syn,
            mode="L2",
            lookahead_degree=config.running.lookahead_degree,
            api = api
        )
    else:
        raise ValueError()
    
    population = PEPopulation(
        # api=api, keep_selected=True, selection_mode="rank"
        api=api, keep_selected=True, selection_mode="rank",initial_variation_api_fold=config.running.initial_variation_api_fold,next_variation_api_fold=config.running.next_variation_api_fold
    )

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    compute_fid_vote = _ComputeFID(priv_data=embeded_data, embedding=embedding_syn)
    # compute_fid_vote = _ComputeFID(priv_data=embeded_data, embedding=embedding_syn, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: -1})
    # compute_fid_variation = _ComputeFID(priv_data=embeded_data, embedding=embedding_syn, filter_criterion={VARIATION_API_FOLD_ID_COLUMN_NAME: 0})
    save_text_to_csv = SaveTextToCSV(output_folder=os.path.join(exp_folder, "synthetic_text"))

    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    num_private_samples = len(data.data_frame)
    delta = 1.0 / num_private_samples / np.log(num_private_samples)

    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, save_text_to_csv, compute_fid_vote],
        # callbacks=[save_checkpoints, save_text_to_csv, compute_fid_vote, compute_fid_variation],
        loggers=[csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[config.running.num_samples] * config.running.total_iterations,
        delta=delta,
        epsilon=config.running.epsilon,
        noise_multiplier=config.running.noise_multiplier,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output',type=str,default="results/text")
    parser.add_argument('--data',type=str,default="lsun/bedroom_train")
    parser.add_argument('--llm',type=str,choices=['openai','huggingface','qwen'],default='huggingface')
    parser.add_argument('--embedding',type=str,choices=['huggingface','dpldm','infinity'],default='huggingface')
    parser.add_argument('--voting',type=str,choices=['image','text'],default='image')
    parser.add_argument('--dataset',type=str,choices=['lsun','cat','camelyon17','waveui','lex10k','europeart','mmcelebahq','wingit','spritefright','imagenet100','omni'],default='lsun')

    args = parser.parse_args()

    config = OmegaConf.load("textpe/config.yaml")

    main(args, config)
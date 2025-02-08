from dotenv import load_dotenv
import argparse
from text import text
from pe.logging import setup_logging
from pe.runner import PE
from pe.population import PEPopulation
from pe.api.text import LLMAugPE
from pe.llm import OpenAILLM, HuggingfaceLLM
from pe.embedding.text import SentenceTransformer
from pe.histogram import NearestNeighbors
from pe.callback import SaveCheckpoints
from pe.callback import ComputeFID
from pe.callback import SaveTextToCSV
from pe.logger import CSVPrint
from pe.logger import LogPrint

import pandas as pd
import os
import numpy as np

pd.options.mode.copy_on_write = True

def main(args):
    
    exp_folder = args.output
    current_folder = os.path.dirname(os.path.abspath(__file__))

    load_dotenv()

    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    data = text(root_dir=args.data,file_name='caption_0.csv')

    if args.llm=='huggingface':
        llm = HuggingfaceLLM(max_completion_tokens=448, model_name_or_path="gpt2", temperature=1.0)
    elif args.llm=='openai':
        llm = OpenAILLM(max_completion_tokens=1000, model="gpt-4o-mini-2024-07-18", temperature=1.2, num_threads=4)
    else:
        raise ValueError("llm argument not recognized.")
    
    api = LLMAugPE(
        llm=llm,
        random_api_prompt_file=os.path.join(current_folder, "random_api_prompt.json"),
        variation_api_prompt_file=os.path.join(current_folder, "variation_api_prompt.json"),
    )
    embedding = SentenceTransformer(model="sentence-t5-base")
    histogram = NearestNeighbors(
        embedding=embedding,
        mode="L2",
        lookahead_degree=0,
    )
    population = PEPopulation(
        api=api, initial_variation_api_fold=6, next_variation_api_fold=6, keep_selected=True, selection_mode="rank"
    )

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    compute_fid = ComputeFID(priv_data=data, embedding=embedding)
    save_text_to_csv = SaveTextToCSV(output_folder=os.path.join(exp_folder, "synthetic_text"))

    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    num_private_samples = len(data.data_frame)
    delta = 1.0 / num_private_samples / np.log(num_private_samples)

    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, save_text_to_csv, compute_fid],
        loggers=[csv_print, log_print],
    )
    pe_runner.run(
        num_samples_schedule=[2000] * 11,
        delta=delta,
        epsilon=1.0,
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output',type=str,default="results/text/LSUN_huggingface/part_0")
    parser.add_argument('--data',type=str,default="lsun/bedroom_train/Salesforce/blip-image-captioning-large")
    parser.add_argument('--llm',type=str,choices=['openai','huggingface'],default='huggingface')

    args = parser.parse_args()

    main(args)
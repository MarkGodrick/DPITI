from pe.data import Data

from pe.logging import setup_logging, execution_logger

from pe.callback import SaveTextToCSV

from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME

import argparse

import os

import sys

def main(args):


    data = Data()

    data.load_checkpoint(args.input)

    data = data.filter({VARIATION_API_FOLD_ID_COLUMN_NAME: args.target})

    SaveTextToCSV(output_folder=args.output)(data)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",type=str)
    parser.add_argument("--output",type=str)
    parser.add_argument("--target",type=int,default=-1,choices=[-1,0])

    args = parser.parse_args()

    setup_logging(log_file=os.path.join(args.output, "log.txt"))
    execution_logger.info("\nExecuting {}...\ninput: {}\noutput: {}\ntarget: {}".format(sys.argv[0],args.input,args.output,args.target))

    main(args)
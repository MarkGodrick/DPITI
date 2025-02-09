#! /bin/bash
python caption/script.py
python textpe/run_pe.py
python run_stable_diff.py
python evaluation/evaluator.py
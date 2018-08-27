#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="emnlp_baseline_run_dg"
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
rm -r /datastore/liu121/nosqldb2/emnlp_ukplab/data/bbn_kn/*
rm -r /datastore/liu121/nosqldb2/emnlp_ukplab/data/cadec/*
rm -r /datastore/liu121/nosqldb2/emnlp_ukplab/data/nvd/*

python ../data_generator.py
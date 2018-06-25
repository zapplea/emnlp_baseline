#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="emnlp_baseline"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

echo "loading"
module load python/3.6.1
module load cudnn/v6
module load cuda/8.0.61
module load keras/2.1.3-py36
module load protobuf/3.5.1
echo "loaded"
rm /datastore/liu121/nosqldb2/emnlp_ukplab/models/*
rm /datastore/liu121/nosqldb2/emnlp_ukplab/data/pkl/*
python ../Train_NER_Eng.py
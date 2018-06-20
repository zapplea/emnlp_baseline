#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="liu121"
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=express
echo "loading"
module load python/3.6.1
echo "loaded"

if test $1 = "pp";
then
    python preprocess.py
elif test $1 = "len";
then
    python max_sentence_len.py
fi
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

if test $1 = "dg";
then
    python data_generator.py
elif test $1 = "sample";
then
    python data_sampling.py
elif test $1 = "stat";
then
    python statistics.py
elif test $1 = "len";
then
    python max_sentence_len.py
fi
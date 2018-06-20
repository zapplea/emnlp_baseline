#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="liu121"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
echo "loading"
module load python/3.6.1
echo "loaded"

if test $1 = "split";
then
    python data_split2.py
elif test $1 = "sampling";
then
    python data_sampling.py
elif test $1 = "test";
then
    python data_split_unittest.py
elif test $1 = "org";
then
    python check_org.py
elif test $1 = "stat";
then
    python type_statistics.py
elif test $1 = "max_len";
then
    python max_sentence_len.py
fi
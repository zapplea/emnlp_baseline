#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="liu121"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
echo "loading"
module load python/3.6.1
echo "loaded"

if test $1 = "gen";
then
    python data_generator.py
elif test $1  = "sat";
then
    python statistics.py
elif test $1 = "split";
then
    python data_split.py
fi
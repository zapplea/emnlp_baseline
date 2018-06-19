#!/bin/bash

#SBATCH --get-user-env
#SBATCH --job-name="liu121"
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

echo "loading"
module load python/3.6.1
echo "loaded"

if [ $1 = "n" ]
then
    ./conlleval.txt -d '\t' -o 'OTHER' -r < $2
elif [ $1 = "y" ]
then
    python evaluate_overlap.py -f $2 -o 'OTHER'
fi
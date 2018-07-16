#!/bin/bash

if [ $1 = "bbn_kn" ];
then
    sbatch run_learn.sh bbn_kn 1.0
    sbatch run_learn.sh bbn_kn 2.0
    sbatch run_learn.sh bbn_kn 4.0
    sbatch run_learn.sh bbn_kn 8.0
    sbatch run_learn.sh bbn_kn 16.0
elif [ $1 = "nvd" ];
then
    sbatch run_learn.sh nvd 1.0
    sbatch run_learn.sh nvd 2.0
    sbatch run_learn.sh nvd 4.0
    sbatch run_learn.sh nvd 8.0
    sbatch run_learn.sh nvd 16.0
elif [ $1 = "cadec" ];
then
    sbatch run_learn.sh cadec 1.0
    sbatch run_learn.sh cadec 2.0
    sbatch run_learn.sh cadec 4.0
    sbatch run_learn.sh cadec 8.0
    sbatch run_learn.sh cadec 16.0
fi
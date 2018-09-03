#!/bin/bash

if [ $1 = "bbn_kn" ];
then
    sbatch run_learn.sh bbn_kn 1.0
    sbatch run_learn.sh bbn_kn 2.0
    sbatch run_learn.sh bbn_kn 4.0
    sbatch run_learn.sh bbn_kn 8.0
    sbatch run_learn.sh bbn_kn 16.0
fi
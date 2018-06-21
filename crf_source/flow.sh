#!/bin/bash
rm /datastore/liu121/nosqldb2/crf_source/report/*
rm /datastore/liu121/nosqldb2/crf_source/model/*
sbatch run_learn.sh 0
sbatch run_learn.sh 1
sbatch run_learn.sh 2
sbatch run_learn.sh 3
sbatch run_learn.sh 4
#!/bin/bash

rm /datastore/liu121/nosqldb2/emnlp_ukplab/models/*
rm /datastore/liu121/nosqldb2/emnlp_ukplab/data/pkl/*

sbatch run_learn.sh bbn
sbatch run_learn.sh cadec
sbatch run_learn.sh nvd
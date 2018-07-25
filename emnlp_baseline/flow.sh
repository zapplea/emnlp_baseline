#!/bin/bash
stage1=$1
if [ $USER = "liu121" ]
then
    if [ $stage1 = "True" ]
    then
        echo remove
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/*
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/*
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/*
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/*

        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/*
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/*
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/*
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/*
    elif [ $stage1 = "False1" ]
    then
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_bbn_kn/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_nvd/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec_simple/*

        rm /datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/report/report_*\.*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/report/report_*\.*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/report/report_*\.*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/report/report_*\.*
    elif [ $stage1 = "False2" ]
    then
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_bbn_kn/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_nvd/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec_simple/*

        rm /datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/report/report_*\.*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/report/report_*\.*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/report/report_*\.*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/report/report_*\.*
    fi
fi

if [ $USER = "liu121" ]
then
    if [ $stage1 = "True" ]
    then
        echo "true"

        sbatch run_learn.sh 0 $stage1 "bbn_bbn_kn"
        sbatch run_learn.sh 0 $stage1 "bbn_cadec"
#        sbatch run_learn.sh 0 $stage1 "bbn_cadec_simple"
        sbatch run_learn.sh 0 $stage1 "bbn_nvd"

        sbatch run_learn.sh 0 $stage1 "conll_bbn_kn"
        sbatch run_learn.sh 0 $stage1 "conll_cadec"
#        sbatch run_learn.sh 0 $stage1 "conll_cadec_simple"
        sbatch run_learn.sh 0 $stage1 "conll_nvd"

    elif [ $stage1 = "False1" ]
    then
        sbatch run_learn.sh 0 $stage1 "bbn_bbn_kn"
#        sbatch run_learn.sh 1 $stage1 "bbn_bbn_kn"
#        sbatch run_learn.sh 2 $stage1 "bbn_bbn_kn"
#        sbatch run_learn.sh 3 $stage1 "bbn_bbn_kn"
#        sbatch run_learn.sh 4 $stage1 "bbn_bbn_kn"

        sbatch run_learn.sh 0 $stage1 "bbn_cadec"
##        sbatch run_learn.sh 1 $stage1 "bbn_cadec"
#        sbatch run_learn.sh 2 $stage1 "bbn_cadec"
#        sbatch run_learn.sh 3 $stage1 "bbn_cadec"
##        sbatch run_learn.sh 4 $stage1 "bbn_cadec"

        sbatch run_learn.sh 0 $stage1 "bbn_nvd"
##        sbatch run_learn.sh 1 $stage1 "bbn_nvd"
#        sbatch run_learn.sh 2 $stage1 "bbn_nvd"
#        sbatch run_learn.sh 3 $stage1 "bbn_nvd"
##        sbatch run_learn.sh 4 $stage1 "bbn_nvd"

        sbatch run_learn.sh 0 $stage1 "bbn_cadec_simple"
##        sbatch run_learn.sh 1 $stage1 "bbn_cadec_simple"
#        sbatch run_learn.sh 2 $stage1 "bbn_cadec_simple"
#        sbatch run_learn.sh 3 $stage1 "bbn_cadec_simple"
##        sbatch run_learn.sh 4 $stage1 "bbn_cadec_simple"
    elif [ $stage1 = "False2" ]
    then
        sbatch run_learn.sh 0 $stage1 "conll_bbn_kn"
#        sbatch run_learn.sh 1 $stage1 "conll_bbn_kn"
#        sbatch run_learn.sh 2 $stage1 "conll_bbn_kn"
#        sbatch run_learn.sh 3 $stage1 "conll_bbn_kn"
#        sbatch run_learn.sh 4 $stage1 "conll_bbn_kn"



        sbatch run_learn.sh 0 $stage1 "conll_cadec_simple"
#        sbatch run_learn.sh 1 $stage1 "conll_cadec_simple"
#        sbatch run_learn.sh 2 $stage1 "conll_cadec_simple"
#        sbatch run_learn.sh 3 $stage1 "conll_cadec_simple"
#        sbatch run_learn.sh 4 $stage1 "conll_cadec_simple"

        sbatch run_learn.sh 0 $stage1 "conll_cadec"
#        sbatch run_learn.sh 1 $stage1 "conll_cadec"
#        sbatch run_learn.sh 2 $stage1 "conll_cadec"
#        sbatch run_learn.sh 3 $stage1 "conll_cadec"
#        sbatch run_learn.sh 4 $stage1 "conll_cadec"
#
        sbatch run_learn.sh 0 $stage1 "conll_nvd"
#        sbatch run_learn.sh 1 $stage1 "conll_nvd"
#        sbatch run_learn.sh 2 $stage1 "conll_nvd"
#        sbatch run_learn.sh 3 $stage1 "conll_nvd"
#        sbatch run_learn.sh 4 $stage1 "conll_nvd"
    fi
fi
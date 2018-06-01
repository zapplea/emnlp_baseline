#!/bin/bash
stage1=$1
if [ $USER = "liu121" ]
then
    if [ $stage1 = "True" ]
    then
        echo rm is forbidden
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/model_bbn/*
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/model_bbn200/*
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/model_conll/*
        rm -r /datastore/liu121/nosqldb2/emnlp_baseline/model_conll200/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/report_bbn_kn/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/report_cadec/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/report_cadec_simple/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/report_conll/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/report_nvd/*
    elif [ $stage1 = "False1" ]
    then
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_bbn_kn/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_nvd/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec_simple/*
    elif [ $stage1 = "False2" ]
    then
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_bbn_kn/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_nvd/*
        rm /datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec_simple/*

    fi
fi

if [ $USER = "liu121" ]
then
    if [ $stage1 = "True" ]
    then
        echo "true"

        sbatch run_learn.sh 0 $stage1 "bbn_kn"
        sbatch run_learn.sh 0 $stage1 "bbn_kn200"
#        sbatch run_learn.sh 1 $stage1 "bbn_kn"
#        sbatch run_learn.sh 2 $stage1 "bbn_kn"
#        sbatch run_learn.sh 3 $stage1 "bbn_kn"
#        sbatch run_learn.sh 4 $stage1 "bbn_kn"
#        sbatch run_learn.sh 5 $stage1 "bbn_kn"

        sbatch run_learn.sh 0 $stage1 "conll"
        sbatch run_learn.sh 0 $stage1 "conll200"
#        sbatch run_learn.sh 1 $stage1 "conll"
#        sbatch run_learn.sh 2 $stage1 "conll"
#        sbatch run_learn.sh 3 $stage1 "conll"
#        sbatch run_learn.sh 4 $stage1 "conll"
#        sbatch run_learn.sh 5 $stage1 "conll"
    elif [ $stage1 = "False0" ]
    then
        sbatch run_learn.sh 0 $stage1 "bbn_bbn_kn"
#        sbatch run_learn.sh 1 $stage1 "bbn_bbn_kn"
        sbatch run_learn.sh 2 $stage1 "bbn_bbn_kn"
        sbatch run_learn.sh 3 $stage1 "bbn_bbn_kn"
#        sbatch run_learn.sh 4 $stage1 "bbn_bbn_kn"
    elif [ $stage1 = "False1" ]
    then
        sbatch run_learn.sh 0 $stage1 "bbn_bbn_kn"
#        sbatch run_learn.sh 1 $stage1 "bbn_bbn_kn"
        sbatch run_learn.sh 2 $stage1 "bbn_bbn_kn"
        sbatch run_learn.sh 3 $stage1 "bbn_bbn_kn"
#        sbatch run_learn.sh 4 $stage1 "bbn_bbn_kn"

        sbatch run_learn.sh 0 $stage1 "bbn_cadec"
#        sbatch run_learn.sh 1 $stage1 "bbn_cadec"
        sbatch run_learn.sh 2 $stage1 "bbn_cadec"
        sbatch run_learn.sh 3 $stage1 "bbn_cadec"
#        sbatch run_learn.sh 4 $stage1 "bbn_cadec"
#
        sbatch run_learn.sh 0 $stage1 "bbn_nvd"
#        sbatch run_learn.sh 1 $stage1 "bbn_nvd"
        sbatch run_learn.sh 2 $stage1 "bbn_nvd"
        sbatch run_learn.sh 3 $stage1 "bbn_nvd"
#        sbatch run_learn.sh 4 $stage1 "bbn_nvd"

        sbatch run_learn.sh 0 $stage1 "bbn_cadec_simple"
#        sbatch run_learn.sh 1 $stage1 "bbn_cadec_simple"
        sbatch run_learn.sh 2 $stage1 "bbn_cadec_simple"
        sbatch run_learn.sh 3 $stage1 "bbn_cadec_simple"
#        sbatch run_learn.sh 4 $stage1 "bbn_cadec_simple"
    elif [ $stage1 = "False2" ]
    then
        sbatch run_learn.sh 0 $stage1 "conll_bbn_kn"
#        sbatch run_learn.sh 1 $stage1 "conll_bbn_kn"
        sbatch run_learn.sh 2 $stage1 "conll_bbn_kn"
        sbatch run_learn.sh 3 $stage1 "conll_bbn_kn"
#        sbatch run_learn.sh 4 $stage1 "conll_bbn_kn"



        sbatch run_learn.sh 0 $stage1 "conll_cadec_simple"
#        sbatch run_learn.sh 1 $stage1 "conll_cadec_simple"
        sbatch run_learn.sh 2 $stage1 "conll_cadec_simple"
#        sbatch run_learn.sh 3 $stage1 "conll_cadec_simple"
#        sbatch run_learn.sh 4 $stage1 "conll_cadec_simple"

        sbatch run_learn.sh 0 $stage1 "conll_cadec"
#        sbatch run_learn.sh 1 $stage1 "conll_cadec"
        sbatch run_learn.sh 2 $stage1 "conll_cadec"
        sbatch run_learn.sh 3 $stage1 "conll_cadec"
#        sbatch run_learn.sh 4 $stage1 "conll_cadec"
#
        sbatch run_learn.sh 0 $stage1 "conll_nvd"
#        sbatch run_learn.sh 1 $stage1 "conll_nvd"
        sbatch run_learn.sh 2 $stage1 "conll_nvd"
        sbatch run_learn.sh 3 $stage1 "conll_nvd"
#        sbatch run_learn.sh 4 $stage1 "conll_nvd"
    fi
    #sbatch run_learn.sh 6
    #sbatch run_learn.sh 7
fi
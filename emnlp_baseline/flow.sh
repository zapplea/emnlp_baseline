#!/bin/bash
stage1=$1
casing=$2
if [ $USER = "liu121" ]
then
    if [ $stage1 = "True" ]
    then
        if [ $casing = "False" ]
        then
            echo remove
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/model/model*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/model/model*
            #rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/model/model*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/model/model*

            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/report/report*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/report/report*
            #rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/report/report*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/report/report*

            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/model/model*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/model/model*
            #rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/model/model*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/model/model*

            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/report/report*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/report/report*
            #rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/report/report*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/report/report*
        else
            echo remove
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/model/casing_model*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/model/casing_model*
            #rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/model/casing_model*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/model/casing_model*

            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/report/casing_report*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/report/casing_report*
            #rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/report/casing_report*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/report/casing_report*

            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/model/casing_model*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/model/casing_model*
            #rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/model/casing_model*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/model/casing_model*

            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/report/casing_report*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/report/casing_report*
            #rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/report/casing_report*
            rm -r /datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/report/casing_report*
        fi
    fi
fi

if [ $USER = "liu121" ]
then
    if [ $stage1 = "True" ]
    then
        echo "true"
        if [ $casing = "False" ]
        then
            sbatch run_learn.sh 0 $stage1 "bbn_bbn_kn" "False"
            sbatch run_learn.sh 0 $stage1 "bbn_cadec" "False"
            sbatch run_learn.sh 0 $stage1 "bbn_nvd" "False"

            sbatch run_learn.sh 0 $stage1 "conll_bbn_kn" "False"
            sbatch run_learn.sh 0 $stage1 "conll_cadec" "False"
            sbatch run_learn.sh 0 $stage1 "conll_nvd" "False"
        else
            sbatch run_learn.sh 0 $stage1 "bbn_bbn_kn" "True"
            sbatch run_learn.sh 0 $stage1 "bbn_cadec" "True"
            sbatch run_learn.sh 0 $stage1 "bbn_nvd" "True"

            sbatch run_learn.sh 0 $stage1 "conll_bbn_kn" "True"
            sbatch run_learn.sh 0 $stage1 "conll_cadec" "True"
            sbatch run_learn.sh 0 $stage1 "conll_nvd" "True"
        fi
    elif [ $stage1 = "False1" ]
    then
        if [ $casing = "False" ]
        then
            sbatch run_learn.sh 0 $stage1 "bbn_bbn_kn" "False"
            sbatch run_learn.sh 1 $stage1 "bbn_bbn_kn" "False"
            sbatch run_learn.sh 2 $stage1 "bbn_bbn_kn" "False"

            sbatch run_learn.sh 0 $stage1 "bbn_cadec" "False"
            sbatch run_learn.sh 1 $stage1 "bbn_cadec" "False"
            sbatch run_learn.sh 2 $stage1 "bbn_cadec" "False"

            sbatch run_learn.sh 0 $stage1 "bbn_nvd" "False"
            sbatch run_learn.sh 1 $stage1 "bbn_nvd" "False"
            sbatch run_learn.sh 2 $stage1 "bbn_nvd" "False"
        else
            sbatch run_learn.sh 0 $stage1 "bbn_bbn_kn" "True"
            sbatch run_learn.sh 1 $stage1 "bbn_bbn_kn" "True"
            sbatch run_learn.sh 2 $stage1 "bbn_bbn_kn" "True"

            sbatch run_learn.sh 0 $stage1 "bbn_cadec" "True"
            sbatch run_learn.sh 1 $stage1 "bbn_cadec" "True"
            sbatch run_learn.sh 2 $stage1 "bbn_cadec" "True"

            sbatch run_learn.sh 0 $stage1 "bbn_nvd" "True"
            sbatch run_learn.sh 1 $stage1 "bbn_nvd" "True"
            sbatch run_learn.sh 2 $stage1 "bbn_nvd" "True"
        fi

    elif [ $stage1 = "False2" ]
    then
        if [ $casing = "False" ]
        then
            sbatch run_learn.sh 0 $stage1 "conll_bbn_kn" "False"
            sbatch run_learn.sh 1 $stage1 "conll_bbn_kn" "False"
            sbatch run_learn.sh 2 $stage1 "conll_bbn_kn" "False"

            sbatch run_learn.sh 0 $stage1 "conll_cadec" "False"
            sbatch run_learn.sh 1 $stage1 "conll_cadec" "False"
            sbatch run_learn.sh 2 $stage1 "conll_cadec" "False"

            sbatch run_learn.sh 0 $stage1 "conll_nvd" "False"
            sbatch run_learn.sh 1 $stage1 "conll_nvd" "False"
            sbatch run_learn.sh 2 $stage1 "conll_nvd" "False"
        else
            sbatch run_learn.sh 0 $stage1 "conll_bbn_kn" "True"
            sbatch run_learn.sh 1 $stage1 "conll_bbn_kn" "True"
            sbatch run_learn.sh 2 $stage1 "conll_bbn_kn" "True"

            sbatch run_learn.sh 0 $stage1 "conll_cadec" "True"
            sbatch run_learn.sh 1 $stage1 "conll_cadec" "True"
            sbatch run_learn.sh 2 $stage1 "conll_cadec" "True"

            sbatch run_learn.sh 0 $stage1 "conll_nvd" "True"
            sbatch run_learn.sh 1 $stage1 "conll_nvd" "True"
            sbatch run_learn.sh 2 $stage1 "conll_nvd" "True"
        fi
    fi
fi
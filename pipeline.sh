#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

# Please comment and uncomment the corresponding part to train and evaluate on 
# different datasets. [CASME | SAMM]

# for CASME
# SUB_LIST=(    sub01 sub02 sub03 sub04 sub05 sub06 sub07 sub08 sub09 \
# sub10 sub11 sub12 sub13 sub14 sub15 sub16 sub17 sub18 sub19 sub20 sub21 sub22 sub23 sub24 sub25 )
 SUB_LIST=(    casme_015 casme_016  casme_019 casme_020 casme_021 casme_022 casme_023 casme_024 \
 casme_025 casme_026 casme_027 casme_029 casme_030 casme_031 casme_032 casme_033 casme_034 \
 casme_035 casme_036 casme_037 casme_038 casme_040 )
 OUTPUT="/code/CodeTest/wzl/RRSN-MEA-0324/output/cas1111/"
 DATASET="cas(me)^2"
 INPUT="/data/DataSets/cas(me)^2/feature_segment_fblabel272/"

# for SAMM

#SUB_LIST=( samm_007 samm_006 samm_008 samm_009 samm_010 samm_011 samm_012 samm_013 samm_014 \
#samm_015 samm_016 samm_017 samm_018 samm_019 samm_020 samm_021 samm_022 samm_023 samm_024 \
#samm_025 samm_026 samm_028 samm_030 samm_031 samm_032 samm_033 samm_034 samm_036 samm_035 \
#samm_037 )
#OUTPUT="/code/CodeTest/wzl/RRSN-MEA-0324/output/samm0723/"
#DATASET="samm"
#INPUT="/data/DataSets/cas(me)^2/feature_segment_sammlabel272/"
for i in ${SUB_LIST[@]}
do
    echo "************ Currently running subject: ${i} ************"
#    # comment the line below if evaluating on available ckpts.
    python trainclassTrimutilv0.py --dataset $DATASET --output $OUTPUT --input $INPUT --subject ${i}  # for training
    python evalclassTrimutilV0.py --dataset $DATASET --output $OUTPUT --input $INPUT --subject ${i}   # for evaluation
done

#output final metrics
python calc_final_scoreclass_uf1.py --output $OUTPUT
# SUB_LIST=( casme_015 casme_016 casme_019 casme_020 casme_021 casme_022 casme_023 casme_024 \
# casme_025 casme_026 casme_027 casme_029 casme_030 casme_031 casme_032 casme_033 casme_034 \
# casme_035 casme_036 casme_037 casme_038 casme_040 )
# OUTPUT="/code/CodeTest/wzl/AUW-GCN/output/flip/"
# DATASET="cas(me)^2"
# INPUT="/data/DataSets/cas(me)^2/feature_segment_fblabel/"
#
## for SAMM
#
##SUB_LIST=( samm_007 samm_006 samm_008 samm_009 samm_010 samm_011 samm_012 samm_013 samm_014 \
##samm_015 samm_016 samm_017 samm_018 samm_019 samm_020 samm_021 samm_022 samm_023 samm_024 \
##samm_025 samm_026 samm_028 samm_030 samm_031 samm_032 samm_033 samm_034 samm_036 samm_035 \
##samm_037 )
##OUTPUT="./tmp_output/sammgc1k5"
##DATASET="samm"
#
#for i in ${SUB_LIST[@]}
#do
#    echo "************ Currently running subject: ${i} ************"
#    # comment the line below if evaluating on available ckpts.
#    python train.py --dataset $DATASET --output $OUTPUT --input $INPUT --subject ${i}  # for training
#    python eval.py --dataset $DATASET --output $OUTPUT --input $INPUT --subject ${i}   # for evaluation
#done
#
##output final metrics
#python calc_final_score.py --output $OUTPUT
# OUTPUT="./output/casme_landmarkdistance"
# DATASET="cas(me)^2"
# INPUT="/data/DataSets/cas(me)^2/feature_landmarkdistance/"
#for i in ${SUB_LIST[@]}
#do
#    echo "************ Currently running subject: ${i} ************"
#    # comment the line below if evaluating on available ckpts.
#    python train.py --dataset $DATASET --output $OUTPUT --input $INPUT --subject ${i}  # for training
#    python eval.py --dataset $DATASET --output $OUTPUT --input $INPUT --subject ${i}   # for evaluation
#done
#
##output final metrics
#python calc_final_score.py --output $OUTPUT
#
#
# OUTPUT="./output/casme_landmarkflow"
# DATASET="cas(me)^2"
# INPUT="/data/DataSets/cas(me)^2/feature_landmarkflow/"
#for i in ${SUB_LIST[@]}
#do
#    echo "************ Currently running subject: ${i} ************"
#    # comment the line below if evaluating on available ckpts.
#    python train.py --dataset $DATASET --output $OUTPUT --input $INPUT --subject ${i}  # for training
#    python eval.py --dataset $DATASET --output $OUTPUT --input $INPUT --subject ${i}   # for evaluation
#done
#
##output final metrics
#python calc_final_score.py --output $OUTPUT
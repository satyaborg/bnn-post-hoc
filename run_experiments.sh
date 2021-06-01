#!/bin/bash
## USAGE: ./run_experiments.sh <path_to_virtualenv>
## activate virtualenv
source "$1"
## datasets used for experiments
declare -a datasets=(
"abalone" "balance-scale" "credit-approval" "german" "ionosphere" 
"landsat-satellite" "letter" "mfeat-karhunen" "mfeat-morphological" "mfeat-zernike" 
"mushroom" "optdigits" "page-blocks" "segment" "spambase" "toy" "vehicle" "waveform-5000" 
"wdbc" "wpbc" "yeast"
)
seed="42"
## main loop
for dataset in "${datasets[@]}"
    do
    python -m run_experiments "$seed" "$dataset"
    echo "$dataset completed!"
done
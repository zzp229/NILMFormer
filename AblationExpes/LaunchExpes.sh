#!/bin/bash

declare -a model_name=("NILMFormerAbStwols" "NILMFormerAbStwotokenst" "NILMFormerAbStwolsandtoken" "NILMFormerAbStwonorm" "NILMFormerAbStrevin" "NILMFormerAbPEAdd" "NILMFormerAbPEfixed" "NILMFormerAbPEtAPE" "NILMFormerAbPElearn" "NILMFormerAbPEno" "NILMFormerAbEmbedConv" "NILMFormerAbEmbedLinear" "NILMFormerAbEmbedPatch")
declare -a window_size=("128" "256" "512")


declare -a cases=("washing_machine" "dishwasher" "kettle" "microwave" "fridge")


for win in ${window_size[@]}; do
  for case in ${cases[@]}; do
    for model in ${model_name[@]}; do
      for seed in 0 1 2; do
        python3 -u UKDALEExperiments.py $task $sp $win $case $model $seed
      done
    done
  done
done


declare -a cases=("WashingMachine" "Dishwasher" "Kettle" "Microwave")

for win in ${window_size[@]}; do
  for case in ${cases[@]}; do
    for model in ${model_name[@]}; do
      for seed in 0 1 2; do
        python3 -u REFITExperiments.sh $task $sp $win $case $model $seed
      done
    done
  done
done


#!/bin/bash

declare -a cases=("WashingMachine" "Dishwasher" "Kettle" "Microwave")


declare -a model_name=("BiGRU" "BiLSTM" "CNN1D" "UNET_NILM" "FCN" "BERT4NILM" "NILMFormer")
declare -a window_size=("128" "256" "512" "360" "720")

for win in ${window_size[@]}; do
  for case in ${cases[@]}; do
    for model in ${model_name[@]}; do
      for seed in 0 1 2; do
        python3 -u REFITExperiments.py $win $case $model $seed
      done
    done
  done
done


declare -a model_name=("ConvNet" "ResNet" "Inception")
declare -a window_size=("day")

for win in ${window_size[@]}; do
  for case in ${cases[@]}; do
    for model in ${model_name[@]}; do
      for seed in 0 1 2; do
        python3 -u REFITExperiments_TSER.py $win $case $model $seed
      done
    done
  done
done

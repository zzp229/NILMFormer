#!/bin/bash

declare -a datasets=("UKDALE" "REFIT")
declare -a appliances=("WashingMachine" "Dishwasher" "Kettle" "Microwave")
declare -a list_models=("BiLSTM" "FCN" "CNN1D" "UNetNILM" "DAResNet" "BERT4NILM" "DiffNILM" "TSILNet" "Energformer" "BiGRU" "STNILM" "NILMFormer")
declare -a window_sizes=("128" "256" "512" "360" "720")

for dataset in ${datasets[@]}; do
  for appliance in ${appliances[@]}; do
    for win in ${window_sizes[@]}; do
      for model in ${list_models[@]}; do
        for seed in 0 1 2; do
          uv run -m expes.launch_one_expe --dataset $dataset --sampling_rate "1min" --appliance $appliance --window_size $win --name_model $model --seed $seed
        done
      done
    done
  done
done


declare -a list_models=("ConvNet" "ResNet" "Inception")
declare -a window_size=("day" "week" "month")

for dataset in ${datasets[@]}; do
  for win in ${window_sizes[@]}; do
    for appliance in ${appliances[@]}; do
      for model in ${list_models[@]}; do
        for seed in 0 1 2; do
          uv run -m expes.launch_one_expe --dataset $dataset --sampling_rate "1min" --appliance $appliance --window_size $win --name_model $model --seed $seed
        done
      done
    done
  done
done

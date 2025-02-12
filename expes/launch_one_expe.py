#################################################################################################################
#
# @copyright : ©2025 EDF
# @author : Adrien Petralia
# @description : NILMFormer - Experiments
#
#################################################################################################################

import argparse
import yaml
import logging
import numpy as np

from src.helpers.utils import *
from src.helpers.preprocessing import (
    UKDALE_DataBuilder,
    REFIT_DataBuilder,
    split_train_test_nilmdataset,
    split_train_test_pdl_nilmdataset,
    nilmdataset_to_tser,
)
from src.helpers.dataset import NILMscaler
from expes.expes_helpers import launch_models_training

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def launch_one_experiment(expes_config):
    np.random.seed(seed=expes_config["seed"])

    logging.info("Process data...")
    if expes_config["dataset"] == "UKDALE":
        data_builder = UKDALE_DataBuilder(
            data_path="data/UKDALE/",
            mask_app=expes_config["app"],
            sampling_rate=expes_config["sampling_rate"],
            window_size=expes_config["window_size"],
        )

        data, st_date = data_builder.get_nilm_dataset(house_indicies=[1, 2, 3, 4, 5])

        if isinstance(expes_config["window_size"], str):
            expes_config["window_size"] = data_builder.window_size

        data_train, st_date_train = data_builder.get_nilm_dataset(
            house_indicies=expes_config["ind_house_train"]
        )
        data_test, st_date_test = data_builder.get_nilm_dataset(
            house_indicies=expes_config["ind_house_test"]
        )

        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_nilmdataset(
                data_train,
                st_date_train,
                perc_house_test=0.2,
                seed=expes_config["seed"],
            )
        )

    elif expes_config["dataset"] == "REFIT":
        data_builder = REFIT_DataBuilder(
            data_path="data/REFIT/RAW_DATA_CLEAN/",
            mask_app=expes_config["app"],
            sampling_rate=expes_config["sampling_rate"],
            window_size=expes_config["window_size"],
        )

        data, st_date = data_builder.get_nilm_dataset(
            house_indicies=expes_config["house_with_app_i"]
        )

        if isinstance(expes_config["window_size"], str):
            expes_config["window_size"] = data_builder.window_size

        data_train, st_date_train, data_test, st_date_test = (
            split_train_test_pdl_nilmdataset(
                data.copy(), st_date.copy(), nb_house_test=2, seed=expes_config["seed"]
            )
        )

        scaler = NILMscaler(
            power_scaling_type=expes_config["power_scaling_type"],
            appliance_scaling_type=expes_config["appliance_scaling_type"],
        )
        data = scaler.fit_transform(data)
        expes_config["scaler"] = scaler
        expes_config["cutoff"] = scaler.appliance_stat2[0]
        expes_config["threshold"] = data_builder.appliance_param[expes_config["app"]][
            "min_threshold"
        ]

        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

        data_train, st_date_train, data_valid, st_date_valid = (
            split_train_test_pdl_nilmdataset(
                data_train, st_date_train, nb_house_test=1, seed=expes_config["seed"]
            )
        )

    logging.info("  Done.")

    scaler = NILMscaler(
        power_scaling_type=expes_config["power_scaling_type"],
        appliance_scaling_type=expes_config["appliance_scaling_type"],
    )
    data = scaler.fit_transform(data)

    if expes_config["name_model"] in ["ConvNet", "ResNet", "Inception"]:
        X, y = nilmdataset_to_tser(data)
        expes_config["scaler"] = scaler

        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

        X_train, y_train = nilmdataset_to_tser(data_train)
        X_valid, y_valid = nilmdataset_to_tser(data_valid)
        X_test, y_test = nilmdataset_to_tser(data_test)

        tuple_data = (
            (X_train, y_train, st_date_train),
            (X_valid, y_valid, st_date_valid),
            (X_test, y_test, st_date_test),
            (X, y, st_date),
        )

    else:
        expes_config["scaler"] = scaler
        expes_config["cutoff"] = scaler.appliance_stat2[0]
        expes_config["threshold"] = data_builder.appliance_param[expes_config["app"]][
            "min_threshold"
        ]

        data_train = scaler.transform(data_train)
        data_valid = scaler.transform(data_valid)
        data_test = scaler.transform(data_test)

        tuple_data = (
            data_train,
            data_valid,
            data_test,
            data,
            st_date_train,
            st_date_valid,
            st_date_test,
            st_date,
        )

    launch_models_training(tuple_data, expes_config)


def main(dataset, sampling_rate, window_size, appliance, name_model, seed):
    """
    Main function to load configuration, update it with parameters,
    and launch an experiment.

    Args:
        dataset (str): Name of the dataset (UKDALE or REFIT).
        sampling_rate (int): Selected sampling rate.
        window_size (int or str): Size of the window (converted to int if possible not day, week or month).
        appliance (str): Selected appliance.
        name_model (str): Name of the model to use for the experiment.
        seed (int): Random seed for reproducibility.
    """

    # Attempt to convert window_size to int
    try:
        window_size = int(window_size)
    except ValueError:
        logging.warning(
            "window_size could not be converted to int. Using its original value: %s",
            window_size,
        )

    logging.info("Dataset: %s", dataset)
    logging.info("Sampling Rate: %s", sampling_rate)
    logging.info("Window Size: %s", window_size)
    logging.info("Appliance : %s", appliance)
    logging.info("Model: %s", name_model)
    logging.info("Seed: %s", seed)

    # Load configurations
    with open("expes/configs/expes_config.yaml", "r") as f:
        expes_config = yaml.safe_load(f)

    with open("expes/configs/datasets_config.yaml", "r") as f:
        datasets_config = yaml.safe_load(f)

        if dataset in datasets_config:
            datasets_config = datasets_config[dataset]
        else:
            logging.error("Wrong dataset name : '%s'.", dataset)

    with open("expes/configs/models_config.yaml", "r") as f:
        baselines_config = yaml.safe_load(f)

        if name_model not in baselines_config:
            logging.error("Wrong model name : '%s'.", name_model)

    # Update experiment config with passed parameters
    expes_config["dataset"] = dataset
    expes_config["appliance"] = appliance
    expes_config["window_size"] = window_size
    expes_config["sampling_rate"] = sampling_rate
    expes_config["seed"] = seed
    expes_config["name_model"] = name_model

    # Safely update configs if keys exist
    if appliance in datasets_config:
        expes_config.update(datasets_config[appliance])
    else:
        logging.error("Appliance '%s' not found in datasets_config.", appliance)

    if name_model in baselines_config:
        expes_config.update(baselines_config[name_model])
    else:
        logging.warning("Model '%s' not found in baselines_config.", name_model)

    # Create directories for results
    path_results = create_dir(expes_config["save_path"])
    path_results = create_dir(f"{path_results}{dataset}_{appliance}_{sampling_rate}/")
    path_results = create_dir(f"{path_results}{window_size}/")

    # Define the path to save experiment results
    expes_config["save_path"] = (
        f"{path_results}{expes_config['name_model']}_{expes_config['seed']}"
    )

    launch_one_experiment(expes_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NILMFormer Experiments.")
    parser.add_argument(
        "--dataset", required=True, type=str, help="Dataset name (UKDALE or REFIT)."
    )
    parser.add_argument(
        "--sampling_rate",
        required=True,
        type=str,
        help="Sampling rate, e.g. '30s', '1min', '10min', etc.).",
    )
    parser.add_argument(
        "--window_size",
        required=True,
        type=str,
        help="Window size used for training, e.g. '128' or 'day.",
    )
    parser.add_argument(
        "--appliance",
        required=True,
        type=str,
        help="Selected appliance, e.g., 'WashingMachine'.",
    )
    parser.add_argument(
        "--name_model", required=True, type=str, help="Name of the model for training."
    )
    parser.add_argument(
        "--seed", required=True, type=int, help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    main(
        dataset=args.dataset,
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        appliance=args.appliance,
        name_model=args.name_model,
        seed=args.seed,
    )

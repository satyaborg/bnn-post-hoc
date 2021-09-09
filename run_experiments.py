import os
import sys
import gc
import keras
import warnings
import json
import time
import numpy as np
import pandas as pd
from time import strftime
import logging
from sklearn.model_selection import train_test_split

import argparse

# custom classes and modules
from src import config
from src.dataset import Dataset
from src.models import custom_feedforward
from src.utils import (
    set_rand_seed,
    get_logger,
    set_tf_loglevel,
    create_folder,
    init_params,
    get_logger,
)

from src.calibration import calibration
from src.betacal.calib.utils.dataframe import MyDataFrame

warnings.filterwarnings("ignore")

filename = strftime("%b%d_%H-%M-%S")
logger = get_logger(filename)
set_tf_loglevel(logging.FATAL)

methods = [None, "var_bayes", "sigmoid", "isotonic", "beta"]
columns = ["dataset", "method", "acc", "loss", "brier", "ece", "probas"]


def save_results(methods, dataset, accs, losses, briers, eces, mean_probas, test):
    """Saves all predictions (train + test + validation) along with features"""
    temp = []
    for m in methods:
        res = np.array(
            [dataset, m if m else "None", accs[m], losses[m], briers[m], eces[m]]
        )
        meta_rep = np.tile(res, test.shape[0]).reshape(test.shape[0], 6)
        arr = np.array(mean_probas[m])
        concat_arr = np.concatenate(
            (meta_rep, test, np.expand_dims(arr, axis=1)), axis=1
        )
        temp.append(concat_arr)
    return temp


def train(name, data_home, **kwargs):
    """Main training function"""
    # initialize the dataset
    dataset = Dataset(name=name, data_home=data_home, **kwargs)
    # prepare dataset
    X, y = dataset.read_data()
    hidden_layers = kwargs.get("hidden_layers")
    logger.info(
        f"Name = {name}\nData shape = {X.shape}\nTarget shape = {y.shape}\nTarget classes = {dataset.classes}\nTarget labels = {dataset.names}"
    )
    df = pd.DataFrame(columns=columns)
    keras.backend.clear_session()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=kwargs.get("seed")
    )
    in_dim = x_train.shape[1]
    model = custom_feedforward(
        in_dim=in_dim, hidden_layers=hidden_layers
    )  # sequential approach
    init_values = init_params(model=model, p=kwargs.get("p"))
    accs, losses, briers, eces, mean_probas, test = calibration(
        init_values=init_values,
        methods=methods,
        dataset=dataset,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        score_type="predict_proba",
        **kwargs,
    )
    del model
    gc.collect()
    temp_results = save_results(
        methods, name, accs, losses, briers, eces, mean_probas, test
    )
    final_results.extend(temp_results)
    for method in methods:
        m_text = "None" if method is None else method
        rows = [
            [
                name,
                m_text,
                accs[method],
                losses[method],
                briers[method],
                eces[method],
                mean_probas[method],
            ]
        ]
        df_temp = pd.DataFrame(rows, columns=columns)
        df = df.append(df_temp, ignore_index=True)

    return df


if __name__ == "__main__":
    # ensure that dataset name is provided
    parser = argparse.ArgumentParser(description="Parameters for the training run")
    parser.add_argument("--dataset_name", required=True, type=str, help="The dataset to train on")
    parser.add_argument("--run_idx", required=True, type=int, help="Run index")
    parser.add_argument("--seed", required=True, type=int, help="Random seed")

    args, unknown = parser.parse_known_args()

    dataset_name = args.dataset_name
    seed = args.seed

    if len(sys.argv) < 3:
        logger.info("Please provide name of the dataset ..")
        sys.exit(0)

    # arguments
    config["seed"] = seed
    name = dataset_name
    
    logger.info(f"Dataset: {name}")
    set_rand_seed(config.get("seed"))  # setting random seeds
    start_time = time.time()
    config["filename"] = filename
    # create required folders
    results_path = f"{config['paths'].get('results')}/{filename}"
    create_folder(results_path)  # creates a folder if does not exist
    # save config file
    with open(f"{results_path}/config.json", "w") as file:
        json.dump(config, file)

    # get max epochs for validation stage
    with open(f"max_epochs.json", "r") as f:
        max_epochs = json.load(f)
        config["vb_epochs"] = max_epochs[name]["vb_epochs"]
        config["nn_epochs"] = max_epochs[name]["nn_epochs"]

    data_home = config.get("paths").get("data")
    df_all = MyDataFrame(columns=columns)
    final_results = []
    df = MyDataFrame(columns=columns)
    logger.info("============================================================")
    df = train(name, data_home, **config)
    logger.info("============================================================\n")
    # save results - including x_test, y_test and y_pred
    df_results = pd.DataFrame(np.vstack(final_results))
    col_names = {0: "dataset", 1: "method", 2: "acc", 3: "loss", 4: "brier", 5: "ece"}
    col_names[len(df_results.columns) - 1] = "y_pred"
    col_names[len(df_results.columns) - 2] = "y_true"
    df_results.rename(columns=col_names, inplace=True)
    df_results.to_csv(os.path.join(results_path, f"{name}_results.csv"), index=False)
    table = df.pivot_table(
        values=["acc", "loss"], index=["dataset", "method"], aggfunc=[np.mean, np.std]
    )
    df.to_csv(os.path.join(results_path, "main_results_data_frame.csv"), index=False)
    table.to_csv(os.path.join(results_path, "main_results.csv"))
    end_time = time.time()
    logger.info(f"Total time elapsed: {end_time - start_time:.2f}")

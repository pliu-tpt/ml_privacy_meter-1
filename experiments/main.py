"""This file is the main entry point for running the priavcy auditing."""
import argparse
import logging
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from argument import get_signal_on_argumented_data
from core import (
    load_dataset_for_existing_models,
    load_existing_models,
    load_existing_target_model,
    prepare_datasets,
    prepare_datasets_for_online_attack_overlap,
    prepare_datasets_for_sample_privacy_risk,
    prepare_information_source,
    prepare_models,
    prepare_priavcy_risk_report,
)
from dataset import get_dataset, get_dataset_subset
from plot import plot_roc, plot_signal_histogram
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve
from torch import nn
from util import (
    check_configs,
    load_leave_one_out_models,
    load_models_with_data_idx_list,
    load_models_without_data_idx_list,
    sweep,
)

from privacy_meter.audit import Audit
from privacy_meter.model import PytorchModelTensor

torch.backends.cudnn.benchmark = True


def str2bool(v):
  return str(v).lower() in ("true", "1")

def setup_log(report_dir: str, name: str, save_file: bool) -> logging.Logger:
    """Generate the logger for the current run.
    Args:
        report_dir (str): folder name of the audit
        name (str): Logging file name.
        save_file (bool): Flag about whether to save to file.
    Returns:
        logging.Logger: Logger object for the current run.
    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)

    if save_file:
        log_format = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s"
            )
        filename = f"{report_dir}/log_{name}.log"

        if not Path(filename).is_file():
            open(filename, 'w+')

        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        my_logger.addHandler(log_handler)

    return my_logger

def parse_extra(parser, configs):
    """Using a parser and a base config, modify the config according to the parser
    For bash experiments.
    Args:
        parser (Parser): Parser with the basic arguments
        configs (dict): Dict that we want to change according to the parser
    Returns:
        configs: modified input config
    """
    for key in configs:
        # Generate arguments for top-level keys
        arg_name = '--{}'.format(key)
        parser.add_argument(arg_name, dest=key, default=None,
                            help='{} parameter'.format(arg_name))
        for subkey in configs[key]:
            # Generate arguments for second-level keys
            arg_name = '--{}.{}'.format(key, subkey)
            parser.add_argument(arg_name, dest=subkey, default=None,
                                help='{} parameter'.format(arg_name))
            if isinstance(configs[key][subkey], dict):
                for subsubkey in configs[key][subkey]:
                    # Generate arguments for eventual third-level keys
                    arg_name = '--{}.{}.{}'.format(key, subkey, subsubkey)
                    parser.add_argument(arg_name, dest=subsubkey, default=None,
                                        help='{} parameter'.format(arg_name))
    # Parse command-line arguments
    args, unknown_args = parser.parse_known_args()
    # Update configuration dictionary with command-line arguments
    if args:
        for key in configs:
            if args.__dict__.get(key) is not None:
                configs[key] = args.__dict__.get(key)
            for subkey in configs[key]:
                if args.__dict__.get(subkey) is not None:
                    configs[key][subkey] = args.__dict__.get(subkey)
                if isinstance(configs[key][subkey], dict):
                    for subsubkey in configs[key][subkey]:
                        arg_name = '{}.{}.{}'.format(key, subkey, subsubkey)
                        if args.__dict__.get(subsubkey) is not None:
                            configs[key][subkey][subsubkey] = args.__dict__.get(subsubkey)
    return configs


def collect_signal(model, model_index, dataset, dataset_indeces, log_dir, signal_name, config):
    """
    Function that queries a given model on a given dataset on specific indeces for a given signal name

    Args:
        model (lamnda: Model): lambda function that will init the model eventually.
        model_index (int): model index in the log_dir
        dataset (torchvision.datasets): dataset to query
        dataset_indeces (List[int]) : which indices to query
        log_dir (str): log_dir 
        signal_name (str): signal's name, usually of the form '{primary_signal}_{post_processing}'
        config (dict): eventual other parameters relative to the signal
    Returns:
        signals: the computed or loaded signals
    """
    def create_or_load_signal_file(path,dataset):
        if not os.path.isfile(path):
            signal_storage = np.full((len(dataset)), np.nan)
            np.save(path, signal_storage)
        else:
            signal_storage = np.load(path, allow_pickle=True)
        return signal_storage

    # Create a directory will all the primary signals per model 
    if not os.path.exists(f"{log_dir}/meta_signal/"):
        os.makedirs(f"{log_dir}/meta_signal/")

    if signal_name == "loss_logit_rescaled":

        primary_signal = "loss"

        if not os.path.exists(f"{log_dir}/meta_signal/{primary_signal}/"):
            os.makedirs(f"{log_dir}/meta_signal/{primary_signal}/")

        signal_path = f"{log_dir}/meta_signal/{primary_signal}/{model_index:04d}.npy"
        # check if the primary signal has been computed
        model_signal = create_or_load_signal_file(signal_path, dataset)

        # check if we already queried the primary signals at the given indeces
        # If not, query it
        if np.isnan(model_signal[dataset_indeces]).sum()>0:
            print("load the model and compute signals for model %d" % idx)
            model_pm = model()
            data, targets = get_dataset_subset(
                dataset, dataset_indeces, config["train"]["model_name"], device=config["audit"]["device"]
            )
            model_signal[dataset_indeces] = get_signal_on_argumented_data(
                    model_pm,
                    data,
                    targets,
                    method=config["audit"]["argumentation"],
            )
            np.save(signal_path, model_signal)
            return model_signal[dataset_indeces]
        else:
            print("loading signals for model %d" % idx)
            return model_signal[dataset_indeces]
    else:
         raise NotImplementedError(f"signal {signal_name} has not yet been implemented...")



def metric_results(fpr_list, tpr_list):
    acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
    roc_auc = auc(fpr_list, tpr_list)

    one_percent = tpr_list[np.where(fpr_list < 0.01)[0][-1]]
    tenth_percent = tpr_list[np.where(fpr_list < 0.001)[0][-1]]
    hundredth_percent = tpr_list[np.where(fpr_list < 0.0001)[0][-1]]

    return roc_auc, acc, one_percent, tenth_percent, hundredth_percent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cf",
        type=str,
        default="experiments/config_models_online.yaml",
        help="Yaml file which contains the configurations",
    )

    # Load the parameters
    args, unknown = parser.parse_known_args()
    with open(args.cf, "rb") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    configs = parse_extra(parser, configs) # parsing more stuff

    check_configs(configs)
    # Set the random seed, log_dir and inference_game
    torch.manual_seed(configs["run"]["random_seed"])
    np.random.seed(configs["run"]["random_seed"])

    log_dir = configs["run"]["log_dir"]
    inference_game_type = configs["audit"]["privacy_game"].upper()

    # Create folders for saving the logs if they do not exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_dir = f"{log_dir}/{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # Set up the logger
    logger = setup_log(report_dir, "time_analysis", configs["run"]["time_log"])

    start_time = time.time()

    # Load or initialize models based on metadata
    if os.path.exists((f"{log_dir}/models_metadata.pkl")):
        with open(f"{log_dir}/models_metadata.pkl", "rb") as f:
            model_metadata_list = pickle.load(f)
    else:
        model_metadata_list = {"model_metadata": {}, "current_idx": 0}
    # Load the dataset
    baseline_time = time.time()
    dataset = get_dataset(configs["data"]["dataset"], configs["data"]["data_dir"])

    privacy_game = configs["audit"]["privacy_game"]

    ############################
    # Privacy auditing for a model or an algorithm
    ############################
    if (
        privacy_game in ["avg_privacy_loss_training_algo", "privacy_loss_model"]
        and "online" not in configs["audit"]["algorithm"]
    ):
        # Load the trained models from disk
        if model_metadata_list["current_idx"] > 0:
            target_model_idx_list = load_existing_target_model(
                len(dataset), model_metadata_list, configs
            )
            trained_target_dataset_list = load_dataset_for_existing_models(
                len(dataset),
                model_metadata_list,
                target_model_idx_list,
                configs["data"],
            )

            trained_target_models_list = load_existing_models(
                model_metadata_list,
                target_model_idx_list,
                configs["train"]["model_name"],
                # trained_target_dataset_list,
                dataset,
            )
            num_target_models = configs["train"]["num_target_model"] - len(
                trained_target_dataset_list
            )
        else:
            target_model_idx_list = []
            trained_target_models_list = []
            trained_target_dataset_list = []
            num_target_models = configs["train"]["num_target_model"]

        # Prepare the datasets
        print(25 * ">" + "Prepare the the datasets")
        data_split_info = prepare_datasets(
            len(dataset), num_target_models, configs["data"]
        )

        logger.info(
            "Prepare the datasets costs %0.5f seconds", time.time() - baseline_time
        )

        # Prepare the target models
        print(25 * ">" + "Prepare the the target models")
        baseline_time = time.time()

        new_model_list, model_metadata_list, new_target_model_idx_list = prepare_models(
            log_dir, dataset, data_split_info, configs["train"], model_metadata_list
        )

        model_list = [*new_model_list, *trained_target_models_list]
        data_split_info["split"] = [
            *data_split_info["split"],
            *trained_target_dataset_list,
        ]
        target_model_idx_list = [*new_target_model_idx_list, *target_model_idx_list]

        logger.info(
            "Prepare the target model costs %0.5f seconds", time.time() - baseline_time
        )

        # Prepare the information sources
        print(25 * ">" + "Prepare the information source, including attack models")
        baseline_time = time.time()
        (
            target_info_source,
            reference_info_source,
            metrics,
            log_dir_list,
            model_metadata_list,
        ) = prepare_information_source(
            log_dir,
            dataset,
            data_split_info,
            model_list,
            configs["audit"],
            model_metadata_list,
            target_model_idx_list,
            configs["train"]["model_name"],
        )
        logger.info(
            "Prepare the information source costs %0.5f seconds",
            time.time() - baseline_time,
        )

        # Call core of privacy meter
        print(25 * ">" + "Auditing the privacy risk")
        baseline_time = time.time()
        audit_obj = Audit(
            metrics=metrics,
            inference_game_type=inference_game_type,
            target_info_sources=target_info_source,
            reference_info_sources=reference_info_source,
            fpr_tolerances=None,
            logs_directory_names=log_dir_list,
        )
        audit_obj.prepare()
        audit_results = audit_obj.run()
        logger.info(
            "Prepare the privacy meter result costs %0.5f seconds",
            time.time() - baseline_time,
        )

        # Generate the privacy risk report
        print(25 * ">" + "Generating privacy risk report")


        roc_auc, acc, one_percent, tenth_percent, hundredth_percent = metric_results(audit_results[0][0].fpr, audit_results[0][0].tpr)

        logger.info(
            "AUC %.4f, Accuracy %.4f" % (roc_auc, acc)
        )
        logger.info(
            "TPR@1%%FPR of %.4f, TPR@0.1%%FPR of %.4f, TPR@0.01%%FPR of %.4f" % (
            one_percent, tenth_percent, hundredth_percent)
        )

        baseline_time = time.time()
        prepare_priavcy_risk_report(
            log_dir,
            audit_results,
            configs["audit"],
            save_path=f"{log_dir}/{configs['audit']['report_log']}",
        )
        print(100 * "#")

        logger.info(
            "Prepare the plot for the privacy risk report costs %0.5f seconds",
            time.time() - baseline_time,
        )

    ############################
    # Privacy auditing for a sample
    ############################
    elif configs["audit"]["privacy_game"] == "privacy_loss_sample":
        # Load existing models that match the requirement
        assert (
            "data_idx" in configs["train"]
        ), "data_idx in config.train is not specified"
        assert (
            "data_idx" in configs["audit"]
        ), "data_idx in config.audit is not specified"

        in_model_idx_list = load_models_with_data_idx_list(
            model_metadata_list, [configs["train"]["data_idx"]]
        )
        model_in_list = load_existing_models(
            model_metadata_list,
            in_model_idx_list,
            configs["train"]["model_name"],
            dataset,
        )
        # Train additional models if the existing models are not enough
        if len(in_model_idx_list) < configs["train"]["num_in_models"]:
            data_split_info_in = prepare_datasets_for_sample_privacy_risk(
                len(dataset),
                configs["train"]["num_in_models"] - len(in_model_idx_list),
                configs["train"]["data_idx"],
                configs["data"],
                "include",
                "leave_one_out",
                model_metadata_list,
            )
            new_in_model_list, model_metadata_list, new_matched_in_idx = prepare_models(
                log_dir,
                dataset,
                data_split_info_in,
                configs["train"],
                model_metadata_list,
            )
            model_in_list = [*new_in_model_list, *model_in_list]
            in_model_idx_list = [*new_matched_in_idx, *in_model_idx_list]
        in_model_list_pm = [
            PytorchModelTensor(
                model_obj=model,
                loss_fn=nn.CrossEntropyLoss(),
                batch_size=1000,
                device=configs["audit"]["device"],
            )
            for model in model_in_list
        ]
        if configs["data"]["split_method"] == "uniform":
            out_model_idx_list = load_models_without_data_idx_list(
                model_metadata_list, [configs["train"]["data_idx"]]
            )
        elif configs["data"]["split_method"] == "leave_one_out":
            out_model_idx_list = load_leave_one_out_models(
                model_metadata_list, [configs["train"]["data_idx"]], in_model_idx_list
            )
        else:
            raise ValueError("The split method is not supported")

        model_out_list = load_existing_models(
            model_metadata_list,
            out_model_idx_list,
            configs["train"]["model_name"],
            dataset,
        )
        # Train additional models if the existing models are not enough
        if len(out_model_idx_list) < configs["train"]["num_out_models"]:
            data_split_info_out = prepare_datasets_for_sample_privacy_risk(
                len(dataset),
                configs["train"]["num_out_models"] - len(out_model_idx_list),
                configs["train"]["data_idx"],
                configs["data"],
                "exclude",
                configs["data"]["split_method"],
                model_metadata_list,
                in_model_idx_list,
            )
            (
                new_out_model_list,
                model_metadata_list,
                new_matched_out_idx,
            ) = prepare_models(
                log_dir,
                dataset,
                data_split_info_out,
                configs["train"],
                model_metadata_list,
            )
            model_out_list = [*new_out_model_list, *model_out_list]
            out_model_idx_list = [*new_matched_out_idx, *out_model_idx_list]

        out_model_list_pm = [
            PytorchModelTensor(
                model_obj=model,
                loss_fn=nn.CrossEntropyLoss(),
                batch_size=1000,
                device=configs["audit"]["device"],
            )
            for model in model_out_list
        ]

        # Test the models' performance on the data indicated by the audit.idx
        data, targets = get_dataset_subset(
            dataset, [configs["audit"]["data_idx"]], configs["audit"]["model_name"]
        )
        in_signal = np.array(
            [
                model.get_rescaled_logits(data, targets).item()
                for model in in_model_list_pm
            ]
        )
        out_signal = np.array(
            [
                model.get_rescaled_logits(data, targets).item()
                for model in out_model_list_pm
            ]
        )

        # Generate the privacy risk report
        plot_signal_histogram(
            in_signal,
            out_signal,
            configs["train"]["data_idx"],
            configs["audit"]["data_idx"],
            f"{log_dir}/{configs['audit']['report_log']}/individual_pr_{configs['train']['data_idx']}_{configs['audit']['data_idx']}.png",
        )
        fpr_list, tpr_list, roc_auc = sweep(in_signal, out_signal)
        plot_roc(
            fpr_list,
            tpr_list,
            roc_auc,
            f"{log_dir}/{configs['audit']['report_log']}/individual_pr_roc_{configs['train']['data_idx']}_{configs['audit']['data_idx']}.png",
        )

    ############################
    # Privacy auditing for an model with online attack (i.e., adversary trains models with/without each target points)
    ############################
    elif "online" in configs["audit"]["algorithm"]:
        # The following code is modified from the original code in the repo: https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021
        baseline_time = time.time()
        p_ratio = configs["data"]["keep_ratio"]
        dataset_size = len(dataset)
        # training_size = configs["data"]["dataset_size"]
        number_of_models_lira = configs["train"]["num_in_models"] + configs["train"]["num_out_models"] + configs["train"]["num_target_model"]

        data_split_info_path = f"{log_dir}/data_split_info.pkl"
        keep_matrix_path = f"{log_dir}/keep_matrix.npy"

        if os.path.isfile(data_split_info_path):
            with open(data_split_info_path, "rb") as f:
                data_split_info = pickle.load(f)
            keep_matrix = np.load(keep_matrix_path, allow_pickle=True)
        else:
            data_split_info, keep_matrix = prepare_datasets_for_online_attack_overlap(
                dataset_size=dataset_size,
                num_models=(
                    number_of_models_lira
                ),
                configs=configs["data"]
            )
            with open(data_split_info_path, "wb") as f:
                pickle.dump(data_split_info, f)
            np.save(keep_matrix_path, keep_matrix)

        ## Points from which we compute the actual signal
        target_and_test_idx = np.array(list(data_split_info["split"][0]["target"]) + list(data_split_info["split"][0]["test"]))

        logger.info(
            "Prepare the datasets costs %0.5f seconds",
            time.time() - baseline_time,
        )
        baseline_time = time.time()
        if model_metadata_list["current_idx"] == 0:
            (model_list, model_metadata_dict, trained_model_idx_list) = prepare_models(
                log_dir,
                dataset, # we train on this...
                data_split_info,
                configs["train"],
                model_metadata_list,
            )
            logger.info(
                "Prepare the models costs %0.5f seconds",
                time.time() - baseline_time,
            )
            baseline_time = time.time()
            signals = []
            for i, model in enumerate(model_list):
                model_init = lambda: PytorchModelTensor(
                    model_obj=model,
                    loss_fn=nn.CrossEntropyLoss(),
                    device=configs["audit"]["device"],
                    batch_size=int(configs["audit"]["audit_batch_size"]),
                )
                tmp_signal = collect_signal(model=model_init,
                                            model_index=i,
                                            dataset=dataset,
                                            dataset_indeces=target_and_test_idx,
                                            log_dir=log_dir,
                                            signal_name=configs["audit"]["signal"],
                                            config=configs)
                signals.append(tmp_signal)
            logger.info(
                "Prepare the signals costs %0.5f seconds",
                time.time() - baseline_time,
            )
        else:
            baseline_time = time.time()
            signals = []
            # for idx in range(number_of_models_lira): # we consider that we train lira online setting first.
            for idx in range(model_metadata_list["current_idx"]):
                model_init = lambda: PytorchModelTensor(
                        model_obj=load_existing_models(
                            model_metadata_list,
                            [idx],
                            configs["train"]["model_name"],
                            dataset,
                            device=configs["audit"]["device"]
                        )[0],
                        loss_fn=nn.CrossEntropyLoss(),
                        device=configs["audit"]["device"],
                        batch_size=int(configs["audit"]["audit_batch_size"]),
                    )
                tmp_signal = collect_signal(model=model_init,
                                            model_index=idx,
                                            dataset=dataset,
                                            dataset_indeces=target_and_test_idx,
                                            log_dir=log_dir,
                                            signal_name=configs["audit"]["signal"],
                                            config=configs)
                signals.append(tmp_signal)
            logger.info(
                "Prepare the signals costs %0.5f seconds",
                time.time() - baseline_time,
            )
        baseline_time = time.time()
        signals = np.array(signals)

        target_signal = signals[-1:, :] # LAST MODEL IS TARGET # shape 1 x N + M
        reference_signals = signals[:-1, :] # shape nb_models x N + M
        reference_keep_matrix = keep_matrix[:-1, target_and_test_idx] # shape nb_models x N + M
        membership = keep_matrix[-1:, target_and_test_idx] # shape 1 x N + M

        np.savez(f"{log_dir}/{configs['audit']['report_log']}/lira_target_signal", target_signal)
        np.savez(f"{log_dir}/{configs['audit']['report_log']}/lira_target_keep", membership)

        np.savez(f"{log_dir}/{configs['audit']['report_log']}/lira_reference_signal", reference_signals)
        np.savez(f"{log_dir}/{configs['audit']['report_log']}/lira_reference_keep", reference_keep_matrix)

        np.savez(f"{log_dir}/{configs['audit']['report_log']}/lira_target_samples", data_split_info["split"][0]["target"])
        np.savez(f"{log_dir}/{configs['audit']['report_log']}/lira_test_samples",data_split_info["split"][0]["test"])

        in_signals = []
        out_signals = []
        number_target = len(data_split_info["split"][0]["target"])

        for data_idx in range(number_target):
            in_signals.append(
                reference_signals[reference_keep_matrix[:, data_idx], data_idx]
            ) # selects the models that have data_idx as train
            out_signals.append(
                reference_signals[~reference_keep_matrix[:, data_idx], data_idx]
            ) # selects the models that don't have data_idx as train

        in_size = min(min(map(len, in_signals)), configs["train"]["num_in_models"])
        out_size = min(min(map(len, out_signals)), configs["train"]["num_out_models"])
        in_signals = np.array([x[:in_size] for x in in_signals]).astype("float32")
        out_signals = np.array([x[:out_size] for x in out_signals]).astype("float32")

        mean_in = np.median(in_signals, 1)
        mean_out = np.median(out_signals, 1)
        fix_variance = configs["audit"]["fix_variance"]
        if fix_variance:
            std_in = np.std(in_signals)
            std_out = np.std(in_signals)
        else:
            std_in = np.std(in_signals, 1)
            std_out = np.std(out_signals, 1)


        prediction = []
        answers = []
        print(membership.shape, target_signal.shape, mean_in.shape, mean_out.shape)
        for ans, sc in zip(membership[:,:number_target], target_signal[:,:number_target]):
            if str2bool(configs["audit"]["offline"]):
                pr_in = 0
            else:
                pr_in = -norm.logpdf(sc, mean_in, std_in + 1e-30)
            pr_out = -norm.logpdf(sc, mean_out, std_out + 1e-30)
            score = pr_in - pr_out
            if len(score.shape) == 2:  # the score is of size (data_size, num_arguments)
                prediction.extend(score.mean(1))
            else:
                prediction.extend(score)
            answers.extend(ans)

        prediction = np.array(prediction)
        answers = np.array(answers, dtype=bool)
        print(prediction.shape, answers.shape, prediction, np.isnan(prediction).sum())
        # Last step: compute the metrics
        fpr_list, tpr_list, _ = roc_curve(answers.ravel(), -prediction.ravel())
        acc = np.max(1 - (fpr_list + (1 - tpr_list)) / 2)
        roc_auc = auc(fpr_list, tpr_list)
        logger.info(
            "Prepare the privacy risks results costs %0.5f seconds",
            time.time() - baseline_time,
        )
        one_percent = tpr_list[np.where(fpr_list < 0.01)[0][-1]]
        tenth_percent = tpr_list[np.where(fpr_list < 0.001)[0][-1]]
        hundredth_percent = tpr_list[np.where(fpr_list < 0.0001)[0][-1]]

        logger.info(
            "AUC %.4f, Accuracy %.4f" % (roc_auc, acc)
        )
        logger.info(
            "TPR@1%%FPR of %.4f, TPR@0.1%%FPR of %.4f, TPR@0.01%%FPR of %.4f" % (one_percent, tenth_percent, hundredth_percent)
        )
        plot_roc(
            fpr_list,
            tpr_list,
            roc_auc,
            f"{log_dir}/{configs['audit']['report_log']}/online_attack.png",
        )

    ############################
    # END
    ############################
    logger.info(
        "Run the priavcy meter for the all steps costs %0.5f seconds",
        time.time() - start_time,
    )

import logging
import os
import pickle
import copy
import json

import numpy as np
import pandas as pd

from logger_configs.configurations import datasets_config, default_thres_config
from datasets.data_preprocess.dataset import get_events
from logger_configs.logger import init_logging
from src.evaluation.evaluation_utils import get_dataset_class, get_algo_class, get_chan_num, collect_eval_metrics, \
    combine_entities_eval_metrics, get_dynamic_scores, get_gaussian_kernel_scores
from src.evaluation.evaluation_utils import fit_distributions, get_scores_channelwise
from src.algorithms.algorithm_utils import load_torch_algo
from src.evaluation.trainer import Trainer


def evaluate(saved_model_root, logger, thres_methods=["top_k_time", "best_f1_test"], eval_root_cause=True,
             point_adjust=False, eval_R_model=True, eval_dyn=False, thres_config=None,
             telem_only=True, make_plots=["prc", "score_t"], composite_best_f1=False):
    seed = 42
    saved_model_folders = os.listdir(saved_model_root)
    saved_model_folders.sort()  # Sort directories in alphabetical order
    plots_dir = os.path.join(saved_model_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize dictionary structure to collect results from each entity
    algo_results = {"hr_100_all": [], "hr_150_all": [], "rc_top3_all": [], "val_recons_err": [], "val_loss": [],
                    "std_scores_train": [], "auroc": [], "avg_prec": []}
    eval_methods = ["time-wise"]
    for thres_method in thres_methods + ["tail_prob", "pot"]:
        algo_results[thres_method] = {"hr_100_tp": [], "hr_150_tp": [], "rc_top3_tp": [], "opt_thres": [],
                                      "fscore_comp": [], "rec_e": []}
        for eval_method in eval_methods:
            algo_results[thres_method][eval_method] = {"tp": [], "fp": [], "fn": []}

    algo_R_results = copy.deepcopy(algo_results)
    telemanom_gauss_s_results = copy.deepcopy(algo_results)
    algo_dyn_results = copy.deepcopy(algo_results)
    algo_dyn_gauss_conv_results = copy.deepcopy(algo_results)
    path_decomposition = os.path.normpath(saved_model_root).split(os.path.sep)
    algo_name = path_decomposition[-3]
    me_ds_name = path_decomposition[-4].split("_me")[0]
    ds_class = get_dataset_class(me_ds_name)

    rca_possible = eval_root_cause
    thres_config_dict = None

    for folder_name in saved_model_folders:
        if ".ini" in folder_name or ".csv" in folder_name or ".pdf" in folder_name:
            continue
        elif os.path.split(folder_name)[-1] == "plots":
            continue

        # get dataset
        entity = os.path.split(folder_name)[-1]
        if me_ds_name in ["msl", "smd", "smap"]:
            entity = entity.split("-", 1)[1]
        ds_init_params = {"seed": seed, "entity": entity}
        if me_ds_name == "swat-long":
            ds_init_params["shorten_long"] = False
        if me_ds_name == "damadics-s":
            ds_init_params["drop_init_test"] = True
        dataset = ds_class(**ds_init_params)
        plots_name = os.path.join(plots_dir, algo_name + "_" + me_ds_name + "_" + entity + "_")
        # ds_name = dataset.name
        logger.info("Processing Folder name: {}, {} on me dataset {}, entity {}".format(folder_name, algo_name,
                                                                                  me_ds_name, entity))
        if thres_config is not None:
            thres_config_dict = thres_config(me_ds_name)
        else:
            thres_config_dict = default_thres_config

        # get test scores from pkl
        raw_preds_file = os.path.join(saved_model_root, folder_name, "raw_predictions")
        try:
            with open(raw_preds_file, 'rb') as file:
                preds = pickle.load(file)
        except:
            logger.info("The raw predictions of %s on %s weren't found, this run can't be evaluated" % (algo_name,
                                                                                                  me_ds_name))
            return None

        # Get the true labels
        _, _, _, y_test = dataset.data()

        true_events = get_events(y_test)

        root_causes = None
        if eval_root_cause:
            root_causes = dataset.get_root_causes()
        # Flag that indicates root cause identification evaluation is possible
        rca_possible = eval_root_cause and (preds["score_tc_test"] is not None or preds["error_tc_test"] is not None) \
                       and root_causes is not None

        # Load the predictions
        score_t_test = preds["score_t_test"]
        score_tc_test = preds["score_tc_test"]

        error_tc_test = preds["error_tc_test"]
        error_t_test = preds["error_t_test"]

        score_t_train = preds["score_t_train"]
        score_tc_train = preds["score_tc_train"]
        error_tc_train = preds["error_tc_train"]
        error_t_train = preds["error_t_train"]

        recons_tc_train = preds["recons_tc_train"]
        recons_tc_test = preds["recons_tc_test"]

        try:
            val_recons_err = np.nanmean(preds["val_recons_err"])
        except:
            val_recons_err = None

        try:
            val_loss = preds["val_loss"]
        except:
            val_loss = None

        if telem_only and me_ds_name in ["msl", "smap"]:
            if error_tc_train is not None:
                error_t_train = error_tc_train[:, 0]
                error_tc_train = None
            if error_tc_test is not None:
                error_t_test = error_tc_test[:, 0]
                error_tc_test = None
            if score_tc_test is not None:
                score_t_test = score_tc_test[:, 0]
                score_tc_test = None

        eval_R = eval_R_model and (error_tc_test is not None or error_t_test is not None)
        eval_dyn = eval_dyn and ((error_tc_test is not None) or
                                 (error_t_test is not None))

        # Evaluate on each entity
        logger.info("Evaluating for score_t")
        algo_results = collect_eval_metrics(algo_results=algo_results, score_t_test=score_t_test, y_test=y_test,
                                            thres_methods=thres_methods, logger=logger, true_events=true_events,
                                            rca_possible=rca_possible and (preds["score_tc_test"] is not None),
                                            score_tc_test=score_tc_test,
                                            root_causes=root_causes, score_t_train=score_t_train,
                                            point_adjust=point_adjust, thres_config_dict=thres_config_dict,
                                            eval_methods=eval_methods, make_plots=make_plots, dataset=dataset,
                                            plots_name=plots_name + "base", composite_best_f1=composite_best_f1)

        algo_results["val_recons_err"].append(val_recons_err)
        algo_results["val_loss"].append(val_loss)
        algo_results["std_scores_train"].append(np.std(score_t_train))

        if eval_R:
            if algo_name == "TelemanomAlgo":
                logger.info("Evaluating for static gaussian for TelemanomAlgo")
                # get static gaussian scores. This is usually done in the trainer, but not for this algo
                distr_names = ["univar_gaussian"]
                distr_par_file = os.path.join(saved_model_root, folder_name, "distr_parameters")
                if error_t_train is None or error_tc_train is None:
                    score_t_test_gauss_s = error_t_test
                    score_t_train_gauss_s = None
                    score_tc_test_gauss_s = error_tc_test
                else:
                    distr_params = fit_distributions(distr_par_file, distr_names, predictions_dic=
                    {"train_raw_scores": error_tc_train})[distr_names[0]]
                    score_t_train_gauss_s, _, score_t_test_gauss_s, score_tc_train_gauss_s, _, score_tc_test_gauss_s = \
                        get_scores_channelwise(distr_params, train_raw_scores=error_tc_train,
                                               val_raw_scores=None, test_raw_scores=error_tc_test,
                                               logcdf=True)

                telemanom_gauss_s_results = collect_eval_metrics(algo_results=telemanom_gauss_s_results,
                                                                 score_t_test=score_t_test_gauss_s,
                                                                 y_test=y_test,
                                                                 thres_methods=thres_methods,
                                                                 logger=logger,
                                                                 true_events=true_events,
                                                                 rca_possible=rca_possible,
                                                                 score_tc_test=score_tc_test_gauss_s,
                                                                 root_causes=root_causes,
                                                                 score_t_train=score_t_train_gauss_s,
                                                                 point_adjust=point_adjust,
                                                                 thres_config_dict=thres_config_dict,
                                                                 eval_methods=eval_methods,
                                                                 make_plots=make_plots,
                                                                 dataset=dataset,
                                                                 plots_name=plots_name + "-gauss-s",
                                                                 composite_best_f1=composite_best_f1)

            if error_tc_train is not None and error_tc_test is not None:
                logger.info("Doing mean adjustment of train and test error_tc")
                mean_c_train = np.mean(error_tc_train, axis=0)
                error_tc_train_normed = error_tc_train - mean_c_train
                error_tc_test_normed = error_tc_test - mean_c_train
                error_t_train_normed = np.sqrt(np.mean(error_tc_train_normed ** 2, axis=1))
                error_t_test_normed = np.sqrt(np.mean(error_tc_test_normed ** 2, axis=1))
            else:
                error_t_test_normed = error_t_test
                error_t_train_normed = error_t_train
                error_tc_test_normed = error_tc_test
                error_tc_train_normed = None
            logger.info("Evaluating for error_t")
            algo_R_results = collect_eval_metrics(algo_results=algo_R_results, score_t_test=error_t_test_normed,
                                                  y_test=y_test,
                                                  thres_methods=thres_methods, logger=logger, true_events=true_events,
                                                  rca_possible=rca_possible, score_tc_test=error_tc_test_normed,
                                                  root_causes=root_causes, score_t_train=error_t_train_normed,
                                                  point_adjust=point_adjust,
                                                  thres_config_dict=thres_config_dict, eval_methods=eval_methods,
                                                  make_plots=make_plots, dataset=dataset,
                                                  plots_name=plots_name + "R",
                                                  composite_best_f1=composite_best_f1,
                                                  score_tc_train=error_tc_train_normed)
            algo_R_results["val_recons_err"].append(val_recons_err)
            algo_R_results["val_loss"].append(val_loss)
            if error_t_train is not None:
                algo_R_results["std_scores_train"].append(np.std(error_t_train))

            # dynamic scoring function
        if eval_dyn:
            # dyn_thres_methods = ["best_f1_test"]
            dyn_thres_methods = thres_methods
            logger.info("Evaluating gaussian dynamic scoring for error_t with thres_methods {}".format(dyn_thres_methods))
            long_window = thres_config_dict["dyn_gauss"]["long_window"]
            short_window = thres_config_dict["dyn_gauss"]["short_window"]
            if telem_only and me_ds_name in ["msl", "smap"]:
                score_t_test_dyn, score_tc_test_dyn, score_t_train_dyn, score_tc_train_dyn = get_dynamic_scores(
                    error_tc_train=None, error_tc_test=None, error_t_train=error_t_train, error_t_test=error_t_test,
                    long_window=long_window, short_window=short_window)
            else:
                score_t_test_dyn, score_tc_test_dyn, score_t_train_dyn, score_tc_train_dyn = get_dynamic_scores(
                    error_tc_train, error_tc_test, error_t_train, error_t_test, long_window=long_window,
                    short_window=short_window)
            algo_dyn_results = collect_eval_metrics(algo_results=algo_dyn_results, score_t_test=score_t_test_dyn,
                                                    y_test=y_test, thres_methods=dyn_thres_methods,
                                                    logger=logger, rca_possible=rca_possible, true_events=true_events,
                                                    score_tc_test=score_tc_test_dyn, root_causes=root_causes,
                                                    score_t_train=score_t_train_dyn, point_adjust=point_adjust,
                                                    thres_config_dict=thres_config_dict, eval_methods=eval_methods,
                                                    make_plots=make_plots, dataset=dataset,
                                                    plots_name=plots_name + "dyn",
                                                    composite_best_f1=composite_best_f1,
                                                    score_tc_train=score_tc_train_dyn)
            algo_dyn_results["val_recons_err"].append(val_recons_err)
            algo_dyn_results["val_loss"].append(val_loss)
            if score_t_train_dyn is not None:
                algo_dyn_results["std_scores_train"].append(np.std(score_t_train_dyn))

            kernel_sigma = thres_config_dict["dyn_gauss"]["kernel_sigma"]
            score_t_test_dyn_gauss_conv, score_tc_test_dyn_gauss_conv = get_gaussian_kernel_scores(
                score_t_test_dyn, score_tc_test_dyn, kernel_sigma)
            if score_t_train_dyn is not None:
                score_t_train_dyn_gauss_conv, _ = get_gaussian_kernel_scores(score_t_train_dyn, score_tc_train_dyn,
                                                                             kernel_sigma)
            else:
                score_t_train_dyn_gauss_conv = None
            algo_dyn_gauss_conv_results = collect_eval_metrics(algo_results=algo_dyn_gauss_conv_results,
                                                               score_t_test=score_t_test_dyn_gauss_conv,
                                                               y_test=y_test,
                                                               thres_methods=dyn_thres_methods,
                                                               logger=logger,
                                                               rca_possible=rca_possible,
                                                               true_events=true_events,
                                                               score_tc_test=score_tc_test_dyn_gauss_conv,
                                                               root_causes=root_causes,
                                                               score_t_train=score_t_train_dyn_gauss_conv,
                                                               point_adjust=point_adjust,
                                                               thres_config_dict=thres_config_dict,
                                                               eval_methods=eval_methods,
                                                               make_plots=make_plots, dataset=dataset,
                                                               plots_name=plots_name + "dyn-gauss-conv",
                                                               composite_best_f1=composite_best_f1,
                                                               score_tc_train=None)

    # Combine results from each entity
    final_results, column_names = combine_entities_eval_metrics(algo_results, thres_methods, me_ds_name, algo_name,
                                                                rca_possible, eval_methods=eval_methods)
    if eval_R_model:
        results_R, _ = combine_entities_eval_metrics(algo_R_results, thres_methods, me_ds_name, algo_name + "-R",
                                                     rca_possible, eval_methods=eval_methods)
        final_results = np.concatenate((final_results, results_R), axis=0)

        if algo_name == "TelemanomAlgo":
            results_telem, _ = combine_entities_eval_metrics(telemanom_gauss_s_results, thres_methods, me_ds_name,
                                                             algo_name + "-Gauss-S", rca_possible, eval_methods=eval_methods)
            final_results = np.concatenate((final_results, results_telem), axis=0)

    if eval_dyn:
        results_dyn, _ = combine_entities_eval_metrics(algo_dyn_results, dyn_thres_methods,
                                                       me_ds_name, algo_name + "-dyn",
                                                    rca_possible, eval_methods=eval_methods)
        final_results = np.concatenate((final_results, results_dyn), axis=0)
        results_dyn_gauss_conv, _ = combine_entities_eval_metrics(algo_dyn_gauss_conv_results,
                                                                  dyn_thres_methods,
                                                                  me_ds_name, algo_name + "-dyn-gauss-conv",
                                                                  rca_possible, eval_methods=eval_methods)
        final_results = np.concatenate((final_results, results_dyn_gauss_conv), axis=0)

    results_df = pd.DataFrame(final_results, columns=column_names)
    results_df["folder_name"] = saved_model_root
    new_col_order = list(results_df.columns)[:3] + ["point_adjust"] + list(results_df.columns)[3:]
    results_df["point_adjust"] = point_adjust
    results_df = results_df[new_col_order]
    with open(os.path.join(os.path.dirname(saved_model_root), "config.json")) as file:
        algo_config = json.dumps(json.load(file))
    results_df["config"] = algo_config
    if thres_config_dict is not None:
        results_df["thres_config"] = str(thres_config_dict)
    if point_adjust:
        filename = os.path.join(saved_model_root, "results_point_adjust.csv")
    else:
        filename = os.path.join(saved_model_root, "results.csv")

    results_df.to_csv(filename, index=False)
    logger.info("Saved results to {}".format(filename))


def analyse_from_pkls(results_root:str, thres_methods=["best_f1_test"], eval_root_cause=True, point_adjust=False,
                      eval_R_model=True, eval_dyn=True, thres_config=None, logger=None,
                      telem_only=True, filename_prefix="", rerun_if_ds=None, process_seeds=None, make_plots=[],
                      composite_best_f1=False):
    """
    Function that reads saved predictions and evaluates them for anomaly detection and diagnosis under various
    settings.
    :param results_root: dir where predictions for the algo generated by the trainer in a specific folder structure.
    :param thres_methods: list of thresholding methods with which to evaluate
    :param eval_root_cause: Set it to True if root cause is desired and possible (i.e. if channel-wise scores are provided
    in the predictions, else False
    :param point_adjust: True if point-adjusted evaluation is desired.
    :param eval_R_model: Corresponds to using Errors scoring function. Set it to True only if errors_t or errors_tc are
    avaiable in the predictions. Pre-requisite to be True for eval_dyn.
    :param eval_dyn: Set it to True if Gauss-D and Gauss-D-K scoring function evaluation is desired. Needs eval_R_model
    to be True.
    :param thres_config: A function that takes the dataset name as input and returns a dictionary corresponding to the
    config for each method in thres_methods.
    :param logger: for logging.
    :param telem_only: Only affects the evaluation for MSL and SMAP datasets. If set to True, only the sensor channel,
    i.e. first channel will be used in evaluation. If False, all channels - sensors and commands will be used.
    :param filename_prefix: desired prefix on the filename.
    :param process_seeds: specify a list if only some of the seeds need to be analyzed. If None, all seeds for which
    results.csv doesn't exist will be (re)analyzed.
    :param rerun_if_ds: set the names of specific datasets for which (re)analysis is required. Otherwise only datasets
    for which results.csv doesn't exist will be (re)analyzed.
    :param make_plots: specify which plots are desired. ["prc", "score_t"] are implemented.
    :param composite_best_f1: if set to True, the "best-f1" threshold will be computed as "best-fc1" threshold.
    :return: None. The evalution results are saved as results.csv for each run.
    """
    seed = 42
    result_df_list = []
    ds_folders = os.listdir(results_root)

    if point_adjust:
        result_filename = "results_point_adjust.csv"
    else:
        result_filename = "results.csv"
    if logger is None:
        init_logging(os.path.join(results_root, 'logs'), prefix="eval")
        logger = logging.getLogger(__name__)
    for ds_folder in ds_folders:
        if ds_folder.endswith(".csv") or ds_folder == "logs" or "thres_results" in ds_folder:
            continue
        ds_path = os.path.join(results_root, ds_folder)
        algo_folders = os.listdir(ds_path)
        for algo_folder in algo_folders:
            algo_path = os.path.join(ds_path, algo_folder)
            config_folders = os.listdir(algo_path)
            config_folders = [folder for folder in config_folders if not folder.endswith(".csv")]
            for config_folder in config_folders:
                config_path = os.path.join(algo_path, config_folder)
                run_folders = os.listdir(config_path)
                for run_folder in run_folders:
                    if not run_folder.endswith(".json"):
                        run_path = os.path.join(config_path, run_folder)
                        current_seed = int(run_folder.split("-", 2)[0])
                        if process_seeds is not None:
                            if current_seed not in process_seeds:
                                continue
                        if rerun_if_ds is not None:
                            if (ds_folder in rerun_if_ds) or (rerun_if_ds == 'all'):
                                if os.path.exists(os.path.join(run_path, result_filename)):
                                    os.remove(os.path.join(run_path, result_filename))
                        if not os.path.exists(os.path.join(run_path, result_filename)):
                            entity_folders = os.listdir(run_path)
                            skip_this_run = False  # Flag to indicate that evaluation for this run is impossible
                            for entity_folder in entity_folders:
                                if entity_folder != "plots" and ".csv" not in entity_folder and entity_folder != "logs":
                                    entity_path = os.path.join(run_path, entity_folder)
                                    if not skip_this_run:
                                        try:
                                            with open(os.path.join(entity_path, "raw_predictions"), "rb") as file:
                                                raw_predictions = pickle.load(file)
                                            assert "score_t_test" in raw_predictions.keys()
                                            if np.isnan(raw_predictions["score_t_test"]).any():
                                                skip_this_run = True
                                        except:
                                            me_ds_name = ds_folder.split("_me")[0]
                                            ds_class = get_dataset_class(me_ds_name)
                                            algo_class = get_algo_class(algo_folder)
                                            entity_name = entity_folder.replace("smap-", "").replace("msl-", "").\
                                                replace("smd-", "")
                                            ds_init_params = {"seed": seed, "entity": entity_name}
                                            if me_ds_name == "swat-long":
                                                ds_init_params["shorten_long"] = False
                                            entity_ds = ds_class(**ds_init_params)
                                            repredict = repredict_from_saved_model(entity_path, algo_class=algo_class,
                                                                                   entity=entity_ds, logger=logger)
                                            if not repredict:
                                                logger.warning("Predictions and trained model couldn't be found, evaluation is "
                                                      "impossible for run saved at %s" % run_path)
                                                skip_this_run = True
                            if not skip_this_run:
                                evaluate(run_path, thres_methods=thres_methods, eval_root_cause=eval_root_cause,
                                         point_adjust=point_adjust, eval_R_model=eval_R_model, eval_dyn=eval_dyn,
                                         thres_config=thres_config, logger=logger, telem_only=telem_only,
                                         make_plots=make_plots, composite_best_f1=composite_best_f1)
                        try:
                            result_df = pd.read_csv(os.path.join(run_path, result_filename))
                            if "point_adjust" not in result_df.columns:
                                result_df["point_adjust"] = False
                            result_df_list.append(result_df)
                        except:
                            logger.warning("Results table couldn't be found for run saved at %s" % run_path)
    overall_results = pd.concat(result_df_list, ignore_index=True)

    overall_results.to_csv(os.path.join(results_root, filename_prefix+"overall_" + result_filename))


def repredict_from_saved_model(model_root, algo_class, entity, logger):
    algo_config_filename = os.path.join(model_root, "init_params")
    saved_model_filename = [os.path.join(model_root, filename) for filename in
                            os.listdir(model_root) if "trained_model" in filename]
    if len(saved_model_filename) == 1:
        saved_model_filename = saved_model_filename[0]
    else:
        saved_model_filename.sort(key=get_chan_num)

    additional_params_filename = os.path.join(model_root, "additional_params")
    if len(additional_params_filename) == 1:
        additional_params_filename = additional_params_filename[0]
    try:
        algo_reload = load_torch_algo(algo_class, algo_config_filename, saved_model_filename,
                                      additional_params_filename, eval=True)
        _ = Trainer.predict(algo_reload, entity, model_root, logger=logger)
        return True
    except Exception as e:
        logger.warning(f"An error occurred while loading saved algo and repredicting: {e}")
        return False


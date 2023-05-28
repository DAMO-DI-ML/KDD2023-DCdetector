from fc_score import *
from f1_score_f1_pa import *
from evaluate_utils import *

default_thres_config = {"top_k_time": {},
                        "best_f1_test": {"exact_pt_adj": True},
                        "thresholded_score": {},
                        "tail_prob": {"tail_prob": 2},
                        "tail_prob_1": {"tail_prob": 1},
                        "tail_prob_2": {"tail_prob": 2},
                        "tail_prob_3": {"tail_prob": 3},
                        "tail_prob_4": {"tail_prob": 4},
                        "tail_prob_5": {"tail_prob": 5},
                        "dyn_gauss": {"long_window": 10000, "short_window": 1, "kernel_sigma": 10},
                        "nasa_npt": {"batch_size": 70, "window_size": 30, "telem_only": True,
                                     "smoothing_perc": 0.005, "l_s": 250, "error_buffer": 5, "p": 0.05}}


def threshold_and_predict(score_t_test, y_test, true_events, logger, test_anom_frac, thres_method="top_k_time",
                          point_adjust=False, score_t_train=None, thres_config_dict=dict(), return_auc=False,
                          composite_best_f1=False):
    if thres_method in thres_config_dict.keys():
        config = thres_config_dict[thres_method]
    else:
        config = default_thres_config[thres_method]
    # test_anom_frac = (np.sum(y_test)) / len(y_test)
    auroc = None
    avg_prec = None
    if thres_method == "thresholded_score":
        opt_thres = 0.5
        if set(score_t_test) - {0, 1}:
            logger.error("Score_t_test isn't binary. Predicting all as non-anomalous")
            pred_labels = np.zeros(len(score_t_test))
        else:
            pred_labels = score_t_test

    elif thres_method == "best_f1_test" and point_adjust:
        prec, rec, thresholds = precision_recall_curve(y_test, score_t_test, pos_label=1)
        if not config["exact_pt_adj"]:
            fscore_best_time = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
            opt_num = np.squeeze(np.argmax(fscore_best_time))
            opt_thres = thresholds[opt_num]
            thresholds = np.random.choice(thresholds, size=5000) + [opt_thres]
        fscores = []
        for thres in thresholds:
            _, _, _, _, _, fscore = get_point_adjust_scores(y_test, score_t_test > thres, true_events)
            fscores.append(fscore)
        opt_thres = thresholds[np.argmax(fscores)]
        pred_labels = score_t_test > opt_thres

    elif thres_method == "best_f1_test" and composite_best_f1:
        prec, rec, thresholds = precision_recall_curve(y_test, score_t_test, pos_label=1)
        precs_t = prec
        fscores_c = [get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t) for thres, prec_t in
                     zip(thresholds, precs_t)]
        try:
            opt_thres = thresholds[np.nanargmax(fscores_c)]
        except:
            opt_thres = 0.0
        pred_labels = score_t_test > opt_thres

    elif thres_method == "top_k_time":
        opt_thres = np.nanpercentile(score_t_test, 100 * (1 - test_anom_frac), interpolation='higher')
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif thres_method == "best_f1_test":
        prec, rec, thres = precision_recall_curve(y_test, score_t_test, pos_label=1)
        fscore = [get_f_score(precision, recall) for precision, recall in zip(prec, rec)]
        opt_num = np.squeeze(np.argmax(fscore))
        opt_thres = thres[opt_num]
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif "tail_prob" in thres_method:
        tail_neg_log_prob = config["tail_prob"]
        opt_thres = tail_neg_log_prob
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif thres_method == "nasa_npt":
        opt_thres = 0.5
        pred_labels = get_npt_labels(score_t_test, y_test, config)
    else:
        logger.error("Thresholding method {} not in [top_k_time, best_f1_test, tail_prob]".format(thres_method))
        return None, None
    if return_auc:
        avg_prec = average_precision_score(y_test, score_t_test)
        auroc = roc_auc_score(y_test, score_t_test)
        return opt_thres, pred_labels, avg_prec, auroc
    return opt_thres, pred_labels


# most-top funcion
def evaluate_predicted_labels(pred_labels, y_test, true_events, logger, eval_method="time-wise", breaks=[],
                              point_adjust=False):
    """
    Computes evaluation metrics for the binary classifications given the true and predicted labels
    :param point_adjust: used to judge whether is pa
    :param pred_labels: array of predicted labels
    :param y_test: array of true labels
    :param eval_method: string that indicates whether we evaluate the classification time point-wise or event-wise
    :param breaks: array of discontinuities in the time series, relevant only if you look at event-wise
    :param return_raw: Boolean that indicates whether we want to return tp, fp and fn or prec, recall and f1
    :return: tuple of evaluation metrics
    """

    if eval_method == "time-wise":
        # point-adjust fscore
        if point_adjust:
            fp, fn, tp, prec, rec, fscore = get_point_adjust_scores(y_test, pred_labels, true_events)
        # normal fscore
        else:
            _, prec, rec, fscore, _ = get_accuracy_precision_recall_fscore(y_test, pred_labels)
            tp = np.sum(pred_labels * y_test)
            fp = np.sum(pred_labels) - tp
            fn = np.sum(y_test) - tp
    # event-wise
    else:
        logger.error("Evaluation method {} not in [time-wise, event-wise]".format(eval_method))
        return 0, 0, 0

    return tp, fp, fn, prec, rec, fscore

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, \
    accuracy_score, fbeta_score, average_precision_score


# function: calculate the point-adjust f-scores(whether top k)
def get_point_adjust_scores(y_test, pred_labels, true_events, thereshold_k=0, whether_top_k=False):
    tp = 0
    fn = 0
    for true_event in true_events.keys():
        true_start, true_end = true_events[true_event]
        if whether_top_k is False:
            if pred_labels[true_start:true_end].sum() > 0:
                tp += (true_end - true_start)
            else:
                fn += (true_end - true_start)
        else:
            if pred_labels[true_start:true_end].sum() > thereshold_k:
                tp += (true_end - true_start)
            else:
                fn += (true_end - true_start)
    fp = np.sum(pred_labels) - np.sum(pred_labels * y_test)

    prec, rec, fscore = get_prec_rec_fscore(tp, fp, fn)
    return fp, fn, tp, prec, rec, fscore

def get_adjust_F1PA(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
            
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    return accuracy, precision, recall, f_score


# calculate the point-adjusted f-score
def get_prec_rec_fscore(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    fscore = get_f_score(precision, recall)
    return precision, recall, fscore


def get_f_score(prec, rec):
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec)
    return f_score


# function: calculate the normal edition f-scores
def get_accuracy_precision_recall_fscore(y_true: list, y_pred: list):
    accuracy = accuracy_score(y_true, y_pred)
    # warn_for=() avoids log warnings for any result being zero
    # precision, recall, f_score, _ = prf(y_true, y_pred, average='binary', warn_for=())
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f_score = (2 * precision * recall) / (precision + recall)
    if precision == 0 and recall == 0:
        f05_score = 0
    else:
        f05_score = fbeta_score(y_true, y_pred, average='binary', beta=0.5)
    return accuracy, precision, recall, f_score, f05_score



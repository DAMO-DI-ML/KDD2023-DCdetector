# used by paper: TSB-UAD as the main evaluator
# github: https://github.com/johnpaparrizos/TSB-UAD/blob/main/TSB_AD/utils/metrics.py
import numpy as np
from sklearn import metrics
from metrics.evaluate_utils import find_length,range_convers_new


def extend_postive_range(x, window=16):
    label = x.copy().astype(float)
#     print(label)
    L = range_convers_new(label)  # index of non-zero segments
#     print(L)
    length = len(label)
    for k in range(len(L)):
        s = L[k][0]
        e = L[k][1]
        # x1 is the extended list like [1,2,3] which are non-zero(from the end-e)
        x1 = np.arange(e, min(e + window // 2, length))
        label[x1] += np.sqrt(1 - (x1 - e) / (window))
        # before the start-s
        x2 = np.arange(max(s - window // 2, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / (window))

    label = np.minimum(np.ones(length), label)
    return label


def extend_postive_range_individual(x, percentage=0.2):
    label = x.copy().astype(float)
    L = range_convers_new(label)  # index of non-zero segments
    length = len(label)
    for k in range(len(L)):
        s = L[k][0]
        e = L[k][1]

        l0 = int((e - s + 1) * percentage)

        x1 = np.arange(e, min(e + l0, length))
        label[x1] += np.sqrt(1 - (x1 - e) / (2 * l0))

        x2 = np.arange(max(s - l0, 0), s)
        label[x2] += np.sqrt(1 - (s - x2) / (2 * l0))

    label = np.minimum(np.ones(length), label)
    return label


def TPR_FPR_RangeAUC(labels, pred, P, L):
    product = labels * pred

    TP = np.sum(product)

    # recall = min(TP/P,1)
    P_new = (P + np.sum(labels)) / 2  # so TPR is neither large nor small
    # P_new = np.sum(labels)
    recall = min(TP / P_new, 1)
    # recall = TP/np.sum(labels)
    # print('recall '+str(recall))

    existence = 0
    for seg in L:
        if np.sum(product[seg[0]:(seg[1] + 1)]) > 0:
            existence += 1

    existence_ratio = existence / len(L)
    # print(existence_ratio)

    # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
    # print(existence_ratio)
    TPR_RangeAUC = recall * existence_ratio

    FP = np.sum(pred) - TP
    # TN = np.sum((1-pred) * (1-labels))

    # FPR_RangeAUC = FP/(FP+TN)
    N_new = len(labels) - P_new
    FPR_RangeAUC = FP / N_new

    Precision_RangeAUC = TP / np.sum(pred)

    return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC


def Range_AUC(score_t_test, y_test,  window=5, percentage=0, plot_ROC=False, AUC_type='window'):
    # AUC_type='window'/'percentage'
    score = score_t_test
    labels = y_test
    score_sorted = -np.sort(-score)

    P = np.sum(labels)
    # print(np.sum(labels))
    if AUC_type == 'window':
        labels = extend_postive_range(labels, window=window)
    else:
        labels = extend_postive_range_individual(labels, percentage=percentage)

    # print(np.sum(labels))
    L = range_convers_new(labels)
    TPR_list = [0]
    FPR_list = [0]
    Precision_list = [1]

    for i in np.linspace(0, len(score) - 1, 250).astype(int):
        threshold = score_sorted[i]
        # print('thre='+str(threshold))
        pred = score >= threshold
        TPR, FPR, Precision = TPR_FPR_RangeAUC(labels, pred, P, L)

        TPR_list.append(TPR)
        FPR_list.append(FPR)
        Precision_list.append(Precision)

    TPR_list.append(1)
    FPR_list.append(1)  # otherwise, range-AUC will stop earlier than (1,1)

    tpr = np.array(TPR_list)
    fpr = np.array(FPR_list)
    prec = np.array(Precision_list)

    width = fpr[1:] - fpr[:-1]
    height = (tpr[1:] + tpr[:-1]) / 2
    AUC_range = np.sum(width * height)

    width_PR = tpr[1:-1] - tpr[:-2]
    height_PR = (prec[1:] + prec[:-1]) / 2
    AP_range = np.sum(width_PR * height_PR)

    if plot_ROC:
        return AUC_range, AP_range, fpr, tpr, prec

    return AUC_range


def point_wise_AUC(score_t_test, y_test,  plot_ROC=False):
    # area under curve
    label = y_test
    score = score_t_test
    auc = metrics.roc_auc_score(label, score)
    # plor ROC curve
    if plot_ROC:
        fpr, tpr, thresholds = metrics.roc_curve(label, score)
        # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
        # display.plot()
        return auc, fpr, tpr
    else:
        return auc


def main():
    y_test = np.zeros(100)
    y_test[10:20] = 1
    y_test[50:60] = 1
    pred_labels = np.zeros(100)
    pred_labels[15:17] = 0.5
    pred_labels[55:62] = 0.7
    # pred_labels[51:55] = 1
    # true_events = get_events(y_test)
    point_auc = point_wise_AUC(pred_labels, y_test)
    range_auc = Range_AUC(pred_labels, y_test)
    print("point_auc: {}, range_auc: {}".format(point_auc, range_auc))


if __name__ == "__main__":
    main()
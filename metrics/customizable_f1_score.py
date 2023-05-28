# used by paper: Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series_VLDB 2021
# github: https://github.com/exathlonbenchmark/exathlon
import numpy as np
from metrics.evaluate_utils import range_convers_new

# the existence reward on the bias
def b(bias, i, length):
    if bias == 'flat':
        return 1
    elif bias == 'front-end bias':
        return length - i + 1
    elif bias == 'back-end bias':
        return i
    else:
        if i <= length / 2:
            return i
        else:
            return length - i + 1


def w(AnomalyRange, p):
    MyValue = 0
    MaxValue = 0
    start = AnomalyRange[0]
    AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
    # flat/'front-end bias'/'back-end bias'
    bias = 'flat'
    for i in range(start, start + AnomalyLength):
        bi = b(bias, i, AnomalyLength)
        MaxValue += bi
        if i in p:
            MyValue += bi
    return MyValue / MaxValue


def Cardinality_factor(Anomolyrange, Prange):
    score = 0
    start = Anomolyrange[0]
    end = Anomolyrange[1]
    for i in Prange:
        if start <= i[0] <= end:
            score += 1
        elif i[0] <= start <= i[1]:
            score += 1
        elif i[0] <= end <= i[1]:
            score += 1
        elif start >= i[0] and end <= i[1]:
            score += 1
    if score == 0:
        return 0
    else:
        return 1 / score


def existence_reward(labels, preds):
    '''
    labels: list of ordered pair
    preds predicted data
    '''

    score = 0
    for i in labels:
        if np.sum(np.multiply(preds <= i[1], preds >= i[0])) > 0:
            score += 1
    return score


def range_recall_new(labels, preds, alpha):
    p = np.where(preds == 1)[0]  # positions of predicted label==1
    range_pred = range_convers_new(preds)
    range_label = range_convers_new(labels)

    Nr = len(range_label)  # total # of real anomaly segments

    ExistenceReward = existence_reward(range_label, p)

    OverlapReward = 0
    for i in range_label:
        OverlapReward += w(i, p) * Cardinality_factor(i, range_pred)

    score = alpha * ExistenceReward + (1 - alpha) * OverlapReward
    if Nr != 0:
        return score / Nr, ExistenceReward / Nr, OverlapReward / Nr
    else:
        return 0, 0, 0


def customizable_f1_score(y_test, pred_labels,  alpha=0.2):
    label = y_test
    preds = pred_labels
    Rrecall, ExistenceReward, OverlapReward = range_recall_new(label, preds, alpha)
    Rprecision = range_recall_new(preds, label, 0)[0]

    if Rprecision + Rrecall == 0:
        Rf = 0
    else:
        Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)
    return Rf


def main():
    y_test = np.zeros(100)
    y_test[10:20] = 1
    y_test[50:60] = 1
    pred_labels = np.zeros(100)
    pred_labels[15:19] = 1
    pred_labels[55:62] = 1
    # pred_labels[51:55] = 1
    # true_events = get_events(y_test)
    Rf = customizable_f1_score(y_test, pred_labels)
    print("Rf: {}".format(Rf))


if __name__ == "__main__":
    main()
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema


def get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t, return_prec_rec=False):
    pred_labels = score_t_test > thres
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c


class NptConfig:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

def find_length(data):
    if len(data.shape) > 1:
        return 0
    data = data[:min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max] + base
    except:
        return 125


def range_convers_new(label):
    '''
    input: arrays of binary values
    output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
    '''
    L = []
    i = 0
    j = 0
    while j < len(label):
        while label[i] == 0:
            i += 1
            if i >= len(label):
                break
        j = i + 1
        if j >= len(label):
            if j == len(label):
                L.append((i, j - 1))
            break
        while label[j] != 0:
            j += 1
            if j >= len(label):
                L.append((i, j - 1))
                break
        if j >= len(label):
            break
        L.append((i, j - 1))
        i = j
    return L
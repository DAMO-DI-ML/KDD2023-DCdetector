import numpy as np
from sklearn.metrics import precision_score


def get_events(y_test, outlier=1, normal=0):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
        else:
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events


def get_composite_fscore_raw(y_test, pred_labels,  true_events, return_prec_rec=False):
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn)
    prec_t = precision_score(y_test, pred_labels)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c


def main():
    y_test = np.zeros(100)
    y_test[10:20] = 1
    y_test[50:60] = 1
    pred_labels = np.zeros(100)
    pred_labels[15:17] = 1
    pred_labels[55:62] = 1
    # pred_labels[51:55] = 1
    # true_events = get_events(y_test)
    prec_t, rec_e, fscore_c = get_composite_fscore_raw(pred_labels, y_test, return_prec_rec=True)
#     print("Prec_t: {}, rec_e: {}, fscore_c: {}".format(prec_t, rec_e, fscore_c))


if __name__ == "__main__":
    main()

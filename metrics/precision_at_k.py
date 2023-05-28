# k is defined as the number of anomalies
# only calculate the range top k not the whole set
import numpy as np


def precision_at_k(y_test, score_t_test, pred_labels):
    # top-k
    k = int(np.sum(y_test))
    threshold = np.percentile(score_t_test, 100 * (1 - k / len(y_test)))

    # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
    p_at_k = np.where(pred_labels > threshold)[0]
    TP_at_k = sum(y_test[p_at_k])
    precision_at_k = TP_at_k / k
    return precision_at_k

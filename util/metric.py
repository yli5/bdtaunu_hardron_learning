# This script defines functions to compute performance metrics 
import numpy as np 
from sklearn.metrics import roc_auc_score

def accuracy(y_pred, y_true, pos_label=1, neg_label=0,
                             threshold=0.5):
    """
    Classification accuracy

    y_pred: 1-D np.array, predicted output
    y_true: 1-D np.array, true label of output
    pos_label: notation of positive label, default 1.
    neg_label: notation of negative label, default 0.
    threshold: a real value between [neg_label, pos_label] to 
               make classification decision. 

    """
    y_pred_binary = [pos_label if i > threshold else neg_label 
                     for i in y_pred]
    return np.sum(y_pred_binary == y_true) * 1.0 / len(y_pred)


def precision(y_pred, y_true, pos_label=1, neg_label=0,
                              threshold=0.5):
    """
    Classification precision. 
    Based on the given threshold, if N events are classified as positive,
    and n of these N events are truly positive (has pos_label), precision = n / N.

    y_pred   : 1-D np.array, predicted output
    y_true   : 1-D np.array, true label of output
    pos_label: notation of positive label, default 1.
    neg_label: notation of negative label, default 0.
    threshold: a real value between [neg_label, pos_label] to 
               make classification decision. 

    """
    tp = np.sum( y_pred[y_true==pos_label] > threshold )
    fp = np.sum( y_pred[y_true==neg_label] > threshold )
    return tp * 1.0 / (tp + fp)


def recall(y_pred, y_true, pos_label=1, neg_label=0,
                           threshold=0.5):
    """
    Classification recall.
    Based on the given threshold, if there are N pos_label events, and 
    n of these pos_label events are classified as positive, recall = n / N.

    y_pred   : 1-D np.array, predicted output
    y_true   : 1-D np.array, true label of output
    pos_label: notation of positive label, default 1.
    neg_label: notation of negative label, default 0.
    threshold: a real value between [neg_label, pos_label] to 
               make classification decision. 

    """
    tp = np.sum( y_pred[y_true==pos_label] > threshold )
    fn = np.sum( y_pred[y_true==pos_label] <= threshold )
    return tp * 1.0 / (tp + fn)


def AUC(y_pred, y_true):
    return roc_auc_score(y_true, y_pred)


def estimate_metric(y_pred, y_true, pos_label=1, neg_label=0,
                                    threshold=0.5):
    dict = {}
    dict['accuracy'] = accuracy(y_pred, y_true, pos_label=pos_label,
                                                neg_label=neg_label,
                                                threshold=threshold)
    dict['auc'] = AUC(y_pred, y_true)
    dict['precision'] = precision(y_pred, y_true, pos_label=pos_label,
                                                  neg_label=neg_label,
                                                  threshold=threshold)
    dict['recall'] = recall(y_pred, y_true, pos_label=pos_label,
                                            neg_label=neg_label,
                                            threshold=threshold)
    return dict
    

if __name__ == '__main__':
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    result = estimate_metric(y_pred, y_true)
    print result
    print

    y_true = np.array([0,0,1,0,0,1,1,0,1,1])
    result = estimate_metric(y_pred, y_true)
    print result
    print

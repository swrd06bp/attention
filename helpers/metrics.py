import numpy as np

def get_reduction(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    true_negatives = len(preds) - np.count_nonzero(preds + labels)
    false_positives = len(np.intersect1d(np.where(preds==1)[0], np.where(labels==0)[0]))

    return float(true_negatives)/(float(true_negatives) + float(false_positives))

def get_recall(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    true_positives = np.sum(np.multiply(preds, labels))
    false_negatives = len(np.intersect1d(np.where(preds==0)[0], np.where(labels==1)[0]))
    if float(true_positives + false_negatives) == 0:
        return 0
    else:
        return float(true_positives)/float(true_positives + false_negatives)

def get_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return float(np.count_nonzero(np.equal(preds, labels)))/float(len(preds))

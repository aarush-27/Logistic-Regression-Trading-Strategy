import numpy as np

def confusion_matrix(y_true, y_pred):

    TP = TN = FP = FN = 0

    for i in range(len(y_true)):

        actual = y_true[i,0]
        predicted = y_pred[i,0]

        if actual == 1 and predicted == 1:
            TP += 1

        elif actual == 0 and predicted == 0:
            TN += 1

        elif actual == 0 and predicted == 1:
            FP += 1

        else:
            FN += 1

    return TP, TN, FP, FN

def accuracy(TP, TN, FP, FN):

    return (TP + TN)/(TP + TN + FP + FN)

def precision(TP, FP):
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)

def recall(TP, FN):

    return TP/(TP + FN)

def f1_score(precision, recall):

    return 2*precision*recall/(precision + recall)
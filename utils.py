# utils.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def save_label_map(label_map, outpath):
    with open(outpath, "w") as f:
        json.dump(label_map, f)

def plot_confusion_matrix(y_true, y_pred, classes, outpath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_multiclass_roc(y_true, y_probs, classes, outpath):
    n_classes = len(classes)
    y_true_bin = np.zeros((len(y_true), n_classes))
    for i, t in enumerate(y_true):
        y_true_bin[i, t] = 1
    plt.figure(figsize=(6,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

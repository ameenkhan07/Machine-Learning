from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = 'outputs'


def save_plot(plt, filename):
    """Generates and saves the plot to output file
    Parameters
    ----------
        history: keras callbacks.History object
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if 'png' not in filename:
        filename+='.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))


def _conf_matrix_accuracy(confusion_matrix):
    """Returns accuracy calculated from Confusion Matrix
    """
    corr = sum([confusion_matrix[i][i]
                for i in range(len(confusion_matrix[0]))])
    total = sum([sum(confusion_matrix[i])
                 for i in range(len(confusion_matrix))])
    return ((corr/total)*100)


def _plot_confusion_matrix(cm, filename):
    """Plots the confusion matrix.

    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    title = 'Confusion Matrix'
    filename = title+' '+filename
    title = filename.split('.')[0]

    _c = [str(i) for i in range(10)]
    cmap = plt.cm.Reds
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(_c))
    plt.xticks(tick_marks, _c)
    plt.yticks(tick_marks, _c)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    save_plot(plt, title)
    # plt.show()


def get_confusion_matrix(y, t, filename):
    """Plots confusion matrix and returns calculated accuracy
    """
    cm = confusion_matrix(y, t)
    _plot_confusion_matrix(cm, filename=filename)
    return(_conf_matrix_accuracy(cm))

import os
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from matplotlib.backends.backend_pdf import PdfPages

ff = plt.figure()

MODEL = 'cnn'

# result_dir = "result"
data_dir = "data/OpenNRE"

def PrecisionAtRecall(pAll, rAll, rMark):
    length = len(rAll)
    lo = 0
    hi = length - 1
    mark = length >> 1
    error = rMark - rAll[mark]
    while np.abs(error) > 0.005:
        if error > 0:
            hi = mark - 1
        else:
            lo = mark + 1
        mark = (hi + lo) >> 1
        error = rMark - rAll[mark]
    return pAll[mark], rAll[mark], mark

rel_map = {}
with open(os.path.join(data_dir,"rel2id.txt"),'r') as f:
    relations = f.readlines()
for index,rel in enumerate(relations):
    rel_map[rel.strip()] = index

color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']

test_model = ['cnn' + '+sen_att']
test_epoch = ['9']
avg_pres = []
for temp, (model, step) in enumerate(zip(test_model, test_epoch)):
    y_scores = pd.read_csv(os.path.join("data/test_results.tsv"),delimiter="\t",header=None).values
    y_true_labels = pd.read_csv("data/OpenNRE/test.csv",delimiter="\t",header=None)[3].values
    y_true = []
    for label in y_true_labels:
        print(rel_map[label.strip()])
    y_scores = np.argmax(y_scores)
    y_true = tf.one_hot(y_true,len(rel_map))
    y_scores = np.reshape(y_scores, (-1))
    y_true = np.reshape(y_true, (-1))
    precision, recall, threshold = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    avg_pres.append(average_precision)
    recall = recall[::-1]
    precision = precision[::-1]
    plt.plot(recall[:], precision[:], lw=2, color=color[1], label="baseline")

# lines_cnn = open('cnn.txt').readlines()
# lines_cnn = [t.strip().split()[:2] for t in lines_cnn]
# precision_cnn = np.array([t[0] for t in lines_cnn], dtype=np.float32)
# recall_cnn = np.array([t[1] for t in lines_cnn], dtype=np.float32)
# plt.plot(recall_cnn, precision_cnn, lw=2, color=color[-1], label="CNN+ATT")
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.3, 1.0])
# plt.xlim([0.0, 0.4])
# plt.title('Precision-Recall Area={0:0.4f}'.format(avg_pres[-1]))
# plt.legend(loc="upper right")
# plt.grid(True)
# plt.savefig('sgd_' + MODEL)
# plt.plot(range(10), range(10), "o")
# plt.show()
# ff.savefig("pr.pdf", bbox_inches='tight')

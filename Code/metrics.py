import numpy as np
import tensorflow as tf

from pdb import set_trace

log_2_e = 1.44269504089  # Constant to convert to binary entropies


class MetricsTracker():

    def __init__(self, top_10=False):
        self.total_samples = 0
        self.flush()

    def flush(self, flush_totals=False):
        if flush_totals:
            self.total_samples = 0
        self.entropy = 0
        self.acc = 0.0
        self.acc_count = 0

    def add_observation(self, targets, predictions, loss):
        # Compute overall statistics, gathering types and predictions accordingly
        num_samples = targets.shape[0]
        self.entropy += log_2_e * loss.numpy() * num_samples
        self.acc += (num_samples * tf.metrics.binary_accuracy(targets, predictions)).numpy()
        self.acc_count += num_samples
        self.total_samples += int(num_samples)

    def get_stats(self):
        loss = self.entropy / self.acc_count if self.acc_count > 0 else 0
        acc = self.acc / self.acc_count if self.acc_count > 0 else 0
        return self.total_samples, "{0:.3f}".format(loss), "{0:.2%}".format(acc)


if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, accuracy_score, auc

    y_preds, y_true = [], []
    with open('../data/valid_ep1.pkl', 'rb') as f:
        data = pickle.load(f)
        y_preds, y_true = data

    y_true = y_true[0]

    """ if aligned and say is aligned
        precision tp / (tp + fp)
        recall tp / (tp + fn)
    """
    plt.subplots(2,2)
    data = []
    roc = []
    ys = sorted(list(zip(y_true, y_preds)), key=lambda x: x[1])
    for threshold in np.arange(min(y_preds), max(y_preds), .01):
        fn, tp, fp, tn = 0, 0, 0, 0
        for t, p in ys:
            if t == 1.0:
                if p >= threshold:
                    tp += 1
                else:
                    fn += 1
            if t == 0.0:
                if p >= threshold:
                    fp += 1
                else:
                    tn += 1

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        tpr = recall
        fpr = fp /( fp + tn)
        data.append((precision, recall, threshold))
        roc.append((tpr, fpr, threshold))

    recalls = [x[1] for x in data]
    precisions = [x[0] for x in data]

    plt.subplot(2, 2, 1)
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall', fontsize='small')
    plt.ylabel('Precision', fontsize='small')
    plt.title('RN Aligned and Is Aligned at Epoch 1', fontdict={'fontsize': 7})

    tprs = [x[0] for x in roc]
    fprs = [x[1] for x in roc]
    p = plt.subplot(2, 2, 2)
    plt.plot(fprs, tprs, marker='.')
    plt.xlabel('False Positive Rate ' + 'AUC {:.3f}'.format(auc(fprs, tprs)), fontsize='small')
    plt.ylabel('True Positive Rate', fontsize='small')
    plt.title('RN Aligned and Is Aligned at Epoch 1', fontdict={'fontsize': 7})

    """ if not aligned and say is not aligned
        precision tp / (tp + fp)
        recall tp / (tp + fn)
    """
    data = []
    roc = []
    ys = sorted(list(zip(y_true, y_preds)), key=lambda x: x[1])
    for threshold in np.arange(min(y_preds) + .01, max(y_preds), .01):
        fn, tp, fp, tn = 0, 0, 0, 0
        for t, p in ys:
            if t == 0.0:
                if p < threshold:
                    tp += 1
                else:
                    fn += 1
            if t == 1.0:
                if p < threshold:
                    fp += 1
                else:
                    tn += 1

        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        tpr = recall
        fpr = fp /( fp + tn)
        data.append((precision, recall, threshold))
        roc.append((tpr, fpr, threshold))

    recalls = [x[1] for x in data]
    precisions = [x[0] for x in data]
    plt.subplot(2, 2, 3)
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall', fontsize='small')
    plt.ylabel('Precision', fontsize='small')
    plt.title('RN Not Aligned and Is Not Aligned at Epoch 1', fontdict={'fontsize': 7})

    tprs = [x[0] for x in roc]
    fprs = [x[1] for x in roc]
    plt.subplot(2, 2, 4)
    plt.plot(fprs, tprs, marker='.')
    plt.xlabel('False Positive Rate ' + 'AUC {:.3f}'.format(auc(fprs, tprs)), fontsize='small')

    plt.ylabel('True Positive Rate', fontsize='small')

    plt.title('RN Not Aligned and Is Not Aligned at Epoch 1', fontdict={'fontsize': 7})
    plt.tight_layout()
    plt.show()

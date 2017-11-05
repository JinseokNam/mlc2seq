import numpy
import scipy.sparse as sp
import logging
from six.moves import xrange
from collections import OrderedDict

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)


def list2sparse(A, n_labels=None):
    if n_labels is None:
        n_labels_ = 0
        for a in A:
            if n_labels_ < numpy.max(a):
                n_labels_ = numpy.max(a)
        n_labels = n_labels_

    n_samples = len(A)
    mat = sp.dok_matrix((n_samples, n_labels))
    for idx in xrange(n_samples):
        for item in A[idx]:
            mat[idx, item] = 1

    return mat.tocsr()


def is_sparse(matrix):
    return sp.issparse(matrix)


def is_binary_matrix(matrix):
    return numpy.all(numpy.logical_xor(matrix != 1, matrix != 0))


def sparse2dense(sparse_matrix):
    """ convert a sparse matrix into a dense matrix of 0 or 1.

    """
    assert sp.issparse(sparse_matrix)

    return numpy.asarray(sparse_matrix.toarray())


def prepare_evaluation(targets, preds):
    if is_sparse(targets):
        targets = sparse2dense(targets)

    if is_sparse(preds):
        preds = sparse2dense(preds)

    assert numpy.array_equal(targets.shape, preds.shape)
    assert is_binary_matrix(targets)
    assert is_binary_matrix(preds)

    return (targets, preds)


def subset_accuracy(true_targets, predictions, per_sample=False, axis=0):

    result = numpy.all(true_targets == predictions, axis=axis)

    if not per_sample:
        result = numpy.mean(result)

    return result


def hamming_loss(true_targets, predictions, per_sample=False, axis=0):

    result = numpy.mean(numpy.logical_xor(true_targets, predictions),
                        axis=axis)

    if not per_sample:
        result = numpy.mean(result)

    return result


def compute_tp_fp_fn(true_targets, predictions, axis=0):
    # axis: axis for instance

    tp = numpy.sum(true_targets * predictions, axis=axis).astype('float32')
    fp = numpy.sum(numpy.logical_not(true_targets) * predictions,
                   axis=axis).astype('float32')
    fn = numpy.sum(true_targets * numpy.logical_not(predictions),
                   axis=axis).astype('float32')

    return (tp, fp, fn)


def example_f1_score(true_targets, predictions, per_sample=False, axis=0):
    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    example_f1 = 2*tp / (2*tp + fp + fn)

    if per_sample:
        f1 = example_f1
    else:
        f1 = numpy.mean(example_f1)

    return f1


def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    if average == 'micro':
        f1 = 2*numpy.sum(tp) / \
            float(2*numpy.sum(tp) + numpy.sum(fp) + numpy.sum(fn))

    elif average == 'macro':

        def safe_div(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with numpy.errstate(divide='ignore', invalid='ignore'):
                c = numpy.true_divide(a, b)
            return c[numpy.isfinite(c)]

        f1 = numpy.mean(safe_div(2*tp, 2*tp + fp + fn))

    return f1


def f1_score(true_targets, predictions, average='micro', axis=0):
    """
        average: str
            'micro' or 'macro'
        axis: 0 or 1
            label axis
    """
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)

    return f1


def average_precision(true_targets, predictions, per_sample=False, axis=0):
    pass


def compute_all_measures(targets, preds, mb_sz=5000, verbose=0):
    """
    Evaluates the model performance with respect to the following measures:
        Subset accuracy
        Hamming accuracy
        Example-based F1
        Label-based Micro F1
        Label-based Macro F1

    Parameters
    ----------

    targets: sparse matrix of shape (n_instances, n_labels)
        Ground truth

    preds: sparse matrix of shape (n_instances, n_labels)
        Binary predictions by the model

    Returns
    -------

    eval_ret: OrderedDict
        A dictionary that contains evaluation results

    """
    assert targets.shape == preds.shape

    # excluding the <EOS> label
    targets = targets[:, 1:]
    preds = preds[:, 1:]

    n_instances, n_labels = targets.shape
    _mb_sz = mb_sz

    acc_, hl_, exf1_ = [], [], []
    total_tp = numpy.zeros((n_labels,))
    total_fp = numpy.zeros((n_labels,))
    total_fn = numpy.zeros((n_labels,))

    if verbose:
        LOGGER.info('Started to evaluate the predictions')

    for idx in xrange(0, n_instances, _mb_sz):
        if idx + _mb_sz >= n_instances:
            _mb_sz = n_instances - idx

        trg = targets[idx:idx+_mb_sz, :]
        pred = preds[idx:idx+_mb_sz, :]
        assert trg.shape == pred.shape

        trg, pred = prepare_evaluation(trg, pred)

        acc_ += list(subset_accuracy(trg, pred, axis=1, per_sample=True))
        hl_ += list(hamming_loss(trg, pred, axis=1, per_sample=True))
        exf1_ += list(example_f1_score(trg, pred, axis=1, per_sample=True))

        tp, fp, fn = compute_tp_fp_fn(trg, pred, axis=0)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if verbose:
            LOGGER.info('Evaluated {} / {} instances'.format(idx + _mb_sz,
                                                             n_instances))
        del trg, pred

    del targets, preds

    assert len(acc_) == n_instances

    acc = numpy.mean(acc_)
    hl = numpy.mean(hl_)
    exf1 = numpy.mean(exf1_)
    mif1 = f1_score_from_stats(tp, fp, fn, average='micro')
    maf1 = f1_score_from_stats(tp, fp, fn, average='macro')

    eval_ret = OrderedDict([('Subset accuracy', acc),
                            ('Hamming accuracy', 1 - hl),
                            ('Example-based F1', exf1),
                            ('Label-based Micro F1', mif1),
                            ('Label-based Macro F1', maf1)])

    return eval_ret


if __name__ == '__main__':
    A = numpy.array([[1, 1, 0], [1, 0, 0]])
    B = numpy.array([[0, 1, 0], [1, 1, 1]])

    instance_axis = 0
    label_axis = 1

    print('Example-based F1')
    print(example_f1_score(A, B, per_sample=True, axis=label_axis))

    print('Micro F1')
    print(f1_score(A, B, average='micro', axis=instance_axis))

    print('Macro F1')
    print(f1_score(A, B, average='macro', axis=instance_axis))

    print('Subset accuracy')
    print(subset_accuracy(A, B, axis=label_axis))

    print('Hamming loss')
    print(hamming_loss(A, B, axis=label_axis))

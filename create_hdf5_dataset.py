#!/usr/bin/env python

import logging
import io
import argparse
import numpy
import h5py

from fuel.datasets.hdf5 import H5PYDataset

from utils import load_pretrained_embeddings

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)


def count_num_lines(doc_path):
    with io.open(doc_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            pass
    return i + 1


def convert_documents(doc_path, pretrained_word_emb, _mb_sz):
    vocab = {word: index for (index, word) in pretrained_word_emb['vocab']}
    Wemb = pretrained_word_emb['Wemb']

    n_max_docs = count_num_lines(doc_path)
    mb_sz = min(n_max_docs, _mb_sz)
    word_dim = Wemb.shape[1]

    docs = []

    with open(doc_path) as f:
        for doc_id, doc in enumerate(f):
            words = doc.strip().split()
            idx_list = []
            for word in words:
                if word in vocab:
                    idx_list.append(vocab[word])

            assert len(idx_list) > 0, '{}'.format(doc_id)

            docs.append(idx_list)

            if len(docs) == mb_sz:
                doc_vectors = numpy.zeros([mb_sz, word_dim])

                for idx_in_mb, word_indices in enumerate(docs):
                    doc_vectors[idx_in_mb] = Wemb[word_indices].mean(axis=0)

                yield doc_vectors[:]

                docs = []

        if len(docs) > 0:
            doc_vectors = numpy.zeros([len(docs), word_dim])

            for idx_in_mb, word_indices in enumerate(docs):
                doc_vectors[idx_in_mb] = Wemb[word_indices].mean(axis=0)

            yield doc_vectors[:]

    pass


def load_vocab(path):
    with io.open(path, encoding='utf-8') as f:
        items = [line.strip().split()[1] for line in f]

    vocab = {item: idx for idx, item in enumerate(items)}

    return vocab


def convert_label_sets(label_path, label_vocab):
    n_sets = count_num_lines(label_path)

    target_vectors = [None] * n_sets

    with open(label_path) as f:
        for label_set_id, label_set in enumerate(f):
            labels = label_set.strip().split()

            label_indices = [label_vocab[label]
                             for label in labels if label in label_vocab]

            target_vectors[label_set_id] = label_indices

    return target_vectors


def main(trd_path, trl_path, vad_path, val_path, tsd_path, tsl_path,
         label_vocab_path, word_emb_path, output_path):

    label_vocab = load_vocab(label_vocab_path)
    pretrained_emb = load_pretrained_embeddings(word_emb_path)

    LOGGER.info('Converting labels...')
    train_label = convert_label_sets(trl_path, label_vocab)
    valid_label = convert_label_sets(val_path, label_vocab)
    test_label = convert_label_sets(tsl_path, label_vocab)
    LOGGER.info('Done')

    n_trd, n_vad, n_tsd = \
        len(train_label), len(valid_label), len(test_label)

    LOGGER.info('Number of train label sets: {}'.format(n_trd))
    LOGGER.info('Number of valid label sets: {}'.format(n_vad))
    LOGGER.info('Number of test label sets: {}'.format(n_tsd))

    n_total_docs = n_trd + n_vad + n_tsd
    n_dim = pretrained_emb['Wemb'].shape[1]
    mb_sz = 500000

    with h5py.File(output_path, mode='w') as f:
        features = f.create_dataset('features', (n_total_docs, n_dim),
                                    dtype='float32')
        n_processed = 0
        LOGGER.info('Converting train documents ...')
        for train_data in convert_documents(trd_path, pretrained_emb, mb_sz):
            features[n_processed: n_processed+train_data.shape[0]] = train_data
            n_processed += train_data.shape[0]
            LOGGER.info('{} / {}'.format(n_processed, n_total_docs))
        assert n_processed == n_trd
        LOGGER.info('Done')

        LOGGER.info('Converting valid documents ...')
        for valid_data in convert_documents(vad_path, pretrained_emb, mb_sz):
            features[n_processed: n_processed+valid_data.shape[0]] = valid_data
            n_processed += valid_data.shape[0]
            LOGGER.info('{} / {}'.format(n_processed, n_total_docs))
        assert n_processed == n_vad + n_trd
        LOGGER.info('Done')

        LOGGER.info('Converting test documents ...')
        for test_data in convert_documents(tsd_path, pretrained_emb, mb_sz):
            features[n_processed: n_processed+test_data.shape[0]] = test_data
            n_processed += test_data.shape[0]
            LOGGER.info('{} / {}'.format(n_processed, n_total_docs))
        assert n_processed == n_total_docs
        LOGGER.info('Done')

        _dtype = h5py.special_dtype(vlen=numpy.dtype('uint16'))
        targets = f.create_dataset('targets', (n_total_docs,), dtype=_dtype)
        all_target_labels = train_label + valid_label + test_label

        assert n_total_docs == len(all_target_labels)

        targets[...] = numpy.array(all_target_labels)

        # assign labels to the dataset
        features.dims[0].label = 'batch'
        features.dims[1].label = 'feature'
        targets.dims[0].label = 'batch'

        targets_shapes = f.create_dataset(
            'targets_shapes', (n_total_docs, 1), dtype='int32')
        targets_shapes[...] = numpy.array(
            [len(labels) for labels in all_target_labels])[:, None]

        targets.dims.create_scale(targets_shapes, 'shapes')
        targets.dims[0].attach_scale(targets_shapes)

        targets_shape_labels = f.create_dataset(
            'targets_shape_labels', (1,), dtype='S6')
        targets_shape_labels[...] = ['length'.encode('utf8')]

        targets.dims.create_scale(targets_shape_labels, 'shape_labels')
        targets.dims[0].attach_scale(targets_shape_labels)

        split_dict = {
            'train': {'features': (0, n_trd), 'targets': (0, n_trd)},
            'valid': {'features': (n_trd, n_trd + n_vad),
                      'targets': (n_trd, n_trd + n_vad)},
            'test': {'features': (n_trd + n_vad, n_total_docs),
                     'targets': (n_trd + n_vad, n_total_docs)}}

        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
        f.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trd', type=str, required=True)
    parser.add_argument('--trl', type=str, required=True)
    parser.add_argument('--vad', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--tsd', type=str, required=True)
    parser.add_argument('--tsl', type=str, required=True)
    parser.add_argument('--label_vocab', type=str, required=True)
    parser.add_argument('--word_emb', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    main(args.trd, args.trl, args.vad, args.val, args.tsd, args.tsl,
         args.label_vocab, args.word_emb, args.output)

#!/usr/bin/env python

from __future__ import print_function

import io
import argparse
from collections import OrderedDict
from six.moves import xrange

from misc import dfs_topsort


def extract_hierarchy(path_hierarchy, name_id_mapping):
    with io.open(name_id_mapping) as f:
        name_id_map = OrderedDict([
            (idx+1, line.strip().split('=')[0].replace(' ', '_'))
            for idx, line in enumerate(f)])

    graph = {}
    child_desc = {}
    with io.open(path_hierarchy, encoding='utf8') as f:
        for line in f:
            parent, child = line.strip().split(',')
            parent, child = int(parent), int(child)
            assert parent in name_id_map, '{}'.format(parent)
            assert child in name_id_map, '{}'.format(child)

            parent = name_id_map[parent]
            child = name_id_map[child]

            if parent not in graph:
                graph[parent] = []
            graph[parent].append(child)

            if child not in graph:
                graph[child] = []

    return graph, child_desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_hierarchy', type=str, required=True)
    parser.add_argument('--name_id_mapping', type=str, required=True)
    parser.add_argument('--load_old_label_vocab', type=str, required=True)
    parser.add_argument('--save_label_vocab', type=str, required=True)

    args = parser.parse_args()

    hs, label_desc = extract_hierarchy(args.path_to_hierarchy,
                                       args.name_id_mapping)

    order = dfs_topsort(hs, 'MeSH')

    label_vocab = OrderedDict([(order[idx], idx) for idx in xrange(len(order))])

    with io.open(args.load_old_label_vocab) as f:
        orig_label_vocab = OrderedDict(
            [(label.split()[1], idx) for idx, label in enumerate(f)]
        )

    with io.open(args.save_label_vocab, 'w', encoding='utf8') as f:
        for label, label_idx in label_vocab.items():
            if label in orig_label_vocab:
                f.write(u'{}\t{}\n'.format(label_idx, label))

    '''
    ilabel_vocab = {idx: order[idx] for idx in xrange(len(order))}

    with io.open(args.label_path) as f:
        for line in f:
            labels = line.strip().split()
            labels_new_indices = [label_vocab[label] for label in labels]
            new_label_order = [ilabel_vocab[label_idx] for label_idx
                               in sorted(labels_new_indices)]

            print(' '.join(new_label_order))
    '''

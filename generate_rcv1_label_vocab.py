#!/usr/bin/env python

from __future__ import print_function

import io
import argparse
from collections import OrderedDict
from six.moves import xrange

from misc import dfs_topsort


def extract_hierarchy(path_hierarchy):
    graph = {}
    child_desc = {}
    with io.open(path_hierarchy, encoding='utf8') as f:
        for line in f:
            _, parent, _, child = line.strip().split()[:4]
            desc = '_'.join(line.strip().split()[5:])

            if parent not in graph:
                graph[parent] = []
            graph[parent].append(child)

            if child not in graph:
                graph[child] = []

            if child not in child_desc:
                child_desc[child] = desc

    return graph, child_desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_hierarchy', type=str, required=True)
    parser.add_argument('--save_label_vocab', type=str, required=True)

    args = parser.parse_args()

    hs, label_desc = extract_hierarchy(args.path_to_hierarchy)

    order = dfs_topsort(hs, root='Root')

    label_vocab = OrderedDict([(order[idx], idx) for idx in xrange(len(order))])

    with io.open(args.save_label_vocab, 'w', encoding='utf8') as f:
        for label, label_idx in label_vocab.items():
            f.write(u'{}\t{}\n'.format(label_idx, label))

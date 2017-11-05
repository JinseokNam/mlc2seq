#!/usr/bin/env python

import pprint
import logging
import argparse

from misc import loadz
from evals import list2sparse, compute_all_measures

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)


def main(result_output_path):
    LOGGER.info('Loading data from {}'.format(result_output_path))
    result = loadz(result_output_path)

    LOGGER.info('Converting targets and predictions into sparse matrices')
    n_labels = len(result['label_vocab'])
    preds = list2sparse(result['predictions'], n_labels=n_labels)
    targets = list2sparse(result['targets'], n_labels=n_labels)
    LOGGER.info('Done')

    eval_ret = compute_all_measures(targets, preds, mb_sz=10000, verbose=True)

    return eval_ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    eval_results = main(args.load)
    if args.output is not None:
        with open(args.output, 'w') as f:
            pprint.pprint(list(eval_results.items()), f)
    else:
        pprint.pprint(list(eval_results.items()))

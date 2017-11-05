from __future__ import print_function

import argparse
import sys
import io
import json
import os
import logging
from utils import mul2bin

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)

try:
    import numpy
    from six.moves import xrange
    import six

    import theano
    from theano.sandbox.rng_mrg import MRG_RandomStreams
    from data_iterator import (load_test_data, load_dict)
    from mlc2seq_base import (build_sampler, gen_sample, init_params)
    from utils import (load_params, init_tparams)
    from misc import savez
except ImportError as e:
    EXIT_STATUS = 64
    print('Failed to import: %s' % str(e), file=sys.stderr)
    sys.exit(EXIT_STATUS)


def main(model_path, data_base_path, option_path, saveto, k):

    # load model_options
    with io.open(option_path, encoding='utf8') as f:
        config = json.load(f)

    model_options = config['model']
    test_data_options = config['testdata']
    label_vocab_size = test_data_options['n_labels']
    assert 'reverse_labels' in config['data']
    reverse_labels = config['data']['reverse_labels']

    def join_data_base_path(data_base, options):
        for kk, vv in six.iteritems(options):
            if kk in ['src', 'trg', 'input_vocab', 'label_vocab']:
                options[kk] = os.path.join(data_base, options[kk])

        return options

    test_data_options = join_data_base_path(data_base_path, test_data_options)
    dicts_r, test_stream = load_test_data(**test_data_options)

    word_vocab = load_dict(test_data_options['input_vocab'])
    iword_vocab = dict((vv, kk) for kk, vv in six.iteritems(word_vocab))
    label_vocab = load_dict(test_data_options['label_vocab'],
                            dict_size=label_vocab_size,
                            include_unk=False, reverse=reverse_labels)
    ilabel_vocab = dict((vv, kk) for kk, vv in six.iteritems(label_vocab))

    model_options['n_labels'] = len(label_vocab)

    LOGGER.info('Building model')
    params = init_params(model_options)

    LOGGER.info('Loading parameters from {}'.format(model_path))
    params = load_params(model_path, params)

    LOGGER.info('Initializing parameters')
    tparams = init_tparams(params)

    # use_noise is for dropout
    use_noise = theano.shared(numpy.float32(0.))
    trng = MRG_RandomStreams(1234)

    n_samples = 0

    LOGGER.info('Building sampler')
    f_sample_inits, f_sample_nexts \
        = build_sampler(tparams, model_options, trng, use_noise)

    results = dict()
    results['input_vocab'] = iword_vocab
    results['label_vocab'] = ilabel_vocab
    results['src'] = dict()
    results['predictions'] = dict()
    results['targets'] = dict()
    results['alignments'] = dict()

    for x, x_mask, y, y_mask in test_stream.get_epoch_iterator():
        orig_x = x
        if model_options['label_type'] == 'binary':
            y, y_mask = mul2bin(y, y_mask, model_options['n_bins'])

        x, x_mask = x.T, x_mask.T

        if model_options['enc_dir'] == 'none':
            x_mask[(x == 0) | (x == 1)] = 0.

        for jj in xrange(x.shape[1]):
            sample_encoder_inps = [
                x[:, jj][:, None],
                x_mask[:, jj][:, None]
            ]

            solutions = gen_sample(tparams,
                                   f_sample_inits,
                                   f_sample_nexts,
                                   sample_encoder_inps,
                                   model_options,
                                   trng=trng,
                                   k=k,
                                   max_label_len=50,
                                   argmax=False)

            samples = solutions['samples']
            alignment = solutions['alignments']
            scores = solutions['scores']

            scores = scores / numpy.array([len(s) for s in samples])
            best_sample = samples[scores.argmin()]
            best_alignment = alignment[scores.argmin()]

            results['src'][n_samples + jj] = orig_x[jj]
            results['predictions'][n_samples + jj] = best_sample
            results['alignments'][n_samples + jj] = numpy.array(best_alignment)
            results['targets'][n_samples + jj] = y[jj, y_mask[jj] == 1]

        n_samples += x.shape[1]
        LOGGER.info('Number of processed instances: {}'.format(n_samples))

    LOGGER.info('Making predictions successfully on {} instances'.format(
        n_samples))
    savez(results, saveto, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--base_datapath', type=str)
    parser.add_argument('--saveto', type=str, required=True)

    args = parser.parse_args()

    if args.base_datapath:
        data_base_path = os.path.realpath(args.base_datapath)
    else:
        data_base_path = os.getcwd()

    main(args.model, data_base_path, args.config,
         args.saveto, k=args.k)

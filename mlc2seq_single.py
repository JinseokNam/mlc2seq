import argparse
import binascii
import copy
import io
import json
import logging
import os
import time
import traceback
from collections import OrderedDict

import numpy
import shutil
import six
import theano
from mimir import Logger
from theano import tensor
from six.moves import xrange
from toolz.dicttoolz import merge
from scipy.sparse import coo_matrix

from data_iterator import load_data, UNK_TOKEN
from mlc2seq_base import (pred_probs, build_model, build_sampler,
                          gen_sample, save_params, init_params)
from utils import (load_params, init_tparams, zipp, name_dict,
                   unzip, itemlist, prepare_character_tensor, mul2bin,
                   load_pretrained_embeddings)
from evals import (subset_accuracy, hamming_loss, example_f1_score,
                   compute_tp_fp_fn, f1_score_from_stats, prepare_evaluation)

import optimizers

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)


def do_validation(f_inits, f_nexts, tparams,
                  max_sample_length, trng, options, data_stream):
    """
        Parameters
        ----------

        f_inits: list

        f_nexts: list

        tparams: dict

        max_sample_length: int

        trng

        options: dict

        data_stream: Fuel data stream

    """

    num_labels = options['n_bins'] if options['label_type'] == 'binary' \
        else options['n_labels']

    total_subset_accuracy, total_hamming_loss, \
        total_example_f1_score = [], [], []

    total_tp = numpy.zeros((num_labels-1,))
    total_fp = numpy.zeros((num_labels-1,))
    total_fn = numpy.zeros((num_labels-1,))

    for x, x_mask, y, y_mask in data_stream.get_epoch_iterator():

        if options['label_type'] == 'binary':
            y, y_mask = mul2bin(y, y_mask, options['n_bins'])
        else:
            y, y_mask = mul2bin(y, y_mask, options['n_labels'])

        assert numpy.sum(y[:, 0]) == y.shape[0], '{}:{}'.format(
            numpy.sum(y[:, 0]), y.shape[0])

        y, y_mask = y.T, y_mask.T

        if options['input_token_level'] == 'character':
            x, x_mask = prepare_character_tensor(x)
        else:
            x, x_mask = x.T, x_mask.T

        unk_token_ratio = (x == 1).sum(0) / x_mask.sum(0)
        non_empty_insts = unk_token_ratio <= 0.5

        y = y[:, non_empty_insts]
        y_mask = y_mask[:, non_empty_insts]
        x = x[:, non_empty_insts]
        x_mask = x_mask[:, non_empty_insts]

        if x.shape[1] == 0:
            continue

        predictions_inst_ids = []
        predictions = []

        for i in xrange(x.shape[1]):
            if options['input_token_level'] == 'character':
                encoder_inps = [
                    x[:, :, i][:, :, None],
                    x_mask[:, :, i][:, :, None]
                ]
            else:
                encoder_inps = [
                    x[:, i][:, None],
                    x_mask[:, i][:, None]
                ]

            solutions = gen_sample(tparams,
                                   f_inits,
                                   f_nexts,
                                   encoder_inps,
                                   options,
                                   trng=trng,
                                   k=5,
                                   max_label_len=max_sample_length,
                                   argmax=False)

            sample = solutions['samples']
            # alignment = solutions['alignments']
            score = solutions['scores']

            score = score / numpy.array([len(s) for s in sample])
            ss = sample[score.argmin()]
            # alignment = alignment[score.argmin()]

            if options['label_type'] == 'binary':
                assert type(ss) == list
                assert len(ss) == max_sample_length
                new_ss = [tidx
                          for tidx, s in enumerate(ss) if s == 1]
                ss = new_ss
                assert type(ss) == list

                if len(ss) == 0:
                    ss.append(0)

                if ss[0] == 0:      # if the first token is <EOS>
                    ss = ss[1:] + ss[:1]

            else:
                # prevent duplicate predictions
                ss = set(ss)

            # store predictions for computing evaluation scores
            predictions_inst_ids += [i] * len(ss)
            predictions += ss

        assert len(predictions_inst_ids) == len(predictions)

        # convert predictions to sparse matrix
        preds = coo_matrix(([1] * len(predictions),
                           (predictions, predictions_inst_ids)),  # (row, col)
                           shape=y.shape,
                           dtype=numpy.int8).tocsc()

        y_, preds_ = prepare_evaluation(y[1:, :], preds[1:, :])

        total_subset_accuracy += list(subset_accuracy(y_, preds_,
                                                      per_sample=True))
        total_hamming_loss += list(hamming_loss(y_, preds_,
                                                per_sample=True))
        total_example_f1_score += list(example_f1_score(y_, preds_,
                                                        per_sample=True))

        tp, fp, fn = compute_tp_fp_fn(y_, preds_, axis=1)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    eval_scores = OrderedDict([
                    ('subset_acc', numpy.mean(total_subset_accuracy)),
                    ('hamming_loss', numpy.mean(total_hamming_loss)),
                    ('example_based_f1', numpy.mean(total_example_f1_score)),
                    ('micro_f1', f1_score_from_stats(
                        total_tp,
                        total_fp,
                        total_fn, average='micro')),
                    ('macro_f1', f1_score_from_stats(
                        total_tp,
                        total_fp,
                        total_fn, average='macro'))])

    return eval_scores


def train(experiment_id, data_base_path, output_base_path,
          model_options, data_options, validation_options,
          patience,  # early stopping patience
          max_epochs,
          finish_after,  # finish after this many updates
          clip_c,  # gradient clipping threshold
          lrate,  # learning rate
          optimizer,
          saveto,
          valid_freq,
          time_limit,
          save_freq,   # save the parameters after every saveFreq updates
          sample_freq,   # generate some samples after every sampleFreq
          verbose,
          reload_from=None,
          pretrained_word_emb=None):

    start_time = time.time()

    def join_data_base_path(data_base, options):
        for kk, vv in six.iteritems(options):
            if kk in ['src', 'trg', 'input_vocab',
                      'label_vocab', 'valid_src',
                      'valid_trg']:
                options[kk] = os.path.join(data_base, options[kk])

        return options

    data_options = join_data_base_path(data_base_path, data_options)
    validation_options = join_data_base_path(data_base_path,
                                             validation_options)

    worddicts_r, train_stream, valid_stream = load_data(**data_options)

    model_options['n_input_tokens'] = len(worddicts_r[0])
    model_options['n_labels'] = len(worddicts_r[1])

    if model_options['label_type'] == 'binary':
        model_options['n_bins'] = len(worddicts_r[1])
        model_options['n_labels'] = 2
        max_sample_length = len(worddicts_r[1])
    else:
        max_sample_length = data_options['max_label_length']

    LOGGER.info('Building model')
    params = init_params(model_options)
    # reload parameters
    best_filename = '{}/{}.{}.best.npz'.format(output_base_path,
                                               experiment_id,
                                               saveto)

    if pretrained_word_emb and os.path.exists(pretrained_word_emb):
        assert model_options['input_token_level'] == 'word'
        LOGGER.info('Loading pretrained word embeddings from {}'.format(
           pretrained_word_emb
        ))

        pretrained_emb = load_pretrained_embeddings(pretrained_word_emb)

        # TODO check if the size of the pretrained word embedding equals
        # the size of the initialized word embeddings
        # Also, check whether or not the vocabulary in the pretrained word
        # embeddings is identical to the vocabulary in the model.
        pvocab = pretrained_emb['vocab']    # (idx, word)

        # XXX if the assertians passed, then load the pretrained embeddings
        assert pretrained_emb['Wemb'].dtype == numpy.float32, \
            'The pretrained word embeddings should be float32\n'
        assert pretrained_emb['Wemb'].shape[1] == params['Wemb'].shape[1], \
            '{} does not match {}\n'.format(pretrained_emb['Wemb'].shape[1],
                                            params['Wemb'].shape[1])

        pretrained_word2id = {word: idx for (idx, word) in pvocab}

        param_indices, indices = [], []
        for ii in xrange(len(worddicts_r[0])):

            if ii >= data_options['n_input_tokens']:
                break

            word = worddicts_r[0][ii]
            if word in pretrained_word2id:
                word_idx = pretrained_word2id[word]
                indices.append(word_idx)
                param_indices.append(ii)

        assert len(indices) <= data_options['n_input_tokens']

        params['Wemb'][param_indices] = pretrained_emb['Wemb'][indices]
        # normalize word embeddings
        params['Wemb'] = params['Wemb'] / \
            numpy.sqrt((params['Wemb']**2).sum(axis=1)[:, None])

    if reload_from and os.path.exists(reload_from):
        LOGGER.info('Loading parameters from {}'.format(reload_from))
        params = load_params(reload_from, params)

    LOGGER.info('Initializing parameters')
    tparams = init_tparams(params)

    # use_noise is for dropout
    trng, use_noise, encoder_vars, decoder_vars, \
        opt_ret, costs = build_model(tparams, model_options)

    inps = encoder_vars + decoder_vars

    LOGGER.info('Building sampler')
    f_sample_inits, f_sample_nexts \
        = build_sampler(tparams, model_options, trng, use_noise)

    # before any regularizer
    LOGGER.info('Building functions to compute log prob')
    f_log_probs = [
        theano.function(inps, cost_, name='f_log_probs_%s' % cost_.name,
                        on_unused_input='ignore')
        for cost_ in costs
    ]

    assert len(costs) == 1

    cost = costs[0]
    '''
    for cost_ in costs[1:]:
        cost += cost_
    '''
    cost = cost.mean()

    LOGGER.info('Computing gradient')
    grads = tensor.grad(cost, wrt=itemlist(tparams))

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g ** 2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c ** 2), g / tensor.sqrt(
                g2) * clip_c, g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')

    LOGGER.info('Building optimizers')
    f_grad_shared, f_update, optimizer_state = \
        getattr(optimizers, optimizer)(lr, tparams, grads, inps, cost)

    optimizer_state = name_dict(optimizer_state)

    # TODO set_value optimizer_state
    if reload_from and os.path.exists(reload_from):
        LOGGER.info('Loading optimizer state from {}'.format(reload_from))
        optimizer_state = load_params(reload_from, optimizer_state,
                                      theano_var=True)

    LOGGER.info('Optimization')

    log = Logger(filename='{}/{}.log.jsonl.gz'.format(output_base_path,
                                                      experiment_id))

    best_valid_err = float('inf')
    best_model = None
    total_nsamples = 0
    uidx = 0
    uidx_restore = [0]
    estop = False
    if reload_from and os.path.exists(reload_from):
        rmodel = numpy.load(reload_from)
        if 'uidx' in rmodel:
            uidx_restore = rmodel['uidx']
        if 'best_valid_err' in rmodel:
            best_valid_err = rmodel['best_valid_err']
        if 'total_nsamples' in rmodel and rmodel['total_nsamples'] > 0:
            total_nsamples = rmodel['total_nsamples']

        best_model = [unzip(tparams), unzip(optimizer_state), uidx_restore]

    train_start = time.clock()
    max_updates_per_epoch = total_nsamples / data_options['batch_size']

    try:
        for epoch in xrange(0, max_epochs):
            if total_nsamples > 0 and \
                    uidx + max_updates_per_epoch < uidx_restore[0]:
                uidx += max_updates_per_epoch
                continue

            n_samples = 0
            for x, x_mask, \
                    y, y_mask in train_stream.get_epoch_iterator():

                n_samples += len(x)

                uidx += 1
                if uidx < uidx_restore[0]:
                    continue

                x_length = x_mask.sum(1).mean()

                if model_options['label_type'] == 'binary':
                    old_y = y
                    y, y_mask = mul2bin(y, y_mask, model_options['n_bins'])

                y, y_mask = y.T, y_mask.T

                if data_options['input_token_level'] == 'character':
                    x, x_mask = prepare_character_tensor(x)
                else:
                    x, x_mask = x.T, x_mask.T

                unk_token_ratio = (x == 1).sum(0) / x_mask.sum(0)
                non_empty_insts = unk_token_ratio <= 0.5

                y = y[:, non_empty_insts]
                y_mask = y_mask[:, non_empty_insts]
                x = x[:, non_empty_insts]
                x_mask = x_mask[:, non_empty_insts]

                if x.shape[1] == 0:
                    continue

                encoder_inps = [x, x_mask]
                decoder_inps = [y, y_mask]

                inps = encoder_inps + decoder_inps

                use_noise.set_value(1.)

                log_entry = {'iteration': uidx, 'epoch': epoch}

                # compute cost, grads and copy grads to shared variables
                update_start = time.clock()
                cost = f_grad_shared(*inps)
                f_update(lrate)

                if verbose:
                    log_entry['cost'] = float(cost)
                    log_entry['average_source_length'] = \
                        float(x_length)
                    log_entry['average_target_length'] = \
                        float(y_mask.sum(0).mean())
                    log_entry['update_time'] = time.clock() - update_start
                    log_entry['train_time'] = time.clock() - train_start

                # check for bad numbers, usually we remove non-finite elements
                # and continue training - but not done here
                if not numpy.isfinite(cost):
                    LOGGER.error('NaN detected')
                    return 1., 1., 1.

                # validate model on validation set and early stop if necessary
                if numpy.mod(uidx, valid_freq) == 0:
                    use_noise.set_value(0.)
                    valid_errs = [
                        numpy.mean(
                            pred_probs(f_,
                                       model_options,
                                       valid_stream))
                        for f_ in f_log_probs
                    ]

                    for f_, err_ in zip(f_log_probs, valid_errs):
                        log_entry['validation_%s' % f_.name] = float(err_)

                    valid_scores = do_validation(f_sample_inits,
                                                 f_sample_nexts,
                                                 tparams, max_sample_length,
                                                 trng, model_options,
                                                 valid_stream)

                    for eval_type, score in valid_scores.items():
                        log_entry['validation_%s' % eval_type] = score

                    for f_, err_ in zip(f_log_probs, valid_errs):
                        if not numpy.isfinite(err_):
                            raise RuntimeError(('NaN detected in validation '
                                                'error of %s') % f_.name)
                    valid_err = numpy.array(valid_errs).sum()

                    if valid_err < best_valid_err:
                        best_valid_err = valid_err
                        best_model = [
                            unzip(tparams),
                            unzip(optimizer_state),
                            [uidx]
                        ]

                # save the best model so far
                if numpy.mod(uidx, save_freq) == 0 and \
                        uidx > uidx_restore[0]:
                    LOGGER.info('Saving best model so far')

                    if best_model is not None:
                        params, opt_state, save_at_uidx = best_model
                    else:
                        params = unzip(tparams)
                        opt_state = unzip(optimizer_state)
                        save_at_uidx = [uidx]

                    # save params to exp_id.npz and symlink model.npz to it
                    params_and_state = merge(params, opt_state,
                                             {'uidx': save_at_uidx},
                                             {'best_valid_err': best_valid_err},
                                             {'total_nsamples': total_nsamples})
                    save_params(params_and_state, best_filename)

                # generate some samples with the model and display them
                if sample_freq > 0 and numpy.mod(uidx, sample_freq) == 0:
                    # FIXME: random selection?
                    log_entry['samples'] = []

                    if data_options['input_token_level'] == 'character':
                        batch_size = x.shape[2]
                    else:
                        batch_size = x.shape[1]
                    for jj in xrange(numpy.minimum(5, batch_size)):
                        stats = [('source', ''), ('truth', ''), ('sample', ''),
                                 ('align_sample', '')]
                        log_entry['samples'].append(OrderedDict(stats))

                        if data_options['input_token_level'] == 'character':
                            sample_encoder_inps = [
                                x[:, :, jj][:, :, None],
                                x_mask[:, :, jj][:, :, None]
                            ]
                        else:
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
                                               k=12,
                                               max_label_len=max_sample_length,
                                               argmax=False)

                        sample = solutions['samples']
                        alignment = solutions['alignments']
                        score = solutions['scores']

                        score = score / numpy.array(
                            [len(s) for s in sample])
                        ss = sample[score.argmin()]
                        alignment = alignment[score.argmin()]

                        if model_options['label_type'] == 'binary':
                            # print(y[0], y.shape, old_y.shape)
                            y = old_y.T
                            assert type(ss) == list
                            assert len(ss) == max_sample_length
                            new_ss = [tidx
                                      for tidx, s in enumerate(ss) if s == 1]
                            # print(len(ss), numpy.sum(ss), new_ss)
                            ss = new_ss
                            assert type(ss) == list

                            if len(ss) == 0:
                                ss.append(0)

                            if ss[0] == 0:      # if the first token is <EOS>
                                ss = ss[1:] + ss[:1]

                        if data_options['input_token_level'] == 'character':
                            num_src_words = int((
                                x_mask[:, :, jj].sum(0) > 0).sum())

                            num_chars, num_words, num_samples = x.shape
                            for widx in xrange(num_words):
                                if x_mask[:, widx, jj].sum() == 0:
                                    break
                                for cidx in xrange(num_chars):
                                    cc = x[cidx, widx, jj]
                                    if cc == 0:
                                        break
                                    if cc in worddicts_r[0]:
                                        token = worddicts_r[0][cc]
                                    else:
                                        token = UNK_TOKEN
                                    log_entry['samples'][-1]['source'] \
                                        += token
                                log_entry['samples'][-1]['source'] += \
                                    ' '
                        else:
                            num_src_words = int(x_mask[:, jj].sum())

                            num_words, num_samples = x.shape
                            for vv in x[:, jj]:
                                if vv == 0:
                                    break
                                if vv in worddicts_r[0]:
                                    token = worddicts_r[0][vv]
                                else:
                                    token = UNK_TOKEN
                                log_entry['samples'][-1]['source'] \
                                    += token + ' '

                        for vv in y[:, jj]:
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                token = worddicts_r[1][vv]
                            else:
                                token = UNK_TOKEN
                            log_entry['samples'][-1]['truth'] += token + ' '

                        for tidx, vv in enumerate(ss):
                            if vv == 0:
                                break
                            if vv in worddicts_r[1]:
                                token = worddicts_r[1][vv]
                            else:
                                token = UNK_TOKEN

                            assert tidx >= 0 and tidx < len(alignment), \
                                '%d\t%d' % (tidx, len(alignment))

                            align_src_word_idx = \
                                (alignment[tidx][
                                    :num_src_words-1]).argmax()
                            aligned_token = '%s_<%d>' % \
                                (token, align_src_word_idx)

                            log_entry['samples'][-1]['sample'] += token + ' '
                            log_entry['samples'][-1]['align_sample'] \
                                += aligned_token + ' '

                # finish after this many updates
                if uidx >= finish_after:
                    LOGGER.info('Finishing after {} iterations'.format(uidx))
                    estop = True
                    break

                if time_limit > 0 and \
                   (time.time() - start_time > time_limit * 60):

                    LOGGER.info('Time limit {} mins is over'.format(
                        time_limit))
                    estop = True

                    break

                if verbose and len(log_entry) > 2:
                    log.log(log_entry)

            LOGGER.info('Completed epoch, seen {} samples'.format(n_samples))
            if total_nsamples == 0:
                total_nsamples = n_samples

            if estop:
                log.log(log_entry)
                break

        if best_model is not None:
            assert len(best_model) == 3
            best_p, best_state, best_uidx = best_model
            zipp(best_p, tparams)
            zipp(best_state, optimizer_state)

        '''
        use_noise.set_value(0.)
        LOGGER.info('Calculating validation cost')
        valid_errs = do_validation(f_log_probs, model_options, valid_stream)
        '''

        if not best_model:
            best_p = unzip(tparams)
            best_state = unzip(optimizer_state)
            best_uidx = [uidx]

        best_p = copy.copy(best_p)
        best_state = copy.copy(best_state)
        params_and_state = merge(best_p,
                                 best_state,
                                 {'uidx': best_uidx},
                                 {'best_valid_err': best_valid_err},
                                 {'total_nsamples': total_nsamples})
        save_params(params_and_state, best_filename)

    except Exception:
        LOGGER.error(traceback.format_exc())
        best_valid_err = -1.
    else:
        # XXX add something needed
        print('Training Done')

    return best_valid_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--base_datapath', type=str)
    parser.add_argument('--base_outputpath', type=str)
    parser.add_argument('--experiment_id', type=str)

    args = parser.parse_args()

    # Load the configuration file
    with io.open(args.config) as f:
        config = json.load(f)
    if args.base_datapath:
        data_base_path = os.path.realpath(args.base_datapath)
    else:
        data_base_path = os.getcwd()

    if args.base_outputpath:
        output_base_path = os.path.realpath(args.base_outputpath)
    else:
        output_base_path = os.getcwd()

    # Create unique experiment ID and backup config file
    if args.experiment_id:
        experiment_id = args.experiment_id
    else:
        experiment_id = binascii.hexlify(os.urandom(3)).decode()

    shutil.copyfile(args.config,
                    '{}/{}.config.json'.format(output_base_path, experiment_id))
    train(experiment_id, data_base_path, output_base_path, config['model'],
          config['data'], config['validation'],
          **merge(config['training'], config['management']))

from __future__ import division

import logging
import os
from collections import OrderedDict

import numpy
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

from utils import (dropout_layer, norm_weight, concatenate, mul2bin,
                   prepare_character_tensor, beam_search)
from layers import get_layer
from six.moves import xrange

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    if options['input_token_level'] == 'character':
        assert 'dim_char' in options and \
            'num_filters' in options and \
            'char_emb_filters' in options

        """
            Define parameters for a character-level model

        """
        # character embedding
        params['Cemb'] = norm_weight(
            options['n_input_tokens'],
            options['dim_char'],
            scale=1/numpy.sqrt(options['n_input_tokens'])
        )

        params = get_layer('conv')[0](
            options,
            params,
            prefix='char_emb_conv',
            nin=options['dim_char'],
            nout=options['num_filters'],
            filter_sizes=options['char_emb_filters']
        )

        options['dim_word'] = \
            options['num_filters'] * len(options['char_emb_filters'])

    else:
        assert 'dim_word' in options

        """
            Define parameters for word embeddings

        """
        # word embedding
        params['Wemb'] = norm_weight(
            options['n_input_tokens'],
            options['dim_word'],
            scale=1/numpy.sqrt(options['n_input_tokens'])
        )

    """
        Define parameters for a word-level model

    """
    # encoder: bidirectional RNN for source words
    if options['enc_dir'] == 'forward' or \
       options['enc_dir'] == 'bidir':
        params = get_layer(options['encoder'])[0](options,
                                                  params,
                                                  prefix='word_encoder',
                                                  nin=options['dim_word'],
                                                  dim=options['encoder_dim'])
    if options['enc_dir'] == 'backward' or \
       options['enc_dir'] == 'bidir':
        params = get_layer(options['encoder'])[0](options,
                                                  params,
                                                  prefix='word_encoder_r',
                                                  nin=options['dim_word'],
                                                  dim=options['encoder_dim'])

    num_directions = 2 if options['enc_dir'] == 'bidir' else 1
    ctxdim = [num_directions * dim_ for dim_ in options['encoder_dim']]
    # ctxdim = num_directions * options['encoder_dim'][-1]

    # init_state
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_label_state',
                                nin=ctxdim,
                                nout=options['decoder_dim'])

    params['Lemb'] = norm_weight(
        options['n_labels'],
        options['dim_label'],
        scale=1/numpy.sqrt(options['n_labels'])
    )

    # decoder for target words
    if options['enc_dir'] == 'none':
        params = get_layer(options['decoder'])[0](options,
                                                  params,
                                                  prefix='label_decoder',
                                                  nin=options['dim_label'],
                                                  dim=options['decoder_dim'])
    else:
        params = get_layer(options['decoder'])[0](options,
                                                  params,
                                                  prefix='label_decoder',
                                                  nin=options['dim_label'],
                                                  dim=options['decoder_dim'],
                                                  dimctx=ctxdim[-1])
    # readout
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_label_logit_lstm',
                                nin=options['decoder_dim'][-1],
                                nout=options['dim_label'],
                                ortho=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_label_logit_prev',
                                nin=options['dim_label'],
                                nout=options['dim_label'],
                                ortho=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_label_logit_ctx',
                                nin=ctxdim[-1],
                                nout=options['dim_label'],
                                ortho=False)
    params = get_layer('ff')[0](options,
                                params,
                                prefix='ff_label_logit',
                                nin=options['dim_label'],
                                nout=options['n_labels'])

    return params


# build a training model
def build_model(tparams, options):
    """ Build a computational graph for model training

    Parameters:
    -------
        tparams : dict
            Model parameters
        options : dict
            Model configurations

    Returns:
    -------
        trng : Randomstream in Theano

        use_noise : TheanoSharedVariable

        encoder_vars : list
            This return value contains TheanoVariables used to construct
            part of the computational graph, especially used in the `encoder`.

        decoder_vars : list
            This return value contains TheanoVariables used to construct
            part of the computational graph, especially used in the `decoder`.

        opt_ret : dict

        costs : list
            A list of costs at word-level or
            both word-level and character-level costs

    """
    opt_ret = dict()

    trng = MRG_RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    word_dp = options['word_drop_prob']

    word_dropout = {'use_noise': use_noise,
                    'dp': word_dp,
                    'trng': trng}

    if options['input_token_level'] == 'character':
        # description string: #characters x #words x #samples
        x = tensor.tensor3('x', dtype='int64')
        x_mask = tensor.tensor3('x_mask', dtype='float32')

        n_characters = x.shape[0]
        n_words = x.shape[1]
        n_samples = x.shape[2]

        # extract character embeddings
        cemb = tparams['Cemb'][x.flatten()]
        cemb = cemb.reshape(
            [
                n_characters,
                n_words,
                n_samples,
                options['dim_char']
            ]
        )

        # compute hidden states of character embeddings in the source language
        cproj = get_layer('conv')[1](
            tparams,
            cemb,
            options,
            options['char_emb_filters'],
            prefix='char_emb_conv',
            mask=x_mask)

        # XXX max over time: # n_chars x # n_words x # n_samples x dim
        word_rep = tensor.max(cproj, axis=0)
        word_repr = word_rep[::-1]

        assert word_rep.ndim == 3

        # NOTE w_mask should be a matrix of elements 1 or 0
        w_mask = x_mask.sum(0)
        w_mask = tensor.clip(w_mask, 0.0, 1.0)
        wr_mask = w_mask[::-1]
    else:
        # description string: #words x #samples
        x = tensor.matrix('x', dtype='int64')
        x_mask = tensor.matrix('x_mask', dtype='float32')

        n_words = x.shape[0]
        n_samples = x.shape[1]

        # for the backward rnn, we just need to invert x and x_mask
        xr = x[::-1]    # reverse words
        xr_mask = x_mask[::-1]

        # extract word embeddings
        wemb = tparams['Wemb'][x.flatten()]
        wemb = wemb.reshape(
            [
                n_words,
                n_samples,
                options['dim_word']
            ]
        )
        wembr = tparams['Wemb'][xr.flatten()]
        wembr = wembr.reshape(
            [
                n_words,
                n_samples,
                options['dim_word']
            ]
        )

        w_mask, wr_mask = x_mask, xr_mask
        word_rep, word_repr = wemb, wembr

        if options.get('fixed_embeddings', False):
            LOGGER.info('NOTE: word embeddings are NOT going to be updated!')
            word_rep = theano.gradient.zero_grad(word_rep)
            word_repr = theano.gradient.zero_grad(word_repr)

    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    encoder_vars = [x, x_mask]  # collection of varaibles used in the encoder
    decoder_vars = [y, y_mask]  # used in the decoder

    n_labels = y.shape[0]

    if options['enc_dir'] == 'forward' or \
       options['enc_dir'] == 'bidir':
        # hidden states for new word embeddings for the forward rnn
        src_proj = get_layer(options['encoder'])[1](tparams,
                                                    word_rep,
                                                    options['encoder_dim'],
                                                    options,
                                                    prefix='word_encoder',
                                                    mask=w_mask,
                                                    **word_dropout)

    if options['enc_dir'] == 'backward' or \
       options['enc_dir'] == 'bidir':
        # hidden states for new word embeddings for the backward rnn
        src_projr = get_layer(options['encoder'])[1](tparams,
                                                     word_repr,
                                                     options['encoder_dim'],
                                                     options,
                                                     prefix='word_encoder_r',
                                                     mask=wr_mask,
                                                     **word_dropout)

    if options['enc_dir'] == 'forward':
        ctx = src_proj[-1][0]
        ctx_mean = [state[0][-1] for state in src_proj]
    elif options['enc_dir'] == 'backward':
        ctx = src_projr[-1][0]
        ctx_mean = [state[0][-1] for state in src_projr]
    elif options['enc_dir'] == 'bidir':
        # context will be the concatenation of forward and backward rnns
        ctx = concatenate([src_proj[-1][0], src_projr[-1][0][::-1]],
                          axis=src_proj[-1][0].ndim - 1)
        ctx_mean = [concatenate([fw_state[0][-1], bw_state[0][-1]],
                                axis=fw_state[0].ndim-2)
                    for fw_state, bw_state in zip(src_proj, src_projr)]
    elif options['enc_dir'] == 'none':
        # since there is no source encoder, the following list contains only
        # a single element, which coressponds to a matrix of N x D where N is
        # the number of instances in a batch and D is the dimensionality of
        # instance vectors.

        ctx_mean = [(word_rep * x_mask[:, :, None]).sum(axis=0)]
        ctx_mean = [ctx_mean[0] / x_mask.sum(axis=0)[:, None]]    # if avg

    assert ctx_mean[0].ndim == 2
    assert type(ctx_mean) is list
    if options['enc_dir'] != 'none':
        assert len(ctx_mean) == len(options['encoder_dim'])

    # word embedding (target)
    label_emb = tparams['Lemb'][y.flatten()]
    label_emb = label_emb.reshape([n_labels, n_samples,
                                  options['dim_label']])

    # We will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    label_emb_shifted = tensor.zeros_like(label_emb)
    label_emb_shifted = tensor.set_subtensor(
        label_emb_shifted[1:], label_emb[:-1])
    label_emb = label_emb_shifted

    # initial decoder state
    init_state = get_layer('ff')[1](tparams,
                                    ctx_mean,
                                    options,
                                    prefix='ff_label_state')

    assert len(ctx_mean) == len(init_state)

    if options['enc_dir'] != 'none':
        # decoder - pass through the decoder conditional gru with attention
        label_proj = get_layer(options['decoder'])[1](tparams,
                                                      label_emb,
                                                      options['decoder_dim'],
                                                      options,
                                                      prefix='label_decoder',
                                                      mask=y_mask,
                                                      context=ctx,
                                                      context_mask=w_mask,
                                                      one_step=False,
                                                      init_state=init_state,
                                                      **word_dropout)

        # proj_h: hidden states of the decoder gru
        # ctxs: weighted averages of context, generated by attention module
        # dec_alphas: weights (alignment matrix)
        # num trg words x batch size x num src words

        proj_h_all, ctxs, opt_ret['dec_alphas'] = label_proj
        proj_h = proj_h_all[-1]     # pick activations of the last hidden layer

    else:
        assert options['decoder'] == 'gru'
        label_proj = get_layer(options['decoder'])[1](tparams,
                                                      label_emb,
                                                      options['decoder_dim'],
                                                      options,
                                                      prefix='label_decoder',
                                                      mask=y_mask,
                                                      one_step=False,
                                                      init_state=init_state,
                                                      **word_dropout)

        proj_h = label_proj[-1][0]

        # XXX note that all of the context vectors are same in this case.
        ctxs = tensor.tile(ctx_mean[0], (proj_h.shape[0], 1, 1))

        # XXX no alignment matrix
        opt_ret['dec_alphas'] = tensor.zeros((y.shape[0],
                                              y.shape[1],
                                              x.shape[0]))

    # compute word probabilities
    # hidden at t to word at t
    label_logit_lstm = get_layer('ff')[1](tparams,
                                          proj_h,
                                          options,
                                          prefix='ff_label_logit_lstm',
                                          activ=None)
    # combined representation of label at t-1 to label at t
    label_logit_prev = get_layer('ff')[1](tparams,
                                          label_emb,
                                          options,
                                          prefix='ff_label_logit_prev',
                                          activ=None)
    # context at t to label at t
    label_logit_ctx = get_layer('ff')[1](tparams,
                                         ctxs,
                                         options,
                                         prefix='ff_label_logit_ctx',
                                         activ=None)

    label_logit = tensor.tanh(
        label_logit_lstm +
        label_logit_prev +
        label_logit_ctx
    )

    if options['use_dropout']:
        label_logit = dropout_layer(label_logit, use_noise, word_dp, trng)

    label_logit = get_layer('ff')[1](tparams,
                                     label_logit,
                                     options,
                                     prefix='ff_label_logit',
                                     activ=None)
    label_logit_shp = label_logit.shape
    label_logit = label_logit.reshape(
        [
            label_logit_shp[0] * label_logit_shp[1],
            label_logit_shp[2]
        ]
    )

    label_probs = tensor.nnet.softmax(label_logit)

    # compute cost for labels
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * \
        options['n_labels'] + y_flat

    label_cost = -tensor.log(label_probs.flatten()[y_flat_idx])
    label_cost = label_cost.reshape([y.shape[0], y.shape[1]])
    label_cost = (label_cost * y_mask).sum(0)
    label_cost.name = 'label_cost'

    costs = [label_cost]

    return trng, use_noise, encoder_vars, decoder_vars, opt_ret, costs


# build a sampler
def build_sampler(tparams, options, trng, use_noise):

    if options['input_token_level'] == 'character':
        x = tensor.tensor3('x', dtype='int64')
        x_mask = tensor.tensor3('x_mask', dtype='float32')

        n_characters = x.shape[0]
        n_words = x.shape[1]
        n_samples = x.shape[2]

        # extract character embeddings
        cemb = tparams['Cemb'][x.flatten()]
        cemb = cemb.reshape(
            [
                n_characters,
                n_words,
                n_samples,
                options['dim_char']
            ]
        )

        # compute hidden states of character embeddings
        cproj = get_layer('conv')[1](
            tparams,
            cemb,
            options,
            options['char_emb_filters'],
            prefix='char_emb_conv',
            mask=x_mask)

        word_rep = tensor.max(cproj, axis=0)
        word_repr = word_rep[::-1]

        w_mask = x_mask.sum(0)
        w_mask = tensor.clip(w_mask, 0.0, 1.0)
        wr_mask = w_mask[::-1]
    else:
        x = tensor.matrix('x', dtype='int64')
        x_mask = tensor.matrix('x_mask', dtype='float32')

        n_words = x.shape[0]
        n_samples = x.shape[1]

        # for the backward rnn, we just need to invert x and x_mask
        xr = x[::-1]    # reverse words
        xr_mask = x_mask[::-1]

        # extract word embeddings
        wemb = tparams['Wemb'][x.flatten()]
        wemb = wemb.reshape(
            [
                n_words,
                n_samples,
                options['dim_word']
            ]
        )
        wembr = tparams['Wemb'][xr.flatten()]
        wembr = wembr.reshape(
            [
                n_words,
                n_samples,
                options['dim_word']
            ]
        )

        w_mask, wr_mask = x_mask, xr_mask
        word_rep, word_repr = wemb, wembr

    encoder_vars = [x, x_mask]

    if options['enc_dir'] == 'forward' or \
       options['enc_dir'] == 'bidir':
        # hidden states for new word embeddings for the forward rnn
        src_proj = get_layer(options['encoder'])[1](tparams,
                                                    word_rep,
                                                    options['encoder_dim'],
                                                    options,
                                                    prefix='word_encoder',
                                                    mask=w_mask)

    if options['enc_dir'] == 'backward' or \
       options['enc_dir'] == 'bidir':
        # hidden states for new word embeddings for the backward rnn
        src_projr = get_layer(options['encoder'])[1](tparams,
                                                     word_repr,
                                                     options['encoder_dim'],
                                                     options,
                                                     prefix='word_encoder_r',
                                                     mask=wr_mask)

    if options['enc_dir'] == 'forward':
        ctx = src_proj[-1][0]
        ctx_mean = [state[0][-1] for state in src_proj]
    elif options['enc_dir'] == 'backward':
        ctx = src_projr[-1][0]
        ctx_mean = [state[0][-1] for state in src_projr]
    elif options['enc_dir'] == 'bidir':
        # concatenate forward and backward rnn hidden states
        ctx = concatenate([src_proj[-1][0], src_projr[-1][0][::-1]],
                          axis=src_proj[-1][0].ndim - 1)
        ctx_mean = [concatenate([fw_state[0][-1], bw_state[0][-1]],
                                axis=fw_state[0].ndim-2)
                    for fw_state, bw_state in zip(src_proj, src_projr)]
    elif options['enc_dir'] == 'none':
        ctx_mean = [(word_rep * x_mask[:, :, None]).sum(axis=0)]
        ctx_mean = [ctx_mean[0] / x_mask.sum(axis=0)[:, None]]    # if avg
        ctx = word_rep

    assert ctx_mean[0].ndim == 2
    assert len(ctx_mean) == len(options['encoder_dim'])

    init_state = get_layer('ff')[1](tparams,
                                    ctx_mean,
                                    options,
                                    prefix='ff_label_state')

    assert len(ctx_mean) == len(init_state)

    init_state = tensor.stack(init_state)
    # init_state: # layers x # samples x trg_hid_dim

    LOGGER.info('Building f_init')
    encoder_outs = [init_state, ctx]

    f_init = theano.function(encoder_vars, encoder_outs,
                             name='f_init', profile=False)

    f_inits = [f_init]

    # y: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.tensor3('init_state', dtype='float32')

    label_emb = tparams['Lemb'][y]

    # if it's the first word, emb should be all zero and it is indicated by -1
    label_emb = label_emb * (y[:, None] >= 0)

    init_state_ = \
        [
            init_state[l] for l in xrange(len(options['decoder_dim']))
        ]

    if options['enc_dir'] != 'none':
        # apply one step of conditional gru with attention
        label_proj = get_layer(options['decoder'])[1](tparams,
                                                      label_emb,
                                                      options['decoder_dim'],
                                                      options,
                                                      prefix='label_decoder',
                                                      mask=None,
                                                      context=ctx,
                                                      context_mask=w_mask,
                                                      one_step=True,
                                                      init_state=init_state_)

        # next_label_state: # layers x # samples x rnn hid dim
        next_label_state, ctxs, dec_alphas = label_proj

        proj_h = next_label_state[-1]

    else:
        assert options['decoder'] == 'gru'
        assert len(init_state_) == 1
        label_proj = get_layer(options['decoder'])[1](tparams,
                                                      label_emb,
                                                      options['decoder_dim'],
                                                      options,
                                                      prefix='label_decoder',
                                                      mask=None,
                                                      one_step=True,
                                                      init_state=init_state_[0])

        assert type(label_proj) == list and len(label_proj) == 1
        next_label_state = tensor.stack([label_proj[0][0]])
        proj_h = label_proj[-1][0]

        assert proj_h.ndim == label_emb.ndim, \
            'expected {}: actual {}'.format(label_emb.ndim, proj_h.ndim)

        # XXX note that all of the context vectors are same in this case.
        ctxs = (ctx * x_mask[:, :, None]).sum(axis=0) / \
            x_mask.sum(axis=0)[:, None]

        # XXX no alignment matrix
        dec_alphas = tensor.zeros((y.shape[0],
                                   x_mask.shape[0]))

    label_logit_lstm = get_layer('ff')[1](tparams,
                                          proj_h,
                                          options,
                                          prefix='ff_label_logit_lstm',
                                          activ=None)
    # characters in a label at t-1 to word at t
    label_logit_prev = get_layer('ff')[1](tparams,
                                          label_emb,
                                          options,
                                          prefix='ff_label_logit_prev',
                                          activ=None)
    label_logit_ctx = get_layer('ff')[1](tparams,
                                         ctxs,
                                         options,
                                         prefix='ff_label_logit_ctx',
                                         activ=None)
    label_logit = tensor.tanh(label_logit_lstm +
                              label_logit_prev +
                              label_logit_ctx)

    label_logit = get_layer('ff')[1](tparams,
                                     label_logit,
                                     options,
                                     prefix='ff_label_logit',
                                     activ=None)

    # compute the softmax probability
    next_label_probs = tensor.nnet.softmax(label_logit)

    # sample from softmax distribution to get the sample
    next_label_sample = trng.multinomial(pvals=next_label_probs).argmax(1)

    # compile a function to do the whole thing above, next label probability,
    # sampled label for the next target, next hidden state to be used
    LOGGER.info('Building f_label_next')
    f_lsamp_inps = [x_mask, y, ctx, init_state]
    f_lsamp_outs = [next_label_probs, next_label_sample, next_label_state,
                    dec_alphas]

    f_next = theano.function(f_lsamp_inps, f_lsamp_outs,
                             name='f_next', profile=False)

    f_nexts = [f_next]

    return f_inits, f_nexts


def gen_sample(tparams,
               f_inits,
               f_nexts,     # list of functions to generate outputs
               inps,
               options,
               trng=None,
               k=1,
               max_label_len=10,
               stochastic=True,
               argmax=False):

    assert len(f_inits) == len(f_nexts)

    if len(inps) == 2 and len(f_nexts) == 1:

        x, x_mask = inps
        f_init = f_inits[0]
        f_next = f_nexts[0]
    else:
        raise ValueError('The number of input variables should be equal to '
                         'the number of items in `f_nexts` multiplied by 2')

    assert max_label_len > 0

    # k is the beam size we have
    assert k >= 1

    fixed_length = False
    if 'label_type' in options and options['label_type'] == 'binary':
        fixed_length = True

    live_k = 1

    solutions_ds = [('num_samples', 0), ('samples', []),
                    ('alignments', []), ('scores', [])]

    hypotheses_ds = [
        ('num_samples', live_k),
        ('samples', [[]] * live_k),
        ('alignments', [[]] * live_k),
        ('scores', numpy.zeros(live_k).astype('float32')),
    ]

    solutions = OrderedDict(solutions_ds)
    hypotheses = OrderedDict(hypotheses_ds)

    def _check_stop_condition(solutions, hypotheses, k):
        return solutions['num_samples'] >= k or hypotheses['num_samples'] < 1

    # get initial state of decoder rnn and encoder context
    # ctx0 is 3d tensor of hidden states for the input sentence
    # next_state is a summary of hidden states for the input setence
    # ctx0: (# src words x # sentence (i.e., 1) x # hid dim)
    # next_state: (# sentences (i.e., 1) x # hid dim of the target setence)
    encoder_outs = f_init(*inps)
    next_label_state, ctx0 = encoder_outs[0], encoder_outs[1]

    next_l = -1 * numpy.ones((1, )).astype('int64')  # bos indicator
    l_history = -1 * numpy.ones((1, 1)).astype('int64')

    for ii in xrange(max_label_len):
        live_k = hypotheses['num_samples']

        # NOTE `hyp_samples` is initailized by a list with a single empty list
        # repeat the contexts the number of hypotheses
        # (corresponding to the number of setences)
        # (# src words x 1 x hid dim) -> (# src words x # next_hyp x hid dim)
        ctx = numpy.tile(ctx0, [1, live_k, 1])
        x_mask_ = numpy.tile(x_mask, [1, live_k])

        # inputs to sample word candidates
        lsamp_inps = [x_mask_, next_l, ctx, next_label_state]

        # generate a word for the given last hidden states
        # and previously generated words
        lsamp_outs = f_next(*lsamp_inps)

        next_p = lsamp_outs[0]
        # next_label_sample = wsamp_outs[1]  XXX Not used
        next_label_state = lsamp_outs[2]
        next_alphas = lsamp_outs[3]

        # preparation of inputs to beam search
        # XXX change next_word_state: (# layers x # samples x hid dim)
        #       -> (# samples x hid dim x # layers)
        next_label_state = next_label_state.transpose([1, 2, 0])
        beam_state = [next_label_state, next_p, next_alphas]

        # perform beam search to generate most probable word sequence
        # with limited budget.
        solutions, hypotheses = \
            beam_search(solutions, hypotheses, beam_state,
                        decode_char=False, k=k,
                        fixed_length=fixed_length)

        if _check_stop_condition(solutions, hypotheses, k):
            break

        # get the last single word for each hypothesis
        next_l = numpy.array([w[-1] for w in hypotheses['samples']])
        l_history = numpy.array(hypotheses['samples'])
        next_label_state = numpy.array(hypotheses['states'])
        next_label_state = next_label_state.transpose([2, 0, 1])

    # dump every remaining one
    if hypotheses['num_samples'] > 0:
        for idx in xrange(hypotheses['num_samples']):
            solutions['samples'].append(
                hypotheses['samples'][idx])
            solutions['scores'].append(
                hypotheses['scores'][idx])
            solutions['alignments'].append(
                hypotheses['alignments'][idx])

    return solutions


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, options, stream):
    probs = []

    n_done = 0

    for x, x_mask, y, y_mask in stream.get_epoch_iterator():
        n_done += len(x)

        if options['label_type'] == 'binary':
            new_y, new_y_mask = mul2bin(y, y_mask, options['n_bins'])
            y, y_mask = new_y, new_y_mask

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

        encoder_inps = [x, x_mask]
        decoder_inps = [y, y_mask]

        inps = encoder_inps + decoder_inps

        pprobs = f_log_probs(*inps)

        for pp in pprobs:
            probs.append(pp)

        if not numpy.isfinite(numpy.mean(probs)):
            raise RuntimeError('non-finite probabilities')

    return numpy.array(probs)


def save_params(params, filename, symlink=None):
    """Save the parameters.

    Saves the parameters as an ``.npz`` file. It optionally also creates a
    symlink to this archive.

    """
    numpy.savez(filename, **params)
    if symlink:
        if os.path.lexists(symlink):
            os.remove(symlink)
        os.symlink(filename, symlink)

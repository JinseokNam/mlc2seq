import numpy
from six.moves import xrange

import theano
from theano import tensor

from utils import uniform_weight, ortho_weight, norm_weight, dropout_layer


def zero_vector(length):
    return numpy.zeros((length, )).astype('float32')


# utility function to slice a tensor
def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]


def _conv(x, tparams, filter_sizes, param_prefix):
    # assert x.ndim == 4

    # TODO refactor this function to handle both 3D and 4D tensor properly
    n_chars = x.shape[0]
    n_words = x.shape[1]

    input_ndim = x.ndim
    if x.ndim == 3:
        n_samples = 1
        char_dim = x.shape[2]
    elif x.ndim == 4:
        n_samples = x.shape[2]
        char_dim = x.shape[3]

    x = x.reshape([n_chars, n_words * n_samples, char_dim])
    x = x.transpose((1, 0, 2))

    total_proj = []
    for filter_ in filter_sizes:
        convolved = tensor.nnet.conv2d(
            x[:, None, :, :],
            tparams['%s_W%d' % (param_prefix, filter_)],
            border_mode=(filter_/2, 0))

        num_filters = \
            tparams['%s_W%d' % (param_prefix, filter_)].shape[0]
        conv_shp = convolved.shape
        proj = convolved.reshape([conv_shp[0], conv_shp[1], conv_shp[2]])
        proj = proj.reshape([n_words, n_samples, num_filters, n_chars])
        proj = proj.transpose((3, 0, 1, 2))
        proj = proj + tparams['%s_b%d' % (param_prefix, filter_)]
        total_proj.append(proj)

    proj = tensor.concatenate(total_proj, axis=total_proj[0].ndim-1)
    # n_chars x n_words x n_samples x num_filters
    # proj = tensor.stack(total_proj).mean(0)

    # XXX what was the reasoning to the following lines?
    if input_ndim == 3 and proj.ndim == 4:
        prj_shp = proj.shape
        proj = proj.reshape([prj_shp[0], prj_shp[1], prj_shp[3]])

    return proj


def _gru(mask, x_t2gates, x_t2prpsl, h_tm1, U, Ux, activ=tensor.tanh):

    dim = U.shape[0]    # dimension of hidden states

    # concatenated activations of the gates in a GRU
    activ_gates = tensor.nnet.sigmoid(x_t2gates + tensor.dot(h_tm1, U))

    # reset and update gates
    reset_gate = _slice(activ_gates, 0, dim)
    update_gate = _slice(activ_gates, 1, dim)

    # compute the hidden state proposal
    in_prpsl = x_t2prpsl + reset_gate * tensor.dot(h_tm1, Ux)
    h_prpsl = activ(in_prpsl) if activ else in_prpsl

    # leaky integrate and obtain next hidden state
    h_t = update_gate * h_tm1 + (1. - update_gate) * h_prpsl

    # if this time step is not valid, discard the current hidden states
    # obtained above and copy the previous hidden states to the current ones.
    if mask.ndim == 1:
        h_t = mask[:, None] * h_t + (1. - mask)[:, None] * h_tm1
    elif mask.ndim == 2:
        h_t = mask[:, :, None] * h_t + (1. - mask)[:, :, None] * h_tm1

    return h_t


def _compute_alignment(h_tm1,       # s_{i-1}
                       prj_annot,   # proj annotations: U_a * h_j for all j
                       Wd_att, U_att,
                       context_mask=None):

    # W_a * s_{i-1}
    prj_h_tm1 = tensor.dot(h_tm1, Wd_att)

    # tanh(W_a * s_{i-1} + U_a * h_j) for all j
    nonlin_proj = tensor.tanh(prj_h_tm1[None, :, :] + prj_annot)

    # v_a^{T} * tanh(.)
    alpha = tensor.dot(nonlin_proj, U_att)
    alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
    alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
    if context_mask:
        alpha = alpha * context_mask
    alpha = alpha / alpha.sum(0, keepdims=True)

    return alpha


def _cond_gru(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, ctx_mask,
              U, Wc, W_comb_att, U_att, Ux, Wcx):

    # attention
    alpha = _compute_alignment(h_, pctx_,
                               W_comb_att, U_att,
                               context_mask=ctx_mask)

    ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

    new_x_ = x_ + tensor.dot(ctx_, Wc)
    new_xx_ = xx_ + tensor.dot(ctx_, Wcx)

    h = _gru(m_, new_x_, new_xx_, h_, U, Ux)

    return h, ctx_, alpha.T  # pstate_, preact, preactx, r, u


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options,
                       param,
                       prefix='ff',
                       nin=None,
                       nout=None,
                       ortho=True):

    if type(nin) is int and type(nout) is int:
        param[prefix + '_W'] = uniform_weight(nin, nout)
        param[prefix + '_b'] = zero_vector(nout)
    else:
        assert type(nout) is list

        if type(nin) is int:
            nin = [nin] + nout[:-1]
        elif type(nin) is list:
            assert len(nin) == len(nout)

        for l, (in_dim, out_dim) in enumerate(zip(nin, nout)):
            prefix_ = prefix + '_%d' % l
            param[prefix_ + '_W'] = uniform_weight(in_dim, out_dim)
            param[prefix_ + '_b'] = zero_vector(out_dim)

    return param


def fflayer(tparams,
            state_below,
            options,
            prefix='rconv',
            activ=tensor.tanh,
            **kwargs):
    if type(state_below) is list:
        n_layers = len(state_below)
        h = [None] * n_layers
        for l in xrange(n_layers):
            prefix_ = prefix + '_%d' % l
            h[l] = (tensor.dot(state_below[l], tparams[prefix_ + '_W']) +
                    tparams[prefix_ + '_b'])
            h[l] = activ(h[l]) if activ else h[l]
            if options['use_dropout'] and ('use_noise' in kwargs and
                                           'dp' in kwargs and
                                           'trng' in kwargs):
                h[l] = dropout_layer(h[l], kwargs['use_noise'],
                                     kwargs['dp'], kwargs['trng'])
    else:
        h = (tensor.dot(state_below, tparams[prefix + '_W']) +
             tparams[prefix + '_b'])
        h = activ(h) if activ else h

        if options['use_dropout'] and ('use_noise' in kwargs and
                                       'dp' in kwargs and
                                       'trng' in kwargs):
            h = dropout_layer(h, kwargs['use_noise'],
                              kwargs['dp'], kwargs['trng'])

    return h


def param_init_conv(options,
                    param,
                    prefix='conv',
                    nin=None,
                    nout=None,
                    filter_sizes=None):

    num_filters = nout
    for filter_ in filter_sizes:
        W = norm_weight(filter_*nin, num_filters)
        W = W.reshape([num_filters, 1, filter_, nin])
        b = zero_vector(num_filters)
        param[prefix + '_W%d' % filter_] = W
        param[prefix + '_b%d' % filter_] = b

    return param


def conv_layer(tparams,
               state_below,
               options,
               filter_sizes,
               prefix='conv',
               activ=tensor.tanh,
               **kwargs):

    h = _conv(state_below, tparams, filter_sizes, param_prefix=prefix)
    h = activ(h) if activ else h

    return h


# GRU layer
def param_init_gru(options, param, prefix='gru', nin=None, dim=None,
                   input_conv=False):

    def _init_gru(in_dim, hid_dim, prefix_):
        param[prefix_ + '_W'] = numpy.concatenate(
            [
                uniform_weight(in_dim, hid_dim),
                uniform_weight(in_dim, hid_dim)
            ],
            axis=1)
        param[prefix_ + '_Wx'] = uniform_weight(in_dim, hid_dim)

        param[prefix_ + '_U'] = numpy.concatenate(
            [
                ortho_weight(hid_dim), ortho_weight(hid_dim)
            ],
            axis=1)
        param[prefix_ + '_b'] = zero_vector(2 * hid_dim)

        param[prefix_ + '_Ux'] = ortho_weight(hid_dim)
        param[prefix_ + '_bx'] = zero_vector(hid_dim)

    assert type(nin) is int

    if type(dim) is int:
        _init_gru(nin, dim, prefix)
    elif type(dim) is list:
        in_dim = nin
        for l, hid_dim in enumerate(dim):
            prefix_ = prefix + '_%d' % l
            _init_gru(in_dim, hid_dim, prefix_)
            in_dim = hid_dim

    return param


def gru_layer(tparams,
              state_below,
              dims,
              options,
              prefix='gru',
              mask=None,
              input_conv=False,
              one_step=False,
              init_state=None,
              **kwargs):

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 4:
        n_samples = state_below.shape[2]
    elif state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert type(dims) is list

    n_layers = len(dims)

    # state_below is the input word embeddings
    hidden_states = [None] * n_layers
    if init_state is None:
        init_state = [None] * n_layers

    if mask is None:
        if one_step:
            mask = tensor.alloc(1., state_below.shape[0])
        else:
            mask = tensor.alloc(1., state_below.shape[0], 1)

    def _prepare_initial_state(input_x, n_samples, hid_dim, init_state_=None):
        if init_state_:
            assert input_x.ndim - 1 == init_state_.ndim, \
                ('The provided initial state is assumed to have '
                 'one less dimension than the input. (%d - 1) != %d') \
                % (input_x.ndim, init_state_.ndim)
            state = [init_state_]
        else:
            if input_x.ndim == 4:
                state = [tensor.alloc(0.,
                                      input_x.shape[1],
                                      n_samples, hid_dim)]
            else:
                state = [tensor.alloc(0., n_samples, hid_dim)]

        return state

    input_x = state_below
    for l, hid_dim in enumerate(dims):
        prefix_ = prefix + '_%d' % l

        proj_x_ = (tensor.dot(input_x, tparams[prefix_ + '_W']) +
                   tparams[prefix_ + '_b'])
        proj_xx = (tensor.dot(input_x, tparams[prefix_ + '_Wx']) +
                   tparams[prefix_ + '_bx'])

        # prepare scan arguments
        _step = _gru
        initial_state = _prepare_initial_state(input_x, n_samples,
                                               hid_dim, init_state[l])
        seqs = [mask, proj_x_, proj_xx]
        shared_vars = [tparams[prefix_ + '_U'], tparams[prefix_ + '_Ux']]

        if one_step:
            rval = _step(*(seqs + initial_state + shared_vars))
        else:
            rval, updates = theano.scan(_step,
                                        sequences=seqs,
                                        outputs_info=initial_state,
                                        non_sequences=shared_vars,
                                        name=prefix_ + '_layer',
                                        n_steps=nsteps,
                                        strict=True)

        if options['use_dropout'] and ('use_noise' in kwargs and
                                       'dp' in kwargs and
                                       'trng' in kwargs):
            rval = dropout_layer(rval, kwargs['use_noise'],
                                 kwargs['dp'], kwargs['trng'])

        hidden_states[l] = [rval]
        input_x = rval

    return hidden_states


# Conditional GRU layer with Attention
def param_init_gru_cond(options,
                        param,
                        prefix='gru_cond',
                        nin=None,
                        dim=None,
                        dimctx=None):

    assert type(dim) is list
    last_dim = dim[-1]

    param = param_init_gru(options, param, prefix=prefix, nin=nin, dim=dim)

    prefix_ = prefix + '_%d' % (len(dim) - 1)
    # context to LSTM
    param[prefix_ + '_Wc'] = numpy.concatenate(
        [
            uniform_weight(dimctx, last_dim), uniform_weight(dimctx, last_dim)
        ], axis=1
    )
    param[prefix_ + '_Wcx'] = uniform_weight(dimctx, last_dim)

    # attention: combined -> hidden
    param[prefix_ + '_W_comb_att'] = uniform_weight(last_dim, dimctx)

    # attention: context -> hidden
    param[prefix_ + '_Wc_att'] = uniform_weight(dimctx, dimctx)

    # attention: hidden bias
    param[prefix_ + '_b_att'] = zero_vector(dimctx)

    # attention:
    param[prefix_ + '_U_att'] = uniform_weight(dimctx, 1)

    return param


def gru_cond_layer(tparams,
                   state_below,
                   dims,
                   options,
                   prefix='gru_cond',
                   mask=None,
                   context=None,
                   one_step=False,
                   init_memory=None,
                   init_state=None,
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    if init_state:
        assert type(init_state) is list

    nsteps = state_below.shape[0]
    n_layers = len(dims)
    hidden_states = [None] * n_layers

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        if one_step:
            mask = tensor.alloc(1., nsteps)
        else:
            mask = tensor.alloc(1., nsteps, 1)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim: %d' % context.ndim

    prefix_ = prefix + '_%d' % (n_layers - 1)
    pctx_ = (tensor.dot(context, tparams[prefix_ + '_Wc_att']) +
             tparams[prefix_ + '_b_att'])

    def _prepare_initial_state(input_x, n_samples, hid_dim,
                               init_state_=None, attention=False):
        if init_state_:
            """
            assert input_x.ndim - 1 == init_state_.ndim, \
                ('The provided initial state is assumed to have '
                 'one less dimension than the input. (%d - 1) != %d') \
                % (input_x.ndim, init_state_.ndim)
            """
            state = [init_state_]
        else:
            state = [tensor.alloc(0., n_samples, hid_dim)]

        if attention:
            if one_step:
                state += [None, None]
            else:
                state += [
                        tensor.alloc(0., n_samples, context.shape[2]),
                        tensor.alloc(1., n_samples, context.shape[0])
                ]

        return state

    input_x = state_below
    for l, hid_dim in enumerate(dims):
        prefix_ = prefix + '_%d' % l
        proj_x_ = (tensor.dot(input_x, tparams[prefix_ + '_W']) +
                   tparams[prefix_ + '_b'])
        proj_xx = (tensor.dot(input_x, tparams[prefix_ + '_Wx']) +
                   tparams[prefix_ + '_bx'])

        seqs = [mask, proj_x_, proj_xx]
        initial_state = _prepare_initial_state(
            input_x, n_samples, hid_dim, init_state[l], l == n_layers-1
        )

        if l < n_layers - 1:
            _step = _gru
            shared_vars = [tparams[prefix_ + '_U'], tparams[prefix_ + '_Ux']]
            non_seqs = []
        else:
            _step = _cond_gru
            shared_vars = [tparams[prefix_ + '_U'], tparams[prefix_ + '_Wc'],
                           tparams[prefix_ + '_W_comb_att'],
                           tparams[prefix_ + '_U_att'],
                           tparams[prefix_ + '_Ux'], tparams[prefix_ + '_Wcx']]
            non_seqs = [pctx_, context, context_mask]

        if one_step:
            rval = _step(*(
                seqs + initial_state + non_seqs + shared_vars))
        else:
            rval, updates = theano.scan(
                _step,
                sequences=seqs,
                outputs_info=initial_state,
                non_sequences=non_seqs + shared_vars,
                name=prefix + '_layers',
                n_steps=nsteps,
                strict=True)

        if l < n_layers - 1:
            rval = [rval]

        if options['use_dropout'] and ('use_noise' in kwargs and
                                       'dp' in kwargs and
                                       'trng' in kwargs):
            rval[0] = dropout_layer(rval[0], kwargs['use_noise'],
                                    kwargs['dp'], kwargs['trng'])
        hidden_states[l] = rval[0]
        input_x = rval[0]

    assert len(rval) == 3

    hidden_states = tensor.stack(hidden_states)

    return hidden_states, rval[1], rval[2]


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': (param_init_fflayer, fflayer),
          'conv': (param_init_conv, conv_layer),
          'gru': (param_init_gru, gru_layer),
          'gru_cond': (param_init_gru_cond, gru_cond_layer)}


def get_layer(name):
    param_init, layer = layers[name]
    return param_init, layer

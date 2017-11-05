import io
import logging
from itertools import count

import numpy
import six

from fuel.datasets.text import TextFile
from fuel.transformers import Merge
from fuel.schemes import ConstantScheme
from fuel.transformers import (Batch, Cache, Mapping, SortMapping, Padding,
                               Filter, Transformer)

FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(__name__)

EOS_TOKEN = '<EOS>'  # 0
UNK_TOKEN = '<UNK>'  # 1
EOW_TOKEN = ' '


class Shuffle(Transformer):
    def __init__(self, data_stream, buffer_size, **kwargs):
        if kwargs.get('iteration_scheme') is not None:
            raise ValueError
        super(Shuffle, self).__init__(
            data_stream, produces_examples=data_stream.produces_examples,
            **kwargs)
        self.buffer_size = buffer_size
        self.cache = [[] for _ in self.sources]

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        if not self.cache[0]:
            self._cache()
        return tuple(cache.pop() for cache in self.cache)

    def _cache(self):
        temp_caches = [[] for _ in self.sources]
        for i in range(self.buffer_size):
            try:
                for temp_cache, data in zip(temp_caches,
                                            next(self.child_epoch_iterator)):
                    temp_cache.append(data)
            except StopIteration:
                if i:
                    pass
                else:
                    raise
        shuffled_indices = numpy.random.permutation(len(temp_caches[0]))
        for i in shuffled_indices:
            for temp_cache, cache in zip(temp_caches, self.cache):
                cache.append(temp_cache[i])


class SortLabels(Transformer):
    def __init__(self, data_stream, **kwargs):
        if kwargs.get('iteration_scheme') is not None:
            raise ValueError
        super(SortLabels, self).__init__(
            data_stream, produces_examples=data_stream.produces_examples,
            **kwargs)

    def transform_example(self, example):
        if 'target_labels' in self.sources:
            example = list(example)

            index = self.sources.index('target_labels')
            labels = example[index]
            example[index] = sorted(labels[:-1]) + [0]

            example = tuple(example)

        return example


def _source_length(sentence_pair):
    """Returns the length of the second element of a sequence.

    This function is used to sort sentence pairs by the length of the
    target sentence.

    """
    return len(sentence_pair[0])


def load_dict(filename, dict_size=0, include_unk=True, reverse=False):
    """Load vocab from TSV with words in last column."""
    assert type(reverse) is bool

    dict_ = {EOS_TOKEN: 0}
    if include_unk:
        dict_[UNK_TOKEN] = 1

    with io.open(filename, encoding='utf8') as f:
        if dict_size > 0:
            indices = range(dict_size + len(dict_) - 1, len(dict_) - 1, -1) \
                if reverse else range(len(dict_), dict_size + len(dict_))
        else:
            indices = count(len(dict_))
        dict_.update(zip(map(lambda x: x.rstrip('\n').split('\t')[-1], f),
                     indices))
    return dict_


def get_stream(source, target, source_input_dict, target_label_dict, batch_size,
               buffer_multiplier=100, input_token_level='word',
               n_input_tokens=0, n_labels=0, reverse_labels=False,
               max_input_length=None, max_label_length=None, pad_labels=True,
               is_sort=True):
    """Returns a stream over sentence pairs.

    Parameters
    ----------
    source : list
        A list of files to read source languages from.
    target : list
        A list of corresponding files in the target language.
    source_word_dict : str
        Path to a tab-delimited text file whose last column contains the
        vocabulary.
    target_label_dict : str
        See `source_char_dict`.
    batch_size : int
        The minibatch size.
    buffer_multiplier : int
        The number of batches to load, concatenate, sort by length of
        source sentence, and split again; this makes batches more uniform
        in their sentence length and hence more computationally efficient.
    n_source_words : int
        The number of words in the source vocabulary. Pass 0 (default) to
        use the entire vocabulary.
    n_target_labels : int
        See `n_chars_source`.

    """
    if len(source) != len(target):
        raise ValueError("number of source and target files don't match")

    # Read the dictionaries
    dicts = [load_dict(source_input_dict, dict_size=n_input_tokens),
             load_dict(target_label_dict, dict_size=n_labels,
                       reverse=reverse_labels, include_unk=False)]

    # Open the two sets of files and merge them
    streams = [
        TextFile(source, dicts[0], level=input_token_level, bos_token=None,
                 eos_token=EOS_TOKEN, encoding='utf-8').get_example_stream(),
        TextFile(target, dicts[1], level='word', bos_token=None,
                 unk_token=None,
                 eos_token=EOS_TOKEN, encoding='utf-8').get_example_stream()
    ]
    merged = Merge(streams, ('source_input_tokens', 'target_labels'))
    if reverse_labels:
        merged = SortLabels(merged)

    # Filter sentence lengths
    if max_input_length or max_label_length:
        def filter_pair(pair):
            src_input_tokens, trg_labels = pair
            src_input_ok = (not max_input_length) or \
                len(src_input_tokens) <= (max_input_length + 1)
            trg_label_ok = (not max_label_length) or \
                len(trg_labels) <= (max_label_length + 1)

            return src_input_ok and trg_label_ok

        merged = Filter(merged, filter_pair)

    # Batches of approximately uniform size
    large_batches = Batch(
        merged,
        iteration_scheme=ConstantScheme(batch_size * buffer_multiplier)
    )
    # sorted_batches = Mapping(large_batches, SortMapping(_source_length))
    # batches = Cache(sorted_batches, ConstantScheme(batch_size))
    # shuffled_batches = Shuffle(batches, buffer_multiplier)
    # masked_batches = Padding(shuffled_batches,
    #                          mask_sources=('source_chars', 'target_labels'))
    if is_sort:
        sorted_batches = Mapping(large_batches, SortMapping(_source_length))
    else:
        sorted_batches = large_batches
    batches = Cache(sorted_batches, ConstantScheme(batch_size))
    mask_sources = ('source_input_tokens', 'target_labels')
    masked_batches = Padding(batches, mask_sources=mask_sources)

    return masked_batches


def load_data(src, trg,
              valid_src, valid_trg,
              input_vocab,
              label_vocab,
              n_input_tokens,
              n_labels,
              reverse_labels,
              input_token_level,
              batch_size, valid_batch_size,
              max_input_length, max_label_length):
    LOGGER.info('Loading data')

    dictionaries = [input_vocab, label_vocab]
    datasets = [src, trg]
    valid_datasets = [valid_src, valid_trg]

    # load dictionaries and invert them
    vocabularies = [None] * len(dictionaries)
    vocabularies_r = [None] * len(dictionaries)
    vocab_size = [n_input_tokens, n_labels]
    for ii, dd in enumerate(dictionaries):
        vocabularies[ii] = load_dict(dd, dict_size=vocab_size[ii]) if ii == 0 \
                           else load_dict(dd, dict_size=vocab_size[ii],
                                          reverse=reverse_labels,
                                          include_unk=False)
        vocabularies_r[ii] = dict()
        for kk, vv in six.iteritems(vocabularies[ii]):
            vocabularies_r[ii][vv] = kk

    train_stream = get_stream([datasets[0]],
                              [datasets[1]],
                              dictionaries[0],
                              dictionaries[1],
                              n_input_tokens=n_input_tokens,
                              n_labels=n_labels,
                              reverse_labels=reverse_labels,
                              input_token_level=input_token_level,
                              batch_size=batch_size,
                              max_input_length=max_input_length,
                              max_label_length=max_label_length)
    valid_stream = get_stream([valid_datasets[0]],
                              [valid_datasets[1]],
                              dictionaries[0],
                              dictionaries[1],
                              n_input_tokens=n_input_tokens,
                              n_labels=n_labels,
                              reverse_labels=reverse_labels,
                              input_token_level=input_token_level,
                              max_input_length=max_input_length,
                              batch_size=valid_batch_size)

    return vocabularies_r, train_stream, valid_stream


def load_test_data(src,
                   trg,
                   input_vocab,
                   label_vocab,
                   n_input_tokens,
                   n_labels,
                   reverse_labels,
                   input_token_level,
                   batch_size,
                   max_input_length):
    LOGGER.info('Loading test data')

    dictionaries = [input_vocab, label_vocab]
    datasets = [src, trg]

    # load dictionaries and invert them
    vocabularies = [None] * len(dictionaries)
    vocabularies_r = [None] * len(dictionaries)
    vocab_size = [n_input_tokens, n_labels]
    for ii, dd in enumerate(dictionaries):
        vocabularies[ii] = load_dict(dd, dict_size=vocab_size[ii]) if ii == 0 \
                           else load_dict(dd, dict_size=vocab_size[ii],
                                          reverse=reverse_labels,
                                          include_unk=False)
        vocabularies_r[ii] = dict()
        for kk, vv in six.iteritems(vocabularies[ii]):
            vocabularies_r[ii][vv] = kk

    test_stream = get_stream([datasets[0]],
                             [datasets[1]],
                             dictionaries[0],
                             dictionaries[1],
                             n_input_tokens=n_input_tokens,
                             n_labels=n_labels,
                             reverse_labels=reverse_labels,
                             input_token_level=input_token_level,
                             batch_size=batch_size,
                             max_input_length=max_input_length,
                             max_label_length=None,
                             pad_labels=False,
                             is_sort=False)

    return vocabularies_r, test_stream

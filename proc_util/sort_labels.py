import argparse
import io
from itertools import count


def sort_labels(input_filepath, label_vocab_path, output_filepath, reverse):

    dict_size = 1000000
    with io.open(label_vocab_path, encoding='utf8') as f:
        label_vocab = dict()
        if dict_size > 0:
            indices = range(len(label_vocab), dict_size)
        else:
            indices = count(len(label_vocab))
        label_vocab.update(zip(map(
            lambda x: x.rstrip('\n').split('\t')[-1], f), indices))

    with io.open(input_filepath, encoding='utf-8') as fin,\
            io.open(output_filepath, 'w', encoding='utf-8') as fout:

        for line in fin:
            labels = line.strip().split()

            label_index_pairs = [
                (l_, label_vocab[l_]) for l_ in labels if l_ in label_vocab
            ]
            '''
            print type(label_index_pairs)
            print type(label_index_pairs[0])
            print type(label_index_pairs[0][0])
            print type(label_index_pairs[0][0])
            '''
            sorted_labels = sorted(label_index_pairs,
                                   key=lambda x: x[1], reverse=reverse)

            sorted_labels = [
                lp[0] for lp in sorted_labels
            ]

            fout.write('%s\n' % u' '.join(sorted_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--label_vocab', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--reverse', dest='reverse', action='store_true')
    parser.set_defaults(reverse=False)

    args = parser.parse_args()

    sort_labels(args.input, args.label_vocab, args.output, args.reverse)

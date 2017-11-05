import argparse
import io
import itertools


def delete_instances(input_data_path, input_label_path,
                     output_data_path, output_label_path, label_vocab_path):

    label_vocab = set()

    with io.open(label_vocab_path, encoding='utf-8') as fin:
        for line in fin:
            label = line.strip().split('\t')[1]
            if label not in label_vocab:
                label_vocab.add(label)

    with io.open(input_data_path, encoding='utf-8') as fin_data, \
        io.open(input_label_path, encoding='utf-8') as fin_label, \
        io.open(output_data_path, 'w', encoding='utf-8') as fout_data, \
            io.open(output_label_path, 'w', encoding='utf-8') as fout_label:

        def check_labels(fin_data, fin_label, fout_data, fout_label):
            for dd, ll in itertools.izip(fin_data, fin_label):
                ll_ = [l for l in ll.split() if l in label_vocab]
                if len(ll_) == 0:
                    continue

                fout_data.write('%s\n' % dd.strip())
                fout_label.write('%s\n' % ' '.join(ll_))

        check_labels(fin_data, fin_label, fout_data, fout_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--out_data', type=str, required=True)
    parser.add_argument('--out_label', type=str, required=True)
    parser.add_argument('--label_vocab', type=str, required=True)

    args = parser.parse_args()

    delete_instances(args.data, args.label, args.out_data, args.out_label,
                     args.label_vocab)

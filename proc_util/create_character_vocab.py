import argparse
import io
import operator


def generate_character_vocab(input_filepath, output_filepath):

    with io.open(input_filepath, encoding='utf-8') as fin,\
            io.open(output_filepath, 'w', encoding='utf-8') as fout:

        char_dict = dict()

        for line in fin:
            for c in line.strip():
                if c not in char_dict:
                    char_dict[c] = 0

                char_dict[c] += 1

        sorted_vocab = sorted(char_dict.items(), key=operator.itemgetter(1),
                              reverse=True)

        for ch, val in sorted_vocab:
            fout.write('%d\t%s\n' % (val, ch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    generate_character_vocab(args.input, args.output)

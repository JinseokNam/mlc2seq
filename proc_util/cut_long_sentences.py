import argparse
import io


def cut_sentences(input_filepath, output_filepath, max_length, level='word'):

    with io.open(input_filepath, encoding='utf-8') as fin, \
         io.open(output_filepath, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.replace('\r\n', '').strip()
            if level == 'word':
                line = line.split()

            if len(line) > max_length:
                output = ' '.join(line[:max_length]) if level == 'word' \
                    else line[:max_length].strip()
            else:
                output = ' '.join(line) if level == 'word' else line

            fout.write('%s\n' % output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--level', type=str, default='character')
    parser.add_argument('--max', type=int, required=True)

    args = parser.parse_args()

    output_filepath = args.output
    cut_sentences(args.input, args.output, args.max, args.level)

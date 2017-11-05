#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import io
import argparse


def extract_title(wikipedia_header):
    title = wikipedia_header.strip().replace(
        '<', '').replace('>', '').split('title=')[-1]
    title_length = len(title)
    return re.sub('\s+', '_', title[1:title_length - 1]).strip()


def main(markup_data_path, body_path, title_path):
    with io.open(markup_data_path, encoding='utf-8') as fin, \
        io.open(body_path, 'w', encoding='utf-8') as fout_body, \
            io.open(title_path, 'w', encoding='utf-8') as fout_title:

        start_writing = False
        num_docs = 0

        while True:
            line = fin.readline()
            if not line:
                break

            if line.startswith('<mydoc'):
                title = extract_title(line)
                fout_title.write('%s\n' % title.strip())
                start_writing = True
                next_empty_line = fin.readline().strip()

                assert(len(next_empty_line) == 0)

                num_docs += 1

                continue

            if start_writing and line.strip():
                if line.startswith('</mydoc'):
                    assert(start_writing)
                    fout_body.write(u"\n")
                    start_writing = False
                else:
                    fout_body.write('%s ' % line.strip())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki_data', type=str, required=True)
    parser.add_argument('--text_body_output', type=str, required=True)
    parser.add_argument('--title_output', type=str, required=True)

    args = parser.parse_args()

    main(args.wiki_data, args.text_body_output, args.title_output)

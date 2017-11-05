#!/usr/bin/env python

import argparse
import io
import os
import re
import sys

import bs4
from bs4 import BeautifulSoup


def process_single_document(doc):
    assert isinstance(doc, bs4.element.Tag)

    selected_part = ''
    if doc.title:
        selected_part += doc.title.string + '.'
    if doc.body or doc.text:
        body_txt = doc.body.string if doc.body else doc.text
        try:
            splitted = body_txt.split()
            if splitted[-1] == 'Reuter' or \
                    splitted[-1] == 'REUTER':
                selected_part += ' ' + ' '.join(splitted[:-1])
            else:
                selected_part += ' ' + ' '.join(splitted)

        except UnicodeEncodeError:
            print(doc.body.string)
            sys.exit(0)

    replaced = re.sub(' +', ' ',
                      selected_part.replace('\n', ' '))

    return replaced


def main(data_path, trd_path, trl_path, tsd_path, tsl_path):
    train_set = {}
    test_set = {}
    train_label_set = set()
    test_label_set = set()
    train_doc_id, test_doc_id = 0, 0
    for filename in os.listdir(data_path):
        if not filename.endswith('.sgm'):
            continue

        with open('/'.join([data_path, filename])) as fin:
            reuter_docs = BeautifulSoup(fin.read(), 'html.parser').find_all('reuters')

            for doc in reuter_docs:
                labels = [label.string
                          for label in doc.topics.find_all('d')]
                if len(labels) == 0:
                    continue

                if doc['lewissplit'].lower() == 'train' and \
                        doc['topics'].lower() == 'yes':

                    train_set[train_doc_id] = (doc, labels)
                    train_label_set |= set(labels)
                    train_doc_id += 1

                if doc['lewissplit'].lower() == 'test' and \
                        doc['topics'].lower() == 'yes':

                    test_set[test_doc_id] = (doc, labels)
                    test_label_set |= set(labels)
                    test_doc_id += 1

    # delete labels
    common_labels = train_label_set.intersection(test_label_set)

    def filter_labels(labels, common_label_set):
        return [label for label in labels if label in common_label_set]

    def write_to_file(text_body_path, label_path, dataset, common_labels):
        with io.open(text_body_path, 'w', encoding='utf-8') as f_body, \
                io.open(label_path, 'w', encoding='utf-8') as f_label:

            for doc_id, (doc, labels) in dataset.items():
                output = process_single_document(doc)
                labels = filter_labels(labels, common_labels)

                if len(labels) > 0:
                    try:
                        f_body.write('%s\n' % output)
                    except TypeError as e:
                        print(doc)
                        print(output)
                        sys.exit(0)

                    f_label.write('%s\n' % ' '.join(labels))

    write_to_file(trd_path, trl_path, train_set, common_labels)
    write_to_file(tsd_path, tsl_path, test_set, common_labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--trd_path', type=str, required=True)
    parser.add_argument('--trl_path', type=str, required=True)
    parser.add_argument('--tsd_path', type=str, required=True)
    parser.add_argument('--tsl_path', type=str, required=True)

    args = parser.parse_args()

    main(args.data_path,
         args.trd_path, args.trl_path,
         args.tsd_path, args.tsl_path)

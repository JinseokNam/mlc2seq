from __future__ import print_function

from lxml import etree
import os
import argparse
from tempfile import NamedTemporaryFile
import urllib2
import gzip
import io


class RCV1Parser(object):
    def __init__(self, document_path):
        self.body_text = None
        self.labels = None
        self.itemid = None

        with io.open(document_path, encoding='iso-8859-1') as f:
            self.contents = etree.parse(f)

        self.handleID(self.contents.getroot())
        self.handleText(self.contents.findall("text")[0])
        self.handleCodes(self.contents.findall("metadata")[0])

    def handleID(self, newsitem):
        self.itemid = int(newsitem.attrib["itemid"])

    def handleText(self, text):
        self.body_text = ' '.join([line.text.strip() for line in text])

    def handleCodes(self, metadata):
        extracted_codes = []
        for codes in metadata.findall("codes"):
            if codes.attrib['class'] == "bip:topics:1.0":
                extracted_codes = [
                    code.attrib['code'] for code in codes.findall("code")]

        if len(extracted_codes) > 0:
            self.labels = ' '.join(extracted_codes)

    def getID(self):
        return self.itemid

    def getBodyText(self):
        return self.body_text

    def getLabels(self):
        return self.labels


def get_plain_text(src_dir):
    filenames = [os.path.join(src_dir, f) for f in os.listdir(src_dir)
                 if os.path.isfile(os.path.join(src_dir, f)) and
                 f.endswith('.xml')]

    body_label_pairs = {}
    for file_index, filename in enumerate(filenames):
        if (file_index + 1) % 1000 == 0:
            print('{} / {}\r'.format(file_index+1, len(filenames)), end='')

        pa = RCV1Parser(filename)

        assert pa.getID() not in body_label_pairs
        assert type(pa.getID()) == int

        if pa.getLabels() and pa.getBodyText():
            body_label_pairs[pa.getID()] = (pa.getBodyText(), pa.getLabels())

    print('')
    return body_label_pairs


def download_rcv1_token_files():

    base_url = ('http://jmlr.csail.mit.edu/papers/'
                'volume5/lewis04a/a12-token-files')

    def get_filename(dataset_split, number=None):
        if dataset_split == 'train':
            assert number is None
            return 'lyrl2004_tokens_{}.dat.gz'.format(dataset_split)
        elif dataset_split == 'test':
            assert number is not None
            return 'lyrl2004_tokens_{}_pt{}.dat.gz'.format(
                dataset_split, number)

    # download the train file
    def download_file(url):
        handle = urllib2.urlopen(url)

        return handle.read()

    def uncompress_file(zipped_contents):
        with NamedTemporaryFile(suffix='.gz', dir='/tmp') as f:
            f.write(zipped_contents)
            f.flush()

            with gzip.open(f.name) as fin:
                uncompressed_contents = fin.read()

        return uncompressed_contents

    train_contents = uncompress_file(download_file(
        '/'.join([base_url, get_filename('train')])))

    urls = ['/'.join([base_url, get_filename('test', subset_id)])
            for subset_id in range(4)]
    test_contents = '\n'.join(
        [uncompress_file(download_file(url)) for url in urls])

    def collect_doc_ids(data):
        ids = []
        for line in data.split('\n'):
            if line.startswith('.I'):
                t, i = line.strip().split()
                ids.append(int(i))

        return ids

    train_ids = {idx: 1 for idx in collect_doc_ids(train_contents)}
    test_ids = {idx: 1 for idx in collect_doc_ids(test_contents)}

    return (train_ids, test_ids)


def split_data(body_label_dict, train_ids, test_ids):
    train_set = []
    test_set = []

    for doc_id, (body, label) in body_label_dict.items():

        if doc_id in train_ids:
            train_set.append((body, label))
        elif doc_id in test_ids:
            test_set.append((body, label))

    return train_set, test_set


def store_processed_data(output_dir, train_set, test_set):
    trd_path = os.path.join(args.output_dir, 'trd.txt')
    trl_path = os.path.join(args.output_dir, 'trl.txt')
    tsd_path = os.path.join(args.output_dir, 'tsd.txt')
    tsl_path = os.path.join(args.output_dir, 'tsl.txt')

    with io.open(trd_path, 'w', encoding='utf-8') as f_trd,\
            io.open(trl_path, 'w', encoding='utf-8') as f_trl:
        for (body, labels) in train_set:
            f_trd.write(u'{}\n'.format(body.strip()))
            f_trl.write(u'{}\n'.format(labels.strip()))

    with io.open(tsd_path, 'w', encoding='utf-8') as f_tsd, \
            io.open(tsl_path, 'w', encoding='utf-8') as f_tsl:
        for (body, labels) in test_set:
            f_tsd.write(u'{}\n'.format(body.strip()))
            f_tsl.write(u'{}\n'.format(labels.strip()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--swap', type=bool, default=False)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Downloading RCV1 files to obtain train / test split")
    train_ids, test_ids = download_rcv1_token_files()

    print("Extracting plain text from xml files...")
    body_label_dict = get_plain_text(args.src_dir)
    print("Done")

    print("Split data according to the train / test split")
    train_set, test_set = split_data(body_label_dict, train_ids, test_ids)

    print("Store them into the output directory")
    if args.swap:
        store_processed_data(args.output_dir, test_set, train_set)
    else:
        store_processed_data(args.output_dir, train_set, test_set)

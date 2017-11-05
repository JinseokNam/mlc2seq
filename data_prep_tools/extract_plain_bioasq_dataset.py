#!/bin/usr/python

import io
import json
import re
import sys
import argparse


def processing(raw_data_filepath,
               trd_filepath, trl_filepath,
               tsd_filepath, tsl_filepath,
               split_year):

    line_count = 0

    with io.open(raw_data_filepath, encoding='utf-8') as f_raw_data, \
            io.open(trd_filepath, 'w', encoding='utf-8') as fout_trd, \
            io.open(trl_filepath, 'w', encoding='utf-8') as fout_trl, \
            io.open(tsd_filepath, 'w', encoding='utf-8') as fout_tsd, \
            io.open(tsl_filepath, 'w', encoding='utf-8') as fout_tsl:

        dup_pmid = dict()
        while True:
            line = f_raw_data.readline()
            if not line:
                # Reached the end of the input file
                break

            try:
                A = json.loads(line.strip()[:-1])
            except ValueError:
                continue

            line_count = line_count + 1
            if line_count % 100 is 0:
                sys.stdout.write(str(line_count) + '\r')
                sys.stdout.flush()

            pmid = A['pmid'].strip()
            if pmid in dup_pmid:
                continue
            else:
                dup_pmid[pmid] = 1

            doc = A['title'].strip() + ' ' + A['abstractText'].strip()
            mesh = A['meshMajor']
            year = int(re.match(r'\d+', A['year']).group())

            mesh_str = ' '.join([m.strip().replace(' ', '_') for m in mesh])
            if year < split_year:
                fout_trl.write('%s\n' % mesh_str)
                fout_trl.flush()
            elif year >= split_year:
                fout_tsl.write('%s\n' % mesh_str)
                fout_tsl.flush()

            # doc = re.sub(r"((\d+([\.\,]\d*)?)|\.\d+)", "0", doc)
            if year < split_year:
                fout_trd.write('%s\n' % doc)
                fout_trd.flush()
            elif year >= split_year:
                fout_tsd.write('%s\n' % doc)
                fout_tsd.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--traindata_output', type=str, required=True)
    parser.add_argument('--trainlabel_output', type=str, required=True)
    parser.add_argument('--testdata_output', type=str, required=True)
    parser.add_argument('--testlabel_output', type=str, required=True)
    parser.add_argument('--split_year', type=int, required=True)

    args = parser.parse_args()

    processing(args.input,
               args.traindata_output, args.trainlabel_output,
               args.testdata_output, args.testlabel_output, args.split_year)

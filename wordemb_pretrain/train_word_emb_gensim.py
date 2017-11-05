#!/usr/bin/env python

import logging
import argparse

from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()

    sentences = word2vec.LineSentence(args.corpus)
    model = word2vec.Word2Vec(sentences,
                              size=512,
                              sg=1,
                              workers=4,
                              min_count=30,
                              iter=5)

    model.save(args.output)     # pickle dump

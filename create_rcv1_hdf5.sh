#!/bin/bash

DATA_PATH=data/RCV1
WORD_EMB_PATH=output/gensim_pretrained_word_emb.model
OUTPUT_FILENAME=rcv1_dataset.hdf5

python create_hdf5_dataset.py \
    --trd ${DATA_PATH}/trd.delete.tok.max_300.lc.txt \
    --trl ${DATA_PATH}/trl.delete.txt \
    --vad ${DATA_PATH}/vad.delete.tok.max_300.lc.txt \
    --val ${DATA_PATH}/val.delete.txt \
    --tsd ${DATA_PATH}/tsd.delete.tok.max_300.lc.txt \
    --tsl ${DATA_PATH}/tsl.delete.txt \
    --label_vocab ${DATA_PATH}/label_vocab.txt \
    --word_emb ${WORD_EMB_PATH} \
    --output ${DATA_PATH}/${OUTPUT_FILENAME}

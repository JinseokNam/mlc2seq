#!/bin/bash

BASE_PATH=$PWD

DATASET_PATH=${BASE_PATH}/data/BioASQ
WORK_SCRATCH=/data/learned_models/mlc2seq
DATASET=bioasq
INPUT_CONFIG_PATH=${BASE_PATH}/exp_scripts/${DATASET}/config_${DATASET}_encdec.json

if [ $# -eq 0 ]
then
  EXP_ID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 10 | head -n 1)
else
  EXP_ID=$1
fi

OUTPUT_MODEL_DIR=${WORK_SCRATCH}/output
OUTPUT_PREFIX=${DATASET}_${EXP_ID}
CONFIG_PATH=${OUTPUT_PREFIX}.config.json

MODEL_PATH=${OUTPUT_MODEL_DIR}/${OUTPUT_PREFIX}.model.best.npz

jq -c 'setpath(["management", "reload_from"]; "'${MODEL_PATH}'")' ${INPUT_CONFIG_PATH} | python -m simplejson.tool > tmp.$$.json && \
mv tmp.$$.json ${CONFIG_PATH}

THEANO_FLAGS='device=gpu0,floatX=float32,scan.allow_gc=True' python mlc2seq_single.py \
--base_datapath ${DATASET_PATH} \
--base_outputpath ${OUTPUT_MODEL_DIR} \
--config ${CONFIG_PATH} \
--experiment_id ${OUTPUT_PREFIX}

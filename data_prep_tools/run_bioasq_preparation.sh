#!/bin/bash

function create_dir {
  if [[ ! -e $1 ]]; then
    mkdir $1
  elif [[ ! -d $1 ]]; then
    echo "$1 exists but not a directory" 1>&2
    exit
  fi
}

CWD=$PWD
BASE_PATH=${CWD}/data/BioASQ
OUTPUT_BASE=${CWD}/data/BioASQ/MLC2SEQ

create_dir $OUTPUT_BASE

RAW_JSON=${BASE_PATH}/allMeSH.json
trd_base=${OUTPUT_BASE}/trd
trl_base=${OUTPUT_BASE}/trl
vad_base=${OUTPUT_BASE}/vad
val_base=${OUTPUT_BASE}/val
tsd_base=${OUTPUT_BASE}/tsd
tsl_base=${OUTPUT_BASE}/tsl

TRD=$trd_base.txt
TRL=$trl_base.txt
VAD=$vad_base.txt
VAL=$val_base.txt
TSD=$tsd_base.txt
TSL=$tsl_base.txt
CHAR_VOCAB=${OUTPUT_BASE}/char_vocab.txt
WORD_VOCAB=${OUTPUT_BASE}/word_vocab.txt
LABEL_VOCAB=${OUTPUT_BASE}/label_vocab.txt
SPLIT_YEAR=2014
SRC=${CWD}/proc_util
MOSESDEC_PATH=${CWD}/data_prep_tools/mosesdecoder
MAX_SENT_LEN=300
WORD_MIN_CNT=1
LABEL_MIN_CNT=1
N_THREADS=4

python ${CWD}/data_prep_tools/extract_plain_bioasq_dataset.py \
--input ${RAW_JSON} \
--traindata_output ${TRD} \
--trainlabel_output ${TRL} \
--testdata_output ${TSD} \
--testlabel_output ${TSL} \
--split_year ${SPLIT_YEAR}

# shuffle data to break down connection
TRD_SHUF=$trd_base.shuf.txt
TRL_SHUF=$trl_base.shuf.txt
dd if=/dev/urandom of=rand count=$((128*1024)) status=none
shuf --random-source=rand ${TRD} > ${TRD_SHUF}
shuf --random-source=rand ${TRL} > ${TRL_SHUF}
mv ${TRD_SHUF} ${TRD}
mv ${TRL_SHUF} ${TRL}
rm rand

TMP_TRD=${OUTPUT_BASE}/tmp_trd.txt
TMP_TRL=${OUTPUT_BASE}/tmp_trl.txt
TMP_VAL=${OUTPUT_BASE}/tmp_val.txt
TMP_TSL=${OUTPUT_BASE}/tmp_tsl.txt

VAD_SIZE=50000

# split the original train data into the train and validation sets
PREV_TRD_NUM_LINE="$(wc -l ${TRD} | cut -d' ' -f 1)"
cat ${TRD} | head -n ${VAD_SIZE} > ${VAD}
cat ${TRL} | head -n ${VAD_SIZE} > ${VAL}
cat ${TRD} | tail -n $((PREV_TRD_NUM_LINE-VAD_SIZE)) > ${TMP_TRD}
cat ${TRL} | tail -n $((PREV_TRD_NUM_LINE-VAD_SIZE)) > ${TMP_TRL}
mv ${TMP_TRD} ${TRD}
mv ${TMP_TRL} ${TRL}

# create vocabularies
python ${SRC}/create_character_vocab.py --input ${TRD} --output ${CHAR_VOCAB}
awk -v OFS="\t" -v LABEL_MIN_CNT=$LABEL_MIN_CNT '{ for(i=1; i<=NF; i++) w[$i]++ } END {for(i in w) { if(w[i] >= LABEL_MIN_CNT) {print w[i], i}} }' ${TRL} | sort -k 1nr > ${LABEL_VOCAB}

# sort labels of each instance by label frequency
python ${SRC}/sort_labels.py --input ${TRL} --label_vocab ${LABEL_VOCAB} --output ${TMP_TRL}
python ${SRC}/sort_labels.py --input ${VAL} --label_vocab ${LABEL_VOCAB} --output ${TMP_VAL}
python ${SRC}/sort_labels.py --input ${TSL} --label_vocab ${LABEL_VOCAB} --output ${TMP_TSL}
mv ${TMP_TRL} ${TRL}
mv ${TMP_VAL} ${VAL}
mv ${TMP_TSL} ${TSL}

# delete instances which have empty label set
python ${SRC}/delete_instances.py --data ${TRD} --label ${TRL} --out_data $trd_base.delete.txt --out_label $trl_base.delete.txt --label_vocab ${LABEL_VOCAB}
python ${SRC}/delete_instances.py --data ${VAD} --label ${VAL} --out_data $vad_base.delete.txt --out_label $val_base.delete.txt --label_vocab ${LABEL_VOCAB}
python ${SRC}/delete_instances.py --data ${TSD} --label ${TSL} --out_data $tsd_base.delete.txt --out_label $tsl_base.delete.txt --label_vocab ${LABEL_VOCAB}
 
trd_base=$trd_base.delete
trl_base=$trl_base.delete
vad_base=$vad_base.delete
val_base=$val_base.delete
tsd_base=$tsd_base.delete
tsl_base=$tsl_base.delete

# tokenize
cat $trd_base.txt | ${MOSESDEC_PATH}/scripts/tokenizer/normalize-punctuation.perl -l en | \
    ${MOSESDEC_PATH}/scripts/tokenizer/tokenizer.perl -a -l en -threads ${N_THREADS} | \
    ${MOSESDEC_PATH}/scripts/generic/ph_numbers.perl -c > $trd_base.tok.txt

cat $vad_base.txt | ${MOSESDEC_PATH}/scripts/tokenizer/normalize-punctuation.perl -l en | \
    ${MOSESDEC_PATH}/scripts/tokenizer/tokenizer.perl -a -l en -threads ${N_THREADS} | \
    ${MOSESDEC_PATH}/scripts/generic/ph_numbers.perl -c > $vad_base.tok.txt

cat $tsd_base.txt | ${MOSESDEC_PATH}/scripts/tokenizer/normalize-punctuation.perl -l en | \
    ${MOSESDEC_PATH}/scripts/tokenizer/tokenizer.perl -a -l en -threads ${N_THREADS} | \
    ${MOSESDEC_PATH}/scripts/generic/ph_numbers.perl -c > $tsd_base.tok.txt

trd_base=$trd_base.tok
trl_base=$trl_base.tok
vad_base=$vad_base.tok
val_base=$val_base.tok
tsd_base=$tsd_base.tok
tsl_base=$tsl_base.tok

# limit the length of each document
python ${SRC}/cut_long_sentences.py --input $trd_base.txt --output $trd_base.max_${MAX_SENT_LEN}.txt --max ${MAX_SENT_LEN} --level word
python ${SRC}/cut_long_sentences.py --input $vad_base.txt --output $vad_base.max_${MAX_SENT_LEN}.txt --max ${MAX_SENT_LEN} --level word
python ${SRC}/cut_long_sentences.py --input $tsd_base.txt --output $tsd_base.max_${MAX_SENT_LEN}.txt --max ${MAX_SENT_LEN} --level word

trd_base=$trd_base.max_${MAX_SENT_LEN}
trl_base=$trl_base.max_${MAX_SENT_LEN}
vad_base=$vad_base.max_${MAX_SENT_LEN}
val_base=$val_base.max_${MAX_SENT_LEN}
tsd_base=$tsd_base.max_${MAX_SENT_LEN}
tsl_base=$tsl_base.max_${MAX_SENT_LEN}

# lowercase
${MOSESDEC_PATH}/scripts/tokenizer/lowercase.perl -l en < $trd_base.txt > $trd_base.lc.txt
${MOSESDEC_PATH}/scripts/tokenizer/lowercase.perl -l en < $vad_base.txt > $vad_base.lc.txt
${MOSESDEC_PATH}/scripts/tokenizer/lowercase.perl -l en < $tsd_base.txt > $tsd_base.lc.txt

trd_base=$trd_base.lc
trl_base=$trl_base.lc
vad_base=$vad_base.lc
val_base=$val_base.lc
tsd_base=$tsd_base.lc
tsl_base=$tsl_base.lc

# create the word vocabulary
awk -v OFS="\t" -v WORD_MIN_CNT=$WORD_MIN_CNT '{ for(i=1; i<=NF; i++) w[$i]++ } END {for(i in w) { if(w[i] >= WORD_MIN_CNT) {print w[i], i}} }' $trd_base.txt | sort -k 1nr > ${WORD_VOCAB}

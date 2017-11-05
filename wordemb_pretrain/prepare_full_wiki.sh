#!/bin/bash

WIKI_DUMP=$1    # path to the wikipedia dump
CLEANED_WIKI=$2 # output path

TMP_DIR=$(mktemp -d)
TMP_MERGE_FILE=$(mktemp)

# extract plain text
cat ${WIKI_DUMP} | python WikiExtractor.py -o ${TMP_DIR}

find ${TMP_DIR} -type f -name 'wiki_*' -print0 | while IFS= read -r -d '' file
do 
    cat "$file"
done > ${TMP_MERGE_FILE}

TMP_BODY=$(mktemp)
TMP_TITLE=$(mktemp)

python extract_title_and_body.py --wiki_data ${TMP_MERGE_FILE} \
    --text_body_output ${TMP_BODY} \
    --title_output ${TMP_TITLE}

# tokenize & lowercase
cat ${TMP_BODY} | ../data_prep_tools/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl | \
    ../data_prep_tools/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | \
    ../data_prep_tools/mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l en | \
    ../data_prep_tools/mosesdecoder/scripts/generic/ph_numbers.perl -c | \
    ../data_prep_tools/mosesdecoder/scripts/tokenizer/lowercase.perl -l en > ${CLEANED_WIKI}

# clean up temporary files
rm ${TMP_TITLE}
rm ${TMP_BODY}
rm ${TMP_MERGE_FILE}
rm -rf ${TMP_DIR}

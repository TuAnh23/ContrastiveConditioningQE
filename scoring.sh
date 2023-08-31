#!/bin/bash
# Crash if variable used without being set
set -eu

# Setting environment
source /home/tdinh/.bashrc
conda activate KIT_start
which python

# Where to store cache
export TORCH_HOME=/project/OML/tdinh/.cache/torch
export HF_HOME=/project/OML/tdinh/.cache/huggingface

# Where to store experiment outputs
export ARTIFACTS_ROOT_PATH=/export/data1/tdinh

export CUDA_VISIBLE_DEVICES=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # make sure the GPU order is correct

python -u process_wmt_qe_data.py --data_split "dev"


SRC_LANG="en"
TGT_LANG="de"
BPE_ROOT="/home/tdinh/miniconda3/envs/KIT_start/lib/python3.9/site-packages/subword_nmt"
MODEL_DIR="/project/OML/tdinh/KIT_start/models/${SRC_LANG}-${TGT_LANG}"
#MODEL_DIR="/project/OML/tdinh/KIT_start/models/wmt16.en-de.joined-dict.transformer"

BPE="${MODEL_DIR}/bpecodes"

TMP="/export/data1/tdinh/tmp"
batch_size=400
output_dir="output/dev_knownMT/similar_not_replace"  # dev_knownMT dev_otherMT
INPUT="en-de-dev/input"

mkdir -p ${output_dir}

# Tokenize data
for LANG in $SRC_LANG $TGT_LANG; do
#        sacremoses -l $LANG tokenize -a < $INPUT.$LANG > $TMP/preprocessed.tok.$LANG
  perl ../mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 80 -a -l $LANG < $INPUT.$LANG > $TMP/preprocessed.tok.$LANG
  python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
done
# Apply bpe
for LANG in $SRC_LANG $TGT_LANG; do
  python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
done
# Binarize the data for faster translation
fairseq-preprocess --srcdict $MODEL_DIR/dict.$SRC_LANG.txt --tgtdict $MODEL_DIR/dict.$TGT_LANG.txt --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} --testpref $TMP/preprocessed.tok.bpe --destdir $TMP/bin --workers 4
# Translate
fairseq-generate $TMP/bin --path ${MODEL_DIR}/${SRC_LANG}-${TGT_LANG}.pt --beam 5 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 --batch-size ${batch_size} --score-reference > ${output_dir}/fairseq.out
grep ^H ${output_dir}/fairseq.out | cut -d- -f2- | sort -n | cut -f2 > ${output_dir}/sent_log_prob.out

python analyse_prob_diff.py --output_dir ${output_dir}

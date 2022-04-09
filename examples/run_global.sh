#!/bin/bash

export KMER=3
export dataPATH=data_process_template/3_mer
export MODEL_PATH=pre_model/global-model

for DATA in $(ls $dataPATH)
do
  export DATA_PATH=$dataPATH/$DATA
  export OUTPUT_PATH=result/global/$DATA
  python TFBert.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_predict \
    --data_dir $DATA_PATH  \
    --max_seq_length 128 \
    --per_gpu_pred_batch_size=128   \
    --output_dir $MODEL_PATH \
    --predict_dir $OUTPUT_PATH \
    --n_process 8
done

#!/bin/bash

export KMER=3
export MODEL_PATH=pre_model/pretrain/checkpoint-20000
export dataPATH=data_process_template/3_mer

for DATA in $(ls $dataPATH)
do
  export DATA_PATH=$dataPATH/$DATA
  export OUTPUT_PATH=result/individual/$DATA
  python TFBert.py \
    --model_type dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --task_name dnaprom \
    --do_train \
    --do_eval \
    --data_dir $DATA_PATH \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=128   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5 \
    --max_steps  4000 \
    --logging_steps 100 \
    --save_steps 100 \
    --early_stop  5    \
    --output_dir $OUTPUT_PATH \
    --evaluate_during_training \
    --warmup_percent 0.1 \
    --hidden_dropout_prob 0.1 \
    --overwrite_output \
    --weight_decay 0.01 \
    --n_process 8
done

#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/3_ImageNet_merged'
else
    MODELPATH=$2
fi

if [ ! -n "$3" ] ;then
    MODELNAME='llava-v1.5-7b-lora'
else
    MODELNAME=$3
fi

RESULT_DIR="./results/CoIN/"${MODELNAME}"/3_ImageNet"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CoIN.model_others \
        --model-path $MODELPATH \
        --question-file ./playground/data/CoIN/annotations/ImageNet/test.json \
        --image-folder ./playground/data/CoIN/ \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.CoIN.eval_imagenet \
    --test-file ./playground/data/CoIN/annotations/ImageNet/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE

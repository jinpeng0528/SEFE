# #!/bin/bash

if [ "$1" == "1" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/1_ScienceQA.sh 1_ScienceQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/1_ScienceQA_merged llava-v1.5-7b-lora
elif [ "$1" == "2" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/1_ScienceQA.sh 2_TextVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/2_TextVQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/2_TextVQA.sh 2_TextVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/2_TextVQA_merged llava-v1.5-7b-lora
elif [ "$1" == "3" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/1_ScienceQA.sh 3_ImageNet ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/3_ImageNet_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/2_TextVQA.sh 3_ImageNet ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/3_ImageNet_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/3_ImageNet.sh 3_ImageNet ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/3_ImageNet_merged llava-v1.5-7b-lora
elif [ "$1" == "4" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/1_ScienceQA.sh 4_GQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/4_GQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/2_TextVQA.sh 4_GQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/4_GQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/3_ImageNet.sh 4_GQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/4_GQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/4_GQA.sh 4_GQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/4_GQA_merged llava-v1.5-7b-lora
elif [ "$1" == "5" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/1_ScienceQA.sh 5_VizWiz ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/5_VizWiz_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/2_TextVQA.sh 5_VizWiz ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/5_VizWiz_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/3_ImageNet.sh 5_VizWiz ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/5_VizWiz_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/4_GQA.sh 5_VizWiz ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/5_VizWiz_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/5_VizWiz.sh 5_VizWiz ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/5_VizWiz_merged llava-v1.5-7b-lora
elif [ "$1" == "6" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/1_ScienceQA.sh 6_Grounding ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/6_Grounding_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/2_TextVQA.sh 6_Grounding ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/6_Grounding_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/3_ImageNet.sh 6_Grounding ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/6_Grounding_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/4_GQA.sh 6_Grounding ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/6_Grounding_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/5_VizWiz.sh 6_Grounding ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/6_Grounding_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/6_Grounding.sh 6_Grounding ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/6_Grounding_merged llava-v1.5-7b-lora
elif [ "$1" == "7" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/1_ScienceQA.sh 7_VQAv2 ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/7_VQAv2_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/2_TextVQA.sh 7_VQAv2 ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/7_VQAv2_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/3_ImageNet.sh 7_VQAv2 ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/7_VQAv2_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/4_GQA.sh 7_VQAv2 ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/7_VQAv2_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/5_VizWiz.sh 7_VQAv2 ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/7_VQAv2_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/6_Grounding.sh 7_VQAv2 ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/7_VQAv2_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/7_VQAv2.sh 7_VQAv2 ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/7_VQAv2_merged llava-v1.5-7b-lora
else
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/1_ScienceQA.sh 8_OCRVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/8_OCRVQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/2_TextVQA.sh 8_OCRVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/8_OCRVQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/3_ImageNet.sh 8_OCRVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/8_OCRVQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/4_GQA.sh 8_OCRVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/8_OCRVQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/5_VizWiz.sh 8_OCRVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/8_OCRVQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/6_Grounding.sh 8_OCRVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/8_OCRVQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/7_VQAv2.sh 8_OCRVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/8_OCRVQA_merged llava-v1.5-7b-lora
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/Eval/8_OCRVQA.sh 8_OCRVQA ./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/8_OCRVQA_merged llava-v1.5-7b-lora
fi

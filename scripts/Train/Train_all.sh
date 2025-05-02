#!/bin/bash

bash ./scripts/Train/1_ScienceQA.sh
bash ./scripts/Eval/Eval_all.sh 1

bash ./scripts/Train/2_TextVQA.sh
bash ./scripts/Eval/Eval_all.sh 2

bash ./scripts/Train/3_ImageNet.sh
bash ./scripts/Eval/Eval_all.sh 3

bash ./scripts/Train/4_GQA.sh
bash ./scripts/Eval/Eval_all.sh 4

bash ./scripts/Train/5_VizWiz.sh
bash ./scripts/Eval/Eval_all.sh 5

bash ./scripts/Train/6_Grounding.sh
bash ./scripts/Eval/Eval_all.sh 6

bash ./scripts/Train/7_VQAv2.sh
bash ./scripts/Eval/Eval_all.sh 7

bash ./scripts/Train/8_OCRVQA.sh
bash ./scripts/Eval/Eval_all.sh 8

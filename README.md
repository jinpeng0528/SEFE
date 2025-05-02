# [ICML 2025] SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning

This is an official implementation of the paper “SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning”, accepted by ICML 2025.

## Installation
Our environment is set up with CUDA 12.1. To ensure a smooth installation, it is recommended to also use CUDA 12.1.
```Shell
conda create -n sefe python=3.10 -y
conda activate sefe
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn==2.6.3 --no-build-isolation
```

## Data Organization and Structure

To obtain the original images and annotation data for CoIN, please refer to the [official CoIN repository](https://github.com/zackschen/CoIN). We organize the downloaded files in the following directory structure:

```
./playground/data/CoIN
├── ScienceQA
│   └── [Original Data of ScienceQA]
├── TextVQA
│   └── [Original Data of TextVQA]
├── ImageNet
│   └── [Original Data of ImageNet]
├── GQA
│   └── [Original Data of GQA]
├── VizWiz
│   └── [Original Data of VizWiz]
├── COCO
│   └── [Original Data of COCO]
├── OCRVQA
│   └── [Original Data of OCRVQA]
└── annotations
    ├── ScienceQA
    │   ├── train.json
    │   └── test.json
    ├── TextVQA
    │   ├── train.json
    │   └── test.json
    ├── ImageNet
    │   ├── train.json
    │   └── test.json
    ├── GQA
    │   ├── train.json
    │   └── test.json
    ├── VizWiz
    │   ├── train.json
    │   └── test.json
    ├── Grounding
    │   ├── train.json
    │   └── test.json
    ├── VQAv2
    │   ├── train.json
    │   └── test.json
    └── OCRVQA
        ├── train.json
        └── test.json
```
**Notes:**
- **Original Data Directories**: The placeholders `[Original Data of XXX]` represent the datasets (primarily images) downloaded directly from benchmarks such as ScienceQA and TextVQA. These are maintained in their default directory structures.
- **COCO Folder**: Although the CoIN benchmark does not directly include COCO, the Grounding and VQAv2 tasks utilize images from the COCO dataset. Therefore, a `COCO` folder is included.
- **Annotations**: The `train.json` and `test.json` files within the `annotations` directory contain annotations provided by CoIN or modified by our ASD. For consistency, all test sets originally named `val.json` in the CoIN repository have been renamed to `test.json`.

## CoIN-ASD
The `CoIN-ASD/prompts` directory contains all prompts used to create the CoIN-ASD benchmark, which can be utilized to reproduce the data transformation process in CoIN-ASD. Additionally, the `CoIN-ASD/examples` directory provides data samples from various tasks within CoIN-ASD, including both the original data (prior to transformation) and the transformed data, allowing for easy comparison and analysis.

## Pre-trained Weights
Before starting the training process, you need to download three pre-trained models:
- [Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
- [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)
- [LLaVA Projector](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5)

We organize the downloaded models in the following directory structure:
```
./pretrained_weights
├── vicuna-7b-v1.5
├── clip-vit-large-patch14-336
└── llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5
```

## Training and Evaluation
Once the data transformation is complete and structured correctly, you can initiate training by running `./scripts/Train/Train_all.sh`. This script will automatically invoke `./scripts/Eval/Eval_all.sh` after training each task to evaluate all learned tasks. For further details, please refer to the corresponding files.

## Acknowledgement
This repository is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) and [CoIN](https://github.com/zackschen/CoIN) projects. We would like to express our gratitude to the authors for their contributions to the community.

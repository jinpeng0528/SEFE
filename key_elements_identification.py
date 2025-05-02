import os
import sys
import json
import tqdm
import torch
import safetensors.torch as st


step = int(sys.argv[1])

model_dir = "./checkpoints/LLaVA/CoIN/llava-v1.5-7b-lora/"
model_path_list = [
    os.path.join(model_dir, "1_ScienceQA"),
    os.path.join(model_dir, "2_TextVQA"),
    os.path.join(model_dir, "3_ImageNet"),
    os.path.join(model_dir, "4_GQA"),
    os.path.join(model_dir, "5_VizWiz"),
    os.path.join(model_dir, "6_Grounding"),
    os.path.join(model_dir, "7_VQAv2"),
    os.path.join(model_dir, "8_OCRVQA"),
]

if step > 0:
    legacy_topK_path = model_path_list[step - 1] + "_topP.pth"
    legacy_topK = torch.load(legacy_topK_path)

model_path_list = model_path_list[step: step + 1]

for model_path in model_path_list:
    lora_params = st.load_file(os.path.join(model_path, 'adapter_model.safetensors'))
    # lora_params = {k: v.cuda() for k, v in lora_params.items()}

    ori_param_set = set()
    for k in lora_params.keys():
        ori_param = k.replace(".lora_A", "").replace(".lora_B", "")
        ori_param_set.add(ori_param)
    lora_pairs = {k: [] for k in ori_param_set}
    for k in lora_params.keys():
        lora_pairs[k.replace(".lora_A", "").replace(".lora_B", "")].append(k)
        if len(lora_pairs[k.replace(".lora_A", "").replace(".lora_B", "")]) == 2:
            lora_pairs[k.replace(".lora_A", "").replace(".lora_B", "")].sort()

    stats = {}
    lora_pairs = dict(sorted(lora_pairs.items()))
    for i, (name, pair) in tqdm.tqdm(enumerate(lora_pairs.items())):
        # if i == 2:
        #     break
        lora_A = lora_params[pair[0]]
        lora_B = lora_params[pair[1]]
        lora_res = (lora_B @ lora_A).to(torch.float32)

        dims = lora_res.shape
        param_importance = torch.abs(lora_res)

        k_1 = int(0.02 * param_importance.shape[0] * param_importance.shape[1])
        _, flat_indices = torch.topk(param_importance.view(-1), k_1)
        indices_2d = torch.stack([
            torch.div(flat_indices, param_importance.size(1), rounding_mode='floor'),
            flat_indices % param_importance.size(1)
        ]).T

        stats[name] = {
            "top2": indices_2d,
        }

        if step > 0:
            stats[name]["top2"] = torch.cat((stats[name]["top2"], legacy_topK[name]["top2"]), dim=0)

        # indices = stats[name]["top2"].T
        # stats[name]["top2"] = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1], dtype=torch.bool), dims)

    torch.save(stats, model_dir + os.path.basename(model_path).strip("_prev") + "_topP.pth")

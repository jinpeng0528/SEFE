def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")

    assert model_paths[-2][-4:] == 'lora' or model_paths[-2][-3:] == 'sft'
    if model_paths[-2][-4:] == 'lora':
        return model_paths[-2]
    else:
        return '-'.join(model_paths[-2].split('-')[:3])

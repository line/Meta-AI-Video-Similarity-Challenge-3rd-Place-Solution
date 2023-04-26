import timm


def timm_model_config(model_name):
    if model_name == "hf-hub:timm/convnext_base.clip_laion2b_augreg_ft_in1k":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        input_size = (256, 256)
    elif model_name == "hf-hub:timm/convnext_base.clip_laiona_augreg_ft_in1k_384":
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        input_size = (384, 384)
    else:
        mean = timm.get_pretrained_cfg_value(model_name, "mean")
        std = timm.get_pretrained_cfg_value(model_name, "std")
        input_size = timm.get_pretrained_cfg_value(model_name, "input_size")[1:]

    return mean, std, input_size

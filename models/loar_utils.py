# common_utils.py (혹은 models/lora_utils.py)
import torch
def apply_lora_freeze(model: torch.nn.Module, freeze_gpt: bool=True):
    """
    GPT2Model (LoRA 포함)에서 LoRA 파라미터만 학습하도록 설정.
    freeze_gpt=True면 GPT-2 본체 weight은 모두 requires_grad=False,
    LoRA 모듈만 True로 바꿔줌.
    """
    for name, p in model.named_parameters():
        # 이름에 "lora"가 포함된 파라미터만 학습(True)
        if freeze_gpt and ("lora" in name):
            p.requires_grad = True
        elif freeze_gpt:
            p.requires_grad = False
        else:
            # freeze_gpt=False인 경우엔 전부 학습
            p.requires_grad = True
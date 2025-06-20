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


def param_by_option(config, model):
    assert config.fine_tune_mode in ["last-linear-layer", "full-model", "LoRA"]
    for param in model.parameters():
      if config.fine_tune_mode == 'last-linear-layer':
        param.requires_grad = False
      elif config.fine_tune_mode == 'full-model':
        param.requires_grad = True
      elif config.fine_tune_mode == 'LoRA':
        apply_lora_freeze(model, freeze_gpt=True)


def print_param_info(model):
      # ─── 여기에 검증용 코드 추가 ───
    print("--- 학습 가능한(Requires_grad=True) 파라미터 목록 ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50} |  shape: {tuple(param.shape)}")
    print("총 학습 가능 파라미터 개수:", 
          sum(param.numel() for param in model.parameters() if param.requires_grad))
    print("총 파라미터 개수:", 
          sum(param.numel() for param in model.parameters()))
    print("-----------------------------------------------\n")
    # ──────────────────────────────────────────────────
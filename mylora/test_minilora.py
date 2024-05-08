import torch.nn as nn
from minilora import MiniLoraConfig,get_mini_peft_model,LoRALinear
from dataclasses import dataclass
from peft import TaskType
import torch


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x





def test_get_peft_model_with_target_modules():
    model = SimpleModel()
    peft_config = MiniLoraConfig(
        rank=2,
        target_modules=["fc1"]
    )

    new_model = get_mini_peft_model(model, peft_config)

    # 检查 fc1 层是否被替换为 LoRALinear 层
    print(f"fc1是否被替换为loraLinear，答案：{isinstance(new_model.fc1, LoRALinear)}")

    print(f"检查 fc2 层是否保持不变，答案：{isinstance(new_model.fc2, nn.Linear)}")
    print("---------旧的模型----------")
    print(model)
    print("---------新的模型--------")
    print(new_model)





def test_get_peft_model_with_all_linear_layers():
    model = SimpleModel()
    peft_config = MiniLoraConfig(
        rank=1,
        lora_alpha=2,
        target_modules='all_linear'
    )

    new_model = get_mini_peft_model(model, peft_config)

    # 检查 fc1 和 fc2 层都被替换为 LoRALinear 层
    assert isinstance(new_model.fc1, LoRALinear)
    assert isinstance(new_model.fc2, LoRALinear)
    print("----------old model-----------")
    print(model)
    print("----------new model----------------")
    print(new_model)
def test_merge():
    model = SimpleModel()
    peft_config = MiniLoraConfig(
        rank=2,
        lora_alpha=2,
        target_modules='all_linear'
    )

    lora_model = get_mini_peft_model(model, peft_config)
    from minilora import merge_lora_back_to_pertrain_model
    print("---------------lora model-------------")
    print(lora_model)
    merged_model = merge_lora_back_to_pertrain_model(lora_model)
    print("-------------merged model----------")
    print(merged_model)

    merged_model.fc1.weight
    print(f"merged_model.fc1.weight : \n {merged_model.fc1.weight[0,:]}")
    merged_weight = lora_model.fc1.weight + lora_model.fc1.B.weight @ lora_model.fc1.A.weight
    print(f"merged_weight : \n {merged_weight[0,:]}")
    assert torch.allclose(merged_model.fc1.weight,merged_weight)

from myquantization.myquantize import *
from mylora.minilora import *
def test_qlora():
    from model import GPT,GPTConfig
    import copy

    config = GPTConfig(n_layer=1,n_embd=12,n_head=1)
    model = GPT(config)
    print("-------vanilla model---------")
    print(model)
    for name,param in model.named_parameters():
        print(f"{param.data.dtype}## {param.requires_grad} ## {name}  ")


    peft_config = MiniLoraConfig(
        rank=2,
        lora_alpha=2,
        target_modules=['q_proj', 'v_proj'],
        use_qlora=False
    )

    lora_model = get_mini_peft_model(model, peft_config)
    print("-------vanilla lora model---------")
    print(lora_model)
    for name, param in lora_model.named_parameters():
        print(f"{param.data.dtype}## {param.requires_grad} ## {name}  ")





    quantize_config:QuantizationConfig = QuantizationConfig(precision=4,exclude_modules=['lm_head'])
    new_model = copy.deepcopy(model)
    new_model = set_parameters_requires_grad(new_model)
    quantizer = QuantizationHandler(min_val=-1,max_val=10,n_bits=8)
    quantized_model, has_been_replaced = replace_linear_with_linearkbit(new_model,config=quantize_config,quantizer=quantizer)
    print("------------------quantized model -----------------")
    print(quantized_model)
    for name,param in quantized_model.named_parameters():
        print(f"{param.data.dtype}## {param.requires_grad} ## {name}  ")


    peft_config = MiniLoraConfig(
        rank=2,
        lora_alpha=2,
        # target_modules=['q_proj','v_proj','k_proj','ffn_up','ffn_down'],
        target_modules=['q_proj', 'v_proj'],
        use_qlora=True
    )

    qlora_model = get_mini_peft_model(quantized_model, peft_config)

    print("---------------qlora model-------------")
    print(qlora_model)
    for name,param in qlora_model.named_parameters():
        print(f"{param.data.dtype}## {param.requires_grad} ## {name}  ")

    merged_model = merge_lora_back_to_pertrain_model(qlora_model,quantizer)
    print("-------------merged model----------")
    print(merged_model)
    for name,param in merged_model.named_parameters():
        print(f"{param.data.dtype}## {param.requires_grad} ## {name}  ")

    models = dict(
        vanilla_model = model,
        lora_model = lora_model,
        quantized_model = quantized_model,
        qlora_model = qlora_model,
        merged_model = merged_model
    )
    from utils.basic_utils import calculate_model_memory,calculate_model_training_memory
    for model_name,model in models.items():
        print(f"{model_name} member {calculate_model_training_memory(model)[2]} Mb")



if __name__ == '__main__':
    # test_get_peft_model_with_target_modules()
    # test_get_peft_model_with_all_linear_layers()
    # test_merge()
    test_qlora()
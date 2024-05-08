import copy
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch

@dataclass
class MiniLoraConfig:
    rank: int = 32
    lora_dropout: float = 0.1
    lora_alpha : int = 64
    target_modules: list = 'all_linear' #也可以具体指定['fc1','q_proj']
    use_qlora : bool = False
    quantize_modules : list = None
    """
    #要求 quantize_modules和target_module要互斥，qlora层已经对w做了int量化
    非qlora部分直接做量化，quantize_modules是针对非qlora部分做的量化，所以可以
    和target_modules这部分互斥
    """

class LoRALinear(nn.Linear):
    """
    继承自 nn.Linear 的 LoRALinear 层。
    """

    def __init__(self, in_features, out_features, config:MiniLoraConfig, bias=True):
        super().__init__(in_features, out_features, bias=bias)

        # 将原始权重和偏置设置为不可训练
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.rank = config.rank
        assert self.rank>0,'rank must be an positive integar!!'
        self.lora_alpha = config.lora_alpha
        self.scale = self.lora_alpha/self.rank

        self.A = nn.Linear(in_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, out_features, bias=False)
        self.dropout = nn.Dropout(p=config.lora_dropout)
        self.A.weight.data.normal_(std=0.02)
        self.B.weight.data.zero_()

    def forward(self, input):
        linear_output = F.linear(input, self.weight, self.bias)
        lora_output = self.B(self.A(self.dropout(input)))
        return linear_output + self.scale * lora_output

    def merge_lora_weight_and_linear_weight(self):
        new_weight = self.weight + self.B.weight @ self.A.weight * self.scale
        return new_weight

from myquantization.myquantize import QuantizationHandler
class QLoRALinear(LoRALinear):
    def __init__(self, in_features, out_features, config:MiniLoraConfig, bias=True, quantizer:QuantizationHandler=None):
        super().__init__(in_features, out_features, config,bias=bias)
        self.quantizer = quantizer
        self.weight.data = self.quantizer.quantize(self.weight.data)
        # self.weight.data = self.weight.data.to(torch.int8)
        self.use_bias = bias
        if self.use_bias:
            self.bias.data = self.bias.data.to(torch.int8)

    def forward(self, input):
        if input.dtype is not torch.int8:
            quantized_input = self.quantizer.quantize(input)
        else:
            quantized_input = input

        linear_output = F.linear(quantized_input,self.weight,self.bias)
        linear_output = self.quantizer.dequantize(linear_output)

        lora_output = self.B(self.A(self.dropout(input)))
        output = linear_output + self.scale * lora_output
        return output
    def merge_lora_weight_and_linear_weight(self):
        dequant_weight = self.quantizer.dequantize(self.weight)
        new_weight = dequant_weight + self.B.weight @ self.A.weight * self.scale
        return new_weight

class LinearKbitNew(LoRALinear):
    """
    这个类是我为了排查
    """
    def __init__(self, in_features, out_features, config:MiniLoraConfig, bias=True, quantizer:QuantizationHandler=None):
        super().__init__(in_features, out_features, config,bias=bias)
        self.quantizer = quantizer
        self.weight.data = self.quantizer.quantize(self.weight.data)
        # self.weight.data = self.weight.data.to(torch.int8)
        self.use_bias = bias
        if self.use_bias:
            self.bias.data = self.bias.data.to(torch.int8)

    def forward(self, input):
        if input.dtype is not torch.int8:
            quantized_input = self.quantizer.quantize(input)
        else:
            quantized_input = input

        linear_output = F.linear(quantized_input,self.weight,self.bias)
        linear_output = self.quantizer.dequantize(linear_output)
        return linear_output
        """
        这里有一个非常神奇的现象，针对w_proj，如果增加下面的AB参数就可以正常梯度下降，
        如果仅仅只有linear的前向运算，梯度就无法正常传递
        """
        # lora_output = self.B(self.A(self.dropout(input)))
        # output = linear_output + self.scale * lora_output
        # return output
    def merge_lora_weight_and_linear_weight(self):
        dequant_weight = self.quantizer.dequantize(self.weight)
        new_weight = dequant_weight + self.B.weight @ self.A.weight * self.scale
        return new_weight


from typing import Callable
def update_model(model,replace_func:Callable):
    if model is None:
        return None
    for name, module in model.named_children():
        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            if isinstance(module, nn.ModuleList):
                for idx, sub_module in enumerate(module):
                    module[idx] = update_model(sub_module,replace_func)
            elif isinstance(module, nn.ModuleDict):
                for key, sub_module in module.items():
                    module[key] = update_model(sub_module,replace_func)
        elif isinstance(module, nn.Module):
            if hasattr(module, "named_children"):
                update_model(module,replace_func)
            setattr(model, name, replace_func(module, name))
    return model
def set_parameters_requires_grad(module,trainable=False):
    for name,param in module.named_parameters():
        param.requires_grad = trainable
    return module

def get_mini_peft_model(model, peft_config:MiniLoraConfig):
    """
    将原始模型中指定的 nn.Linear 层替换为 LoRALinear 层。

    Args:
        model (torch.nn.Module): 原始模型。
        peft_config (LoraConfig): LoRA 配置,包含 r、lora_dropout 和 target_modules 等参数。

    Returns:
        torch.nn.Module: 新的模型,其中指定的 nn.Linear 层被 LoRALinear 层替换。
    """
    new_model = copy.deepcopy(model)

    # 先把所有参数grad都关闭
    new_model = set_parameters_requires_grad(new_model)
    if peft_config.use_qlora:
        quantizer:QuantizationHandler = QuantizationHandler(min_val=-1,max_val=1,n_bits=8)

    # 只有loralinear中的AB参数grad为True
    def replace_linear(module,module_name):

        if isinstance(module, nn.Linear) and (
                any(module_name in target_module_name for target_module_name in peft_config.target_modules) or
                peft_config.target_modules == 'all_linear'):
            use_bias = module.bias is not None
            if peft_config.use_qlora:
                new_linear = QLoRALinear(module.in_features, module.out_features, peft_config, bias=use_bias, quantizer=quantizer)
                # 如果还没量化，就量化再赋值
                if module.weight.dtype is not torch.int8:
                    quantized_weight = quantizer.quantize(module.weight.data)
                    new_linear.weight.data.copy_(quantized_weight)
                    if module.bias is not None:
                        new_linear.bias.data.copy_(quantizer.quantize(module.bias.data))
                else:
                    new_linear.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        new_linear.bias.data.copy_(module.bias.data)
            else:
                new_linear = LoRALinear(module.in_features, module.out_features, peft_config, bias=use_bias)
                new_linear.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    new_linear.bias.data.copy_(module.bias.data)

            return new_linear
        elif isinstance(module, nn.Linear) and (
                any(module_name in target_module_name for target_module_name in peft_config.quantize_modules) or
                peft_config.quantize_modules == 'all_linear'):
            use_bias = module.bias is not None
            if peft_config.use_qlora:
                new_linear = LinearKBit(module.in_features, module.out_features, bias=use_bias, quantizer=quantizer)
                # new_linear = LinearKbitNew(module.in_features, module.out_features, peft_config, bias=use_bias,
                #                          quantizer=quantizer)
                # 如果还没量化，就量化再赋值
                if module.weight.dtype is not torch.int8:
                    quantized_weight = quantizer.quantize(module.weight.data)
                    new_linear.weight.data.copy_(quantized_weight)
                    if module.bias is not None:
                        new_linear.bias.data.copy_(quantizer.quantize(module.bias.data))
                else:
                    new_linear.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        new_linear.bias.data.copy_(module.bias.data)
                return new_linear

        return module

    new_model = update_model(new_model,replace_linear)

    return new_model
from myquantization.myquantize import LinearKBit,QuantizationHandler
def merge_lora_back_to_pertrain_model(lora_model:nn.Module,quantizer:QuantizationHandler):
    """
    针对普通lora，不需要使用quantizer
    针对qlora，需要使用quantizer
    """
    has_quant_module = False
    for name,module in lora_model.named_modules():
        if isinstance(module,LinearKBit):
            has_quant_module = True
            break
    if has_quant_module:
        assert quantizer is not None,"针对qlora的量化模型，必须传入quantizer，用于把非lora模块转化回普通linear"
    def replace_lora_linear_with_linear(module,module_name):

        if isinstance(module, LoRALinear):
            bias = module.bias
            use_bias = True if bias is not None else False
            in_features = module.in_features
            out_features = module.out_features
            new_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=use_bias)

            merged_weight = module.merge_lora_weight_and_linear_weight()
            new_linear.weight.data.copy_(merged_weight.data)
            if use_bias:
                new_linear.bias.data.copy_(module.bias.data)
            return new_linear
        elif isinstance(module,LinearKBit):
            bias = module.bias
            use_bias = True if bias is not None else False
            in_features = module.in_features
            out_features = module.out_features
            new_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=use_bias)

            # 还需要反量化回float32，否则获得梯度
            dequantized_w = quantizer.dequantize(module.weight.data)
            new_linear.weight.data.copy_(dequantized_w)
            if use_bias:
                dequantized_b = quantizer.dequantize(module.bias.data)
                new_linear.bias.data.copy_(dequantized_b)
            return new_linear
        return module
    new_lora_model = copy.deepcopy(lora_model)
    merged_model = update_model(new_lora_model,replace_lora_linear_with_linear)
    merged_model = set_parameters_requires_grad(merged_model,trainable=True)
    return merged_model

if __name__ == '__main__':
    pass



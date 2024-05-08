"""
1.定义量化函数：
编写一个函数，接受一个浮点数类型的张量作为输入，并将其量化为指定精度的整数类型张量。这个函数将会根据传入的精度参数，将张量的值映射到相应的整数区间上。

2.定义自定义的线性层类（LinearKBit）：
创建一个类，它继承自 PyTorch 中的 nn.Module 类，并用于替换模型中所有的线性层。这个类将会接受一个配置对象作为参数，在初始化函数中根据配置将权重量化为指定精度的整数类型。

3.定义配置类（QuantizationConfig）：
编写一个配置类，用于存储模型量化的相关配置信息，例如需要被排除的模块、量化精度等。

4.在模型中替换线性层：
遍历模型的所有模块，将所有的线性层替换为自定义的 LinearKBit 类。对于需要排除的模块，不做替换。
量化权重：
在替换的线性层中，调用量化函数来将原始的浮点数权重量化为指定精度的整数类型，并赋值给自定义的线性层。
"""
from dataclasses import dataclass,field
from typing import List
import torch
import torch.nn as nn

@dataclass
class QuantizationConfig:
    exclude_modules: List[str] = field(default_factory=lambda: ['lm_head'])
    precision: int = 4


class QuantizationHandler:
    def __init__(self, min_val,max_val,n_bits=8):
        self.n_bits = n_bits
        self.quant_min_val = -2 ** (n_bits - 1)
        self.quant_max_val = 2 ** (n_bits - 1) - 1
        if not min_val:
            min_val = torch.finfo(torch.float32).min
        if not max_val:
            max_val = torch.finfo(torch.float32).max
        self.min_val = min_val
        self.max_val = max_val

        # 根据 n_bits 计算 scale 和 zero_point
        self.scale, self.zero_point = self.calculate_scale_zero_point()

    def calculate_scale_zero_point(self):
        # 根据 n_bits 计算 scale 和 zero_point
        scale = (self.max_val - self.min_val) / (self.quant_max_val - self.quant_min_val)
        zero_point = self.quant_max_val - self.max_val / scale
        return scale, zero_point

    def quantize(self, x):
        # 计算量化后的值
        x_q = (x.div(self.scale) + self.zero_point).round()
        # 将量化后的值限制在合法范围内
        x_q_clipped = torch.clamp(x_q, min=self.quant_min_val, max=self.quant_max_val)
        # TODO 这个代码要不要加?
        x_q_clipped = x_q_clipped.to(torch.int8)
        return x_q_clipped

    def dequantize(self, x_q):
        # 将量化后的值转换为整型
        x_q_int = x_q.to(torch.int8)
        # 计算反量化后的值
        x = self.scale * (x_q_int - self.zero_point)
        # 将结果转换为浮点型
        x = x.to(torch.float32)
        return x



import torch.nn.functional as F

class LinearKBit(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, quantizer: QuantizationHandler = None):
        super().__init__(in_features, out_features, bias)
        self.quantizer = quantizer
        self.weight.requires_grad = False
        self.weight.data = self.quantizer.quantize(self.weight.data)
        # self.weight.data = self.weight.data.to(torch.int8)
        if bias:
            self.bias.requires_grad_(False)
            self.bias.data = self.bias.data.to(torch.int8)
        self.inputs = None

    def forward(self, inputs: torch.Tensor):
        # self.inputs = inputs.clone().detach()
        if inputs.dtype in (torch.float32, torch.float16, torch.float):
            quantized_inputs = self.quantizer.quantize(inputs)
        else:
            quantized_inputs = inputs
        assert quantized_inputs.dtype is torch.int8, '输入必须转化为int8类型'
        output = F.linear(quantized_inputs, self.weight, self.bias)
        output_float32 = self.quantizer.dequantize(output)
        return output_float32

    # def backward(self, grad_output):
    #     # 计算权重的梯度
    #     if self.weight.requires_grad:
    #         grad_weight = torch.matmul(grad_output.t(), self.inputs).to(torch.int8)
    #         self.weight.grad = grad_weight
    #
    #     # 计算偏置的梯度
    #     if self.bias is not None and self.bias.requires_grad:
    #         grad_bias = grad_output.sum(0).to(torch.int8)
    #         self.bias.grad = grad_bias
    #
    #     # 计算输入的梯度
    #     if self.inputs.requires_grad:
    #         if grad_output is not torch.int8:
    #             grad_output = self.quantizer.quantize(grad_output)
    #         grad_input = torch.matmul(grad_output, self.weight)
    #         grad_output = self.quantizer.dequantize(grad_output)
    #         self.inputs.grad = grad_input
    #
    #     return grad_input



def replace_linear_with_linearkbit(
    model,
    config:QuantizationConfig,
    quantizer:QuantizationHandler,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    for name, module in model.named_children():
        print(f"module name is {name}")
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if (isinstance(module, nn.Linear) ) and name not in config.exclude_modules:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in config.exclude_modules):

                use_bias = module.bias is not None

                linearkbit = LinearKBit(in_features=module.in_features,out_features=module.out_features,
                                        bias=use_bias,quantizer=quantizer)
                linearkbit.weight.requires_grad = False
                quantized_old_weight_data = quantizer.quantize(module.weight.data)

                linearkbit.weight.data = quantized_old_weight_data

                model._modules[name] = linearkbit
                has_been_replaced = True

                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                linearkbit.weight.requires_grad = False

        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_linear_with_linearkbit(
                module,
                config,
                quantizer,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced

if __name__ == '__main__':
    # linear_4bit = LinearKBit(2, 2, bias=False)
    # print(linear_4bit.weight)

    from model import GPT,GPTConfig
    config = GPTConfig(n_layer=32,n_embd=256,n_head=4)
    model = GPT(config)
    print(model)
    print("----------------")
    quantize_config:QuantizationConfig = QuantizationConfig(precision=4,exclude_modules=['lm_head'])
    import copy
    new_model = copy.deepcopy(model)
    from mylora.minilora import set_parameters_requires_grad
    new_model = set_parameters_requires_grad(new_model)
    quantizer = QuantizationHandler(min_val=-1,max_val=10,n_bits=8)
    quantized_model, has_been_replaced = replace_linear_with_linearkbit(new_model,config=quantize_config,quantizer=quantizer)
    print(quantized_model)

    from utils.basic_utils import calculate_model_memory
    print(f"origin model size {calculate_model_memory(model)[2]}Mb")
    print(f"quantized model size {calculate_model_memory(quantized_model)[2]}Mb")


"""
这里在替换时是否有必要，对linearkbit中的weight进行量化呢?
不需要，跟普通linear没有区别，
简单设置下不需要梯度
可以做把linear.weight.data转变为int4



                pytorch的类型规则：
                1. 设置linear 不需要梯度时，可以linear.weight.data = new_tensor(int8) 使用int8做为weight
                2. 但是，此时对输入x的要求也要是同类型的，int，这样才能执行y = linear(x)
                3. 设置linear需要梯度时，linear.weight.data = new_tensor(只能是float和complex)
                4. 对于2的进一步思考，为什么加法不要求必须是同类型，而linear(x)的矩阵乘法必须要求呢？* 元素相乘不要求
                为什么lm层不能做量化？为什么要确保layernorm出来后的结果必须是float？
                因为，loraAB是float32，必须要求输入给loraab的x必须要求是float否则不能做矩阵乘法
                所以，lm还是float32，找embeding后的数据x也还是float32，经过input = layernorm((lm(x)))
                
                
                为什么要单独写一个qloralinear类？
                因为，要把输入x，转为int去和量化后的linear做乘法，使用float x去和loraab做矩阵乘法；要求分开处理
                                
                这样就需要我们实现一个qloralinear，其中做这样的操作
                qloralinear 
                1. 把x转换为int4去和linear weight做乘法
                2. lora out + linear out 不需要做类型变化，自动用float做加法
                                
                TODO 
                1. 先实现量化，把pretain模型所有除了lm外的linear都转化为int8
                2. 然后，再用lora替换，全部替换为qlora
                3. 训练后合并环节，再做反量化再做相加合并
                
                为什么linear(x)要求x和linear.weight.data必须是相同类型？
                难道不同类型tensor可以做加法但是不能做乘法吗？
                确实如此，pytorch强制要求矩阵乘法必须是相同类型的tensor，但是A * B逐元素相乘不要求必须是相同类型
                这个可能是受制于cuda的限制，这只是我的一种猜测
"""
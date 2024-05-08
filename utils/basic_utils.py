import torch

import  numpy as np
def moving_average(data, window_size):
    """计算移动平均"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_even_odd_elements(sequence):
    """
    从长度为 n 的序列中获得所有奇数索引和偶数索引的元素。

    参数：
        sequence (torch.Tensor)：输入序列，形状为 [n]。

    返回：
        even_elements (torch.Tensor)：所有偶数索引的元素，形状为 [n//2]。
        odd_elements (torch.Tensor)：所有奇数索引的元素，形状为 [n//2]。
    """
    # 获取偶数索引的元素
    even_elements = sequence[::2]

    # 获取奇数索引的元素
    odd_elements = sequence[1::2]

    return even_elements, odd_elements



def convert_1d_pad_mask_to_2d_attention_mask(attention_mask):
    # 将输入的 attention_mask 调整为下三角形式的 mask
    # 下三角形式的 mask 是模型所需的格式
    if len(attention_mask.size())==2:
        batch_size, seq_len = attention_mask.size()
    elif len(attention_mask.size()) == 3:
        batch_size, pair_size,seq_len = attention_mask.size()

    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long, device=attention_mask.device))
    expanded_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展为 batch 的形状

    # 利用 outer 乘积生成下三角形式的 mask
    adjusted_mask = torch.bmm(attention_mask.unsqueeze(-1), attention_mask.unsqueeze(-2))
    # adjusted_mask = torch.outer(attention_mask, attention_mask).reshape(batch_size, seq_len, seq_len)
    """
    torch.bmm() 期望输入张量的维度为 (batch_size, n, m) 和 (batch_size, m, p)，并返回维度为 (batch_size, n, p) 的结果张量。
    torch.outer() 函数接受两个一维张量作为输入，并返回一个矩阵，其元素是输入向量的乘积。例如，给定向量 a 和 b，torch.outer(a, b) 将返回一个矩阵 C
    所以，这里适合应用bmm而不是，outer
    """

    # 将生成的下三角形式的 mask 和模型所需的 mask 相乘
    final_mask = adjusted_mask * expanded_mask.float()  # 将 mask 张量转换为 float 类型，以兼容 torch.bmm 的要求
    # (batch_size,1,T,T) ，增加1这个维度，当和多头进行fill mask时，会进行广播
    final_mask = final_mask.unsqueeze(1)
    return final_mask

def calculate_model_memory(model, trainable_only=True):
    total_memory_size = 0
    for param_name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        num_elements = param.numel()
        element_size_bytes = param.element_size()
        param_memory_size_bytes = num_elements * element_size_bytes
        total_memory_size += param_memory_size_bytes

    # 将字节数转换为更友好的单位（例如 KB、MB 等）
    total_memory_size_kb = total_memory_size / 1024
    total_memory_size_mb = total_memory_size / (1024 * 1024)

    return total_memory_size, total_memory_size_kb, total_memory_size_mb

def calculate_model_training_memory(model):
    total_memory_size = 0
    for param_name, param in model.named_parameters():
        num_elements = param.numel()
        element_size_bytes = param.element_size()
        param_memory_size_bytes = num_elements * element_size_bytes
        if param.requires_grad:
            total_memory_size += 3*param_memory_size_bytes # 包含参数本身，优化器所需的梯度均值，梯度方差
        else:
            total_memory_size += param_memory_size_bytes  # 只做推理，所以只需要参数本身，不需要梯度信息


    # 将字节数转换为更友好的单位（例如 KB、MB 等）
    total_memory_size_kb = total_memory_size / 1024
    total_memory_size_mb = total_memory_size / (1024 * 1024)

    return total_memory_size, total_memory_size_kb, total_memory_size_mb


import torch
from torchviz import make_dot


def visualize_model(model, dummy_input):
    """
    使用Torchviz可视化PyTorch模型的计算图

    参数:
        model (torch.nn.Module): 需要可视化的PyTorch模型
        input_size (tuple): 模型输入的尺寸,如(1, 3, 224, 224)

    返回:
        None
    """
    # 构造一个dummy输入
    # dummy_input = torch.randn(input_size)
    res = model(dummy_input)[0]
    # 用 make_dot 获取 PyTorch 执行图
    vis_graph = make_dot(res, params=dict(model.named_parameters()))

    # 用 render 渲染图像
    vis_graph.render("model_graph", view=True)

    print("模型计算图已可视化并保存为 model_graph.pdf")


import torch
from torch.utils.tensorboard import SummaryWriter


def visualize_model_with_tensorboard(model, dummy_input):
    """
    使用 TensorBoard 可视化 PyTorch 模型的计算图

    参数:
        model (torch.nn.Module): 需要可视化的 PyTorch 模型
        input_size (tuple): 模型输入的尺寸,如 (1, 3, 224, 224)

    返回:
        None
    """
    # # 构造虚拟输入
    # dummy_input = torch.randn(input_size)

    # 创建 SummaryWriter 实例
    writer = SummaryWriter()

    # 将模型图添加到 TensorBoard
    writer.add_graph(model, input_to_model=dummy_input)

    # 关闭 SummaryWriter
    writer.close()

    print("模型计算图已记录到 TensorBoard, 请运行 'tensorboard --logdir=runs' 并在浏览器中查看")

# 示例使用
if __name__ == "__main__":


    # 定义一个简单的模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.fc2 = torch.nn.Linear(20, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    # 创建一个示例模型
    model = SimpleModel()
    # visualize_model(model,(1,10))
    visualize_model_with_tensorboard(model,(1,10))


    from myquantization.myquantize import QuantizationHandler,QuantizationConfig,replace_linear_with_linearkbit
    quantize_config:QuantizationConfig = QuantizationConfig(precision=4,exclude_modules=['lm_head'])
    import copy
    new_model = copy.deepcopy(model)
    from mylora.minilora import set_parameters_requires_grad
    new_model = set_parameters_requires_grad(new_model)
    quantizer = QuantizationHandler(min_val=-1,max_val=10,n_bits=8)
    quantized_model, has_been_replaced = replace_linear_with_linearkbit(new_model,config=quantize_config,quantizer=quantizer)
    print(quantized_model)

    # 计算模型占用的内存大小
    total_memory_size, total_memory_size_kb, total_memory_size_mb = calculate_model_memory(quantized_model)
    print("-----------quant model---------")
    print(f"模型占用的内存大小：{total_memory_size} 字节")
    print(f"模型占用的内存大小：{total_memory_size_kb} KB")
    print(f"模型占用的内存大小：{total_memory_size_mb} MB")

    print("-----------original model---------")
    # 计算模型占用的内存大小
    total_memory_size, total_memory_size_kb, total_memory_size_mb = calculate_model_memory(model)
    print(f"模型占用的内存大小：{total_memory_size} 字节")
    print(f"模型占用的内存大小：{total_memory_size_kb} KB")
    print(f"模型占用的内存大小：{total_memory_size_mb} MB")

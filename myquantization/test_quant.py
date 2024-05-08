import torch

import torch


def quantize_per_tensor(tensor, scale, zero_point, dtype):
    # 将输入张量按照缩放因子和零点因子进行量化
    # 公式：𝒙𝒒 = 𝑅(𝒙/𝑆) − 𝑍
    quantized_tensor = torch.round(tensor / scale) - zero_point

    # 限制量化后的张量值在合法范围内
    quantized_tensor = torch.clamp(quantized_tensor, torch.iinfo(dtype).min, torch.iinfo(dtype).max)

    # 转换为指定的数据类型
    # 显示指定量化后的数据类型
    # quantized_tensor = quantized_tensor.to(torch.float)  # 将量化后的张量转换为浮点型
    quantized_tensor = quantized_tensor.contiguous()
    quantized_tensor = quantized_tensor.to(dtype)  # 再转换为指定的数据类型

    return quantized_tensor
if __name__ == '__main__':
    # 示例用法
    model_weight = torch.randn(2, 2)

    scale = torch.tensor(0.05)  # 缩放因子
    zero_point = torch.tensor(0)  # 零点因子
    quantized_weight = quantize_per_tensor(model_weight, scale, zero_point, torch.qint8)
    print(quantized_weight)


# # 假设 model 是一个已经定义好的 nn.Linear 层
# model = torch.nn.Linear(10, 20)
#
# # 收集权重的统计信息
# min_val, max_val = torch.min(model.weight), torch.max(model.weight)
#
# # 计算量化尺度和零点
# scale = max(abs(min_val), abs(max_val)) / 127.0
# zero_point = torch.tensor(0, dtype=torch.int32)
#
# # 量化权重
# quantized_weight = quantize_per_tensor(model.weight, scale, zero_point, torch.qint8)
#
# # 将量化后的权重赋值回模型的权重
# model.weight.data = quantized_weight
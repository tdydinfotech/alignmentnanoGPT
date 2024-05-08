
import torch


if __name__ == "__main__":
    # 创建 QuantizationHandler 实例
    quant_handler = QuantizationHandler(min_val=-1,max_val=10,n_bits=8)

    # 示例输入张量
    float_x = torch.tensor([1.5, 2.5, 3.5])

    # 量化过程
    x_q_clip = quant_handler.quantize(float_x)
    print(f"输入：\n{float_x}\n")
    print(f"{quant_handler.n_bits}比特量化后：\n{x_q_clip}")
    print(f"量化后的数据类型是{x_q_clip.dtype}")

    # 反量化过程
    x_re = quant_handler.dequantize(x_q_clip)
    print(f"反量化后：\n{x_re}")
    print(f"反对量化后的数据类型是{x_re.dtype}")


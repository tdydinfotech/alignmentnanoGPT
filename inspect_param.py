import inspect
import torch
if __name__ == '__main__':
    # 定义一个简单的函数
    def example_function(x, y=10, z="default"):
        return x + y

    # 获取函数的参数签名
    signature = inspect.signature(example_function)

    # 打印参数签名信息
    print("函数的参数签名:", signature)

    # 获取参数的名称和默认值
    for parameter_name, parameter in signature.parameters.items():
        print("参数名称:", parameter_name)
        print("参数默认值:", parameter.default)

from model import LayerNorm, GPTConfig

import torch.nn as nn

# 定义一个测试函数
def test_MLP():
    # 创建一个 GPTConfig 对象
    config = GPTConfig()
    config.n_embd = 48
    config.dropout = 0

    # 创建一个 MLP 模块
    mlp = MLP(config)

    # 准备输入数据，例如一个随机张量
    input_data = torch.randn(4, 10,config.n_embd)  # batch_size=32, 长度是10 输入维度为 config.n_embd

    # 将输入数据传递给 MLP 模块，得到输出结果
    output = mlp(input_data)

    # 打印输出结果的形状
    print("Output shape:", output.shape)




def test_layer_norm():
    # 创建一个 LayerNorm 实例
    ndim = 10  # 假设输入张量的维度为10
    bias = True  # 启用偏置
    layer_norm = LayerNorm(ndim, bias)

    # 准备一个输入张量（示例：批次大小为4，特征维度为10）
    input_tensor = torch.randn(4,128, ndim)

    # 将输入张量传递给 LayerNorm 实例的 forward 方法
    output_tensor = layer_norm(input_tensor)

    # 打印输出张量
    print("Output tensor:", output_tensor.shape)
# 导入必要的库
import torch
from model import CausalSelfAttention
def test_self_attn():


    # 创建模型配置
    config = GPTConfig(n_embd=96, n_head=3, block_size=128, dropout=0.1, bias=True)

    # 创建CausalSelfAttention模型
    causal_self_attention = CausalSelfAttention(config)

    # 生成随机输入数据
    input_data = torch.randn(8, 128, 96)  # (B,T,h) (batch_size, sequence_length, embedding_dim)

    # 进行前向传播
    output = causal_self_attention(input_data)

    # 输出结果的形状
    print("Output shape:", output.shape)
from model import MLP
def test_ffn():
    # 创建模型配置
    config = GPTConfig(n_embd=16, n_head=3, block_size=128, dropout=0.1, bias=True, ffn_multiply_scalar=2)

    # FFN
    ffn = MLP(config)

    # 生成随机输入数据
    input_data = torch.randn(8, 128, 16)  # (B,T,h) (batch_size, sequence_length, embedding_dim)

    # 进行前向传播
    output = ffn(input_data)

    # 输出结果的形状
    print("Output shape of FFN:", output.shape)
def test_attn_and_ffn_block():
    from model import Block
    # 创建模型配置
    n_emb = 96
    text_len = 128
    n_head = 3
    batch_size = 8
    config = GPTConfig(n_embd=n_emb, n_head=n_head, block_size=text_len, dropout=0.1, bias=True, ffn_multiply_scalar=2)

    # FFN
    attn_and_ffn = Block(config)

    # 生成随机输入数据
    input_data = torch.randn(batch_size, text_len, n_emb)  # (B,T,h) (batch_size, sequence_length, embedding_dim)

    # 进行前向传播
    output = attn_and_ffn(input_data)

    # 输出结果的形状
    print("Output shape of FFN:", output.shape)

def test_gpt():

    # 创建 GPTConfig 对象
    config = GPTConfig(vocab_size=10, block_size=32,
                       n_embd=16,n_head=2,
                       n_layer=1, dropout=0.1,
                       bias=False,ffn_multiply_scalar=2)
    from model import GPT
    # 创建 GPT 模型实例
    model = GPT(config)
    batch_size = 3
    # 准备输入张量和目标张量（如果需要计算损失的话）
    idx = torch.randint(0, config.vocab_size, (batch_size, config.block_size))  # 示例输入张量，假设批次大小为 2
    targets = torch.randint(0, config.vocab_size, (batch_size, config.block_size))  # 示例目标张量，假设批次大小为 2
    # 测试纯推理模式
    targets = None
    # 调用 forward 方法
    logits, loss = model.forward(idx, targets)

    # 打印输出结果
    print("logits shape:", logits.shape)
    print("loss:", loss)
def test_optimizer():
    from model import GPT
    config = GPTConfig(vocab_size=10, block_size=32,
                       n_embd=16,n_head=2,
                       n_layer=1, dropout=0.1,
                       bias=False,ffn_multiply_scalar=2)
    model = GPT(config)
    beta1 = 0.9
    beta2 = 0.95
    learning_rate = 6e-4  # 最大学习率
    weight_decay = 1e-1
    device_type = 'cpu'
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

def test_rmsnorm():
    from model import RMSNorm
    b, t, h = 2, 128, 8
    input_tensor = torch.randn(b, t, h)
    rms_norm_layer = RMSNorm(h)
    output = rms_norm_layer(input_tensor)
    print(output.shape)

def test_glu():
    from model import GLU
    # 创建 GLU 实例
    config = GPTConfig(n_embd=32, glu_multiply_scalar=3)
    glu_layer = GLU(config)

    # 构造输入张量
    input_tensor = torch.randn(3,128,32)  # 32为batch_size，256为特征维度

    # 前向传播
    output = glu_layer(input_tensor)

    print(output.shape)  # 输出形状

def test_moe():
    from model import GLU,Moe,MoeLayer
    h = 64
    config = GPTConfig(n_embd=h, glu_multiply_scalar=3,num_expert_per_token=2)
    experts = [GLU(config), GLU(config), GLU(config), GLU(config)]
    num_expert = experts.__len__()
    gate = nn.Linear(config.n_embd, num_expert)
    moe1 = Moe(experts=experts, gate=gate, num_experts_per_token=config.num_expert_per_token)
    moe2 = MoeLayer(experts=experts, gate=gate, num_experts_per_token=config.num_expert_per_token)

    # 构造输入张量
    input_tensor = torch.randn(10,128,h)  # 32为batch_size，256为特征维度
    import timeit
    # 定义前向传播函数
    def forward_pass_moe1():
        output = moe1(input_tensor)

    def forward_pass_moe2():
        output = moe2(input_tensor)

    # 测量 moe1 的前向传播时间
    time_moe1 = timeit.timeit(forward_pass_moe1, number=10)

    # 测量 moe2 的前向传播时间
    time_moe2 = timeit.timeit(forward_pass_moe2, number=10)

    # 打印输出形状和时间
    print("Output Shape of moe1:", moe1(input_tensor).shape)
    print("Output Shape of moe2:", moe2(input_tensor).shape)
    print("Elapsed Time for moe1:", time_moe1, "seconds")
    print("Elapsed Time for moe2:", time_moe2, "seconds")
    """
    Output Shape of moe1: torch.Size([10, 128, 64])
    Output Shape of moe2: torch.Size([10, 128, 64])
    Elapsed Time for moe1: 3.488404084 seconds
    Elapsed Time for moe2: 0.08959358300000009 seconds
    3.5/0.09 = 40倍，tensor越大，差别越大，从专家角度计算效率很高
    """


def find_expert_with_where_operator():
    import torch

    # 创建 selected_experts 张量
    selected_experts = torch.tensor([[[0, 1],  # batch 1, seq_len 1
                                      [3, 2],  # batch 1, seq_len 2
                                      [1, 0]],  # batch 1, seq_len 3
                                     [[2, 1],  # batch 2, seq_len 1
                                      [0, 2],  # batch 2, seq_len 2
                                      [3, 1]]])  # batch 2, seq_len 3

    # 假设我们要计算 batch_size = 2 时，对应 selected_experts 中每个专家的索引
    for i in range(4):  # 假设有 4 个专家
        indices = torch.where(selected_experts == i)
        sentenc_idx = indices[0]
        token_idx = indices[1]
        prob_idx = indices[2]
        print(f"Expert {i}:")
        print("sentenc_idx:", sentenc_idx)
        print("token_idx:", token_idx)
        print("prob_idx:", prob_idx)
    """
    sentenc_idx: tensor([0, 0, 1])
    两个代表第一个句子
    token_idx: tensor([0, 2, 1])
    0，2代表第一个句子中的第1个第3个token要被专家0处理
    prob_idx: tensor([0, 1, 0])
    0，1代表，选出来的两个专家中，专家所在的位置，用来映射寻找专家所对应的概率    
    """
def how_topk_grad():
    """
    想要理解topk这样的算子如何处理梯度
    input_tensor 的grad
    tensor([[0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.]])
    看来是只对输出的最大值计算了grad，其他值的grad=0
    """

    # 创建输入张量
    input_tensor = torch.tensor([[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0],
                                 [7.0, 8.0, 9.0]], requires_grad=True)

    # 进行 torch.topk
    values, indices = torch.topk(input_tensor, k=1)
    print("Topk values:", values)
    print("Topk indices:", indices)

    # 计算某个标量值作为损失
    loss = values.sum()
    print("Loss:", loss)

    # 反向传播
    loss.backward()

    # 打印输入张量的梯度
    print("Gradient of input tensor:")
    print(input_tensor.grad)



def check_theta_between_q_and_rotated_q(q,rotated_q):
    # 计算点积
    dot_product = (q * rotated_q).sum(dim=1)

    # 计算模长
    norm_q = torch.norm(q, dim=1)
    norm_rotated_q = torch.norm(rotated_q, dim=1)

    # 计算余弦值
    cos_theta = dot_product / (norm_q * norm_rotated_q)

    # 由于浮点数精度问题，余弦值可能会略微超出[-1, 1]的范围，我们需要将其裁剪到这个范围内
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # 计算夹角，结果以弧度为单位
    theta = torch.acos(cos_theta)

    # 将夹角转换为度数（如果需要）
    theta_degrees = theta * (180.0 / 3.141592653589793)

    # 输出结果
    print("夹角（弧度）:", theta)
    print("夹角（度）:", theta_degrees)




def show_rotate_vec(q,rotated_q):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # 原始向量
    # q = np.array([[-0.0399, 0.9648],
    #               [-0.3585, -0.7673],
    #               [0.4958, 0.5821],
    #               [0.7198, -0.3276],
    #               [-0.9673, 0.0799],
    #               [0.0793, 1.2559]])
    #
    # # 旋转后的向量
    # rotated_q = np.array([[-0.0399, 0.9648],
    #                       [0.4520, -0.7162],
    #                       [-0.7357, 0.2086],
    #                       [-0.6664, 0.4258],
    #                       [0.6928, 0.6798],
    #                       [1.2268, 0.2802]])

    # 设置颜色映射
    color_map = cm.get_cmap('viridis', len(q))

    # 绘制向量图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(len(q)):
        axs[0].quiver(0, 0, q[i, 0], q[i, 1], angles='xy', scale_units='xy', scale=1, color=color_map(i),
                      label=f'Original {i + 1}')
        axs[1].quiver(0, 0, rotated_q[i, 0], rotated_q[i, 1], angles='xy', scale_units='xy', scale=1,
                      color=color_map(i), label=f'Rotated {i + 1}')

    axs[0].set_title('Original Vectors')
    axs[1].set_title('Rotated Vectors')

    # 设置图例
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')

    # 设置坐标轴范围
    axs[0].set_xlim([-1.5, 1.5])
    axs[0].set_ylim([-1.5, 1.5])
    axs[1].set_xlim([-1.5, 1.5])
    axs[1].set_ylim([-1.5, 1.5])

    plt.show()


def test_rotate():
    batch_size = 8

    seq_len = 128
    dim = 32
    q = torch.randn(batch_size,seq_len,dim)
    from model import Rope
    rotated_q = Rope(base_value=10000).apply_rotatory_emb(q)
    print(f'q : {q}\n')
    print(f'rotated_q : {rotated_q}')
    print(f'q : {q.size()}\n')
    print(f'rotated_q : {rotated_q.size()}')
    # check_theta_between_q_and_rotated_q(q,rotated_q)
    # show_rotate_vec(q,rotated_q)

def test_flatten():
    import torch

    # 创建一个形状为(2, 3, 4, 5)的张量
    tensor = torch.randn(2, 3, 4, 5)

    # 将整个张量展平成一维张量
    flattened_tensor_default = torch.flatten(tensor)
    print("Default flattened shape:", flattened_tensor_default.shape)
    # Default flattened shape: torch.Size([120])


    # 从第2个维度开始展平张量
    # flattened_tensor_dim2 = torch.flatten(tensor, start_dim=-2)
    flattened_tensor_dim2 = torch.flatten(tensor, start_dim=2)
    print("Flattened shape from dimension 2:", flattened_tensor_dim2.shape)
    # Flattened shape from dimension 2: torch.Size([2, 3, 20])


    # 从第3个维度开始展平张量
    flattened_tensor_dim3 = torch.flatten(tensor, start_dim=-1)
    print("Flattened shape from dimension 3:", flattened_tensor_dim3.shape)
    # Flattened shape from dimension 3: torch.Size([2, 3, 4, 5])

def test_cache():
    from model import CacheLowFrequecyAllocateMem
    cache = CacheLowFrequecyAllocateMem(max_batch_size=2, max_seq_len=5, n_head=3, head_dim=4,
                                        seq_len_increment_for_cache_allocate=5)
    # 创建第一个新的张量用于保存到缓存中
    new_tensor_1 = torch.randn(2, 3, 3, 4)

    # 保存第一个张量到缓存中
    cache.save_to_cache(new_tensor_1)

    # 获取缓存中的所有张量并打印
    cached_tensors_1 = cache.get_all_from_cache()
    print("Cached Tensors after first save:")
    print(cached_tensors_1.shape)
    print(f'cur cache size {cache.cache.shape}')

    # 创建第二个新的张量用于保存到缓存中
    new_tensor_2 = torch.randn(2, 2, 3, 4)

    # 保存第二个张量到缓存中
    cache.save_to_cache(new_tensor_2)

    # 获取缓存中的所有张量并打印
    cached_tensors_2 = cache.get_all_from_cache()
    print("Cached Tensors after second save:")
    print(cached_tensors_2.shape)
    print(f'cur cache size {cache.cache.shape}')

    # 创建第三个新的张量用于保存到缓存中
    new_tensor_3 = torch.randn(2, 2, 3, 4)

    # 保存第二个张量到缓存中
    cache.save_to_cache(new_tensor_3)

    # 获取缓存中的所有张量并打印
    cached_tensors_3 = cache.get_all_from_cache()
    print("Cached Tensors after 3 save:")
    print(cached_tensors_3.shape)
    print(f'cur cache size {cache.cache.shape}')

    # 创建第三个新的张量用于保存到缓存中
    new_tensor_4 = torch.randn(2, 12, 3, 4)

    # 保存第二个张量到缓存中
    cache.save_to_cache(new_tensor_4)

    # 获取缓存中的所有张量并打印
    cached_tensors_4 = cache.get_all_from_cache()
    print("Cached Tensors after 4 save:")
    print(cached_tensors_4.shape)
    print(f'cur cache size {cache.cache.shape}')


def test_get_batch():
    from mytrain import get_batch
    from mygenerate import load_tokenizer
    encode, decode = load_tokenizer(None)

    config = GPTConfig(batch_size=1,block_size=5)
    X,Y = get_batch(split='test',config=config)

    print(f'x shape is {X.shape}')
    print(f'y shape is {Y.shape}')

    print(decode( X[0,:].tolist() ) )
    print(decode( Y[0,:].tolist() ) )


def test_gpt_with_attention_mask():
    torch.manual_seed(0)
    # 创建 GPTConfig 对象
    config = GPTConfig(vocab_size=10, block_size=32,
                       n_embd=16,n_head=2,
                       n_layer=1, dropout=0.1,
                       bias=False,ffn_multiply_scalar=2)
    from model import GPT
    # 创建 GPT 模型实例
    model = GPT(config)
    batch_size = 3
    # 准备输入张量和目标张量（如果需要计算损失的话）
    idx = torch.randint(0, config.vocab_size, (batch_size, config.block_size))  # 示例输入张量，假设批次大小为 2
    targets = torch.randint(0, config.vocab_size, (batch_size, config.block_size))  # 示例目标张量，假设批次大小为 2
    # 测试纯推理模式
    targets = None
    # TODO 测试
    T = config.block_size
    mask = torch.tril(torch.ones(T, T))
    mask = mask.view(1, 1, T, T)
    attention_mask = mask
    # attention_mask = None
    # 调用 forward 方法
    logits, loss = model.forward(idx, targets,attention_mask=attention_mask)
    # 输出 batch_size,vocb_size 是因为，已经seqlen中的最后一个token做计算了

    # 打印输出结果
    print("logits shape:", logits.shape)
    print("loss:", loss)
    print(f'loss is {logits}')
    """
    loss is tensor([[ 1.2103, -0.3646,  0.9073, -0.5889,  0.2318, -0.2715,  0.0383,  0.4845,
         -0.3163, -0.6215],
        [ 1.2627, -0.5007,  0.7157, -0.3791,  0.2406, -0.1595,  0.0017,  0.3630,
         -0.2837, -0.6964],
        [ 1.5619, -0.3089,  0.5586, -0.6794,  0.0700, -0.2385, -0.1387,  0.3672,
         -0.4552, -0.4663]], grad_fn=<MmBackward0>)
    loss is tensor([[ 1.2103, -0.3646,  0.9073, -0.5889,  0.2318, -0.2715,  0.0383,  0.4845,
             -0.3163, -0.6215],
            [ 1.2627, -0.5007,  0.7157, -0.3791,  0.2406, -0.1595,  0.0017,  0.3630,
             -0.2837, -0.6964],
            [ 1.5619, -0.3089,  0.5586, -0.6794,  0.0700, -0.2385, -0.1387,  0.3672,
             -0.4552, -0.4663]], grad_fn=<MmBackward0>)
    对比使用传入的attention mask 和 None两种情况下的logits是否一致
    结果一样
  """

from utils.basic_utils import convert_1d_pad_mask_to_2d_attention_mask
def test_adjust_attention_mask():
    # 测试示例
    attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        # [1, 1, 0, 0, 0],
        # [1, 1, 1, 1, 1],
        # [1, 1, 1, 1, 0],
        # [1, 0, 0, 0, 0]
    ], dtype=torch.float)
    adjusted_mask = convert_1d_pad_mask_to_2d_attention_mask(attention_mask)
    print(adjusted_mask)

def test_gpt_config():
    # 示例使用
    config = GPTConfig()
    config_dict = config.to_dict()
    print(config_dict)

if __name__ == '__main__':
    # test_gpt_config()
    # test_gpt_with_attention_mask()
    # test_get_batch()
    # test_layer_norm()
    # # 测试
    # input_data = torch.randn(3, 4, 5)  # batch_size=3, sequence_length=4, features=5
    # layer_norm = MyLayerNorm(5,bias=True)  # 对最后一维进行Layer Norm
    # output = layer_norm(input_data)
    # print(output)

    # 调用测试函数
    # test_MLP()

    # test_self_attn()

    # test_ffn()

    # test_attn_and_ffn_block()

    # test_gpt()

    # test_optimizer()

    # test_rmsnorm()

    # test_glu()
    # find_expert_with_where_operator()
    # test_moe()

    # how_topk_grad()
    # test_rotate()
    # test_flatten()
    test_cache()

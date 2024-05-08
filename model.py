import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List,Literal,Optional
from rope import Rope,compute_sin_pos_emb
from cache import CacheLowFrequecyAllocateMem

"""
GPT(
  (transformer): ModuleDict(
    (word_emb): Embedding(6860, 16)
    (pos_emb): Embedding(32, 16)
    (emb_dropout): Dropout(p=0.1, inplace=False)
    (attn): ModuleList(
      (0): Block(
        (layer_norm4attn): LayerNorm()
        (attention): CausalSelfAttention(
          (wqkv): Linear(in_features=16, out_features=48, bias=False)
          (w_proj): Linear(in_features=16, out_features=16, bias=False)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (res_dropout): Dropout(p=0.1, inplace=False)
        )
        (layer_norm4ffn): MyLayerNorm()
        (ffn): MLP(
          (ffn_up): Linear(in_features=16, out_features=16, bias=False)
          (act): GELU(approximate='none')
          (ffn_down): Linear(in_features=16, out_features=16, bias=False)
          (dropout): MyDropout()
        )
      )
    )
    (layer_norm4last): LayerNorm()
  )
  (lm_head): Linear(in_features=16, out_features=6860, bias=False)
)
"""
# 启用异常检测
torch.autograd.detect_anomaly = True
@dataclass
class GPTConfig:
    """GPT 模型的配置"""

    block_size: int = 1024 # 序列长度T
    vocab_size: int = 50304 # GPT-2 的 vocab_size 是 50257，为了效率增加到最接近 64 的倍数
    n_layer: int = 12 # 层的个数
    n_head: int = 12 # 头的个数
    n_embd: int = 768 # emb特征的个数
    dropout: float = 0.0
    bias: bool = True # True: 在 Linear 和 LayerNorm 中有偏置，类似 GPT-2；False: 稍微好一点且更快
    norm_type : Literal['rmsnorm','layernorm'] = 'layernorm'
    ffn_multiply_scalar:int = 4
    glu_multiply_scalar: int = 8/3.0
    device: str = 'cpu'
    batch_size : int = 3
    decay_lr : bool = True
    residual :bool = True
    use_ffn : bool = True # 是否启用ffn层
    use_glu_only : bool = False #是否只用glu作为ffn
    use_mlp_only : bool = True # 是否只用mlp作为ffn

    use_moe : Optional[bool] = False #是否用moe作为ffn 优先级高于use_glu_only，use_mlp_only
    num_expert: Optional[int] = 4 # 使用几个专家
    num_expert_per_token : Optional[int] = 2 # 每个token最多使用几个专家
    expert_type: Literal['glu', 'mlp'] = 'glu'  # expert_type 只能取 'glu' 或 'mlp' 中的一个

    use_abs_int_pos_emb : bool = True # 是否使用位置编码
    use_abs_sin_pos_emb : bool = False # 是否使用sin绝对位置编码
    use_rope_pos_emb : bool = True # 是否使用旋转位置编码

    use_kvcache : bool = False #  是否使用 kvcache
    cache_max_batch_size : int = 32 #缓存本身最大batch_size 由于inference时会使用缓存，在使用缓存时，要求数据x的batch size <=cache_max_batch_size

    ignore_sepicial_target_id : int = -100 #微调时需要使用-100作为question token对应的target_id；预训练时，数据的targte_id中没有这个，不影响loss计算

    model_type :str = 'mygpt'

    # 转换成dict字典格式数据，为了满足lora框架需要
    def to_dict(self):
        return self.__dict__


class LayerNorm(nn.Module):
    """带有可选偏置的 LayerNorm。PyTorch 不支持简单的 bias=False。"""

    def __init__(self, ndim, bias):
        super().__init__()
        # 初始化权重参数
        self.weight = nn.Parameter(torch.ones(ndim))
        # 如果启用偏置，则初始化偏置参数
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None


    def forward(self, input):
        # 请在这里编写关键代码，使用 self.weight 和 self.bias 来实现 Layer Normalization
        normed_input = None
        """
        def layer_norm(input: Tensor,
               normalized_shape: Sequence[int],
               weight: Tensor | None = ...,
               bias: Tensor | None = ...,
               eps: float = ...) -> Tensor
        normalized_shape 必须写，否则会提示错误 
        TypeError: layer_norm() missing 1 required positional argument: 'normalized_shape'
        这个形状就和input的形状的最后一个维度一样，alpha权重也就是这里的weight的大小一样就行
        在每一个特征上做归一化，针对特征来做，先计算特征的均值，在计算特征的标准差，然后归一化
        归一化之后，再对一个特征维度都做缩放 * alpha + beta
        对每一个维度都做下列操作
        normed_feature[0] * weight[0] + beta[0]
        所以weight和beta的形状和特征的形状保持一样，都跟weight初始化时的形状一样就行好        
        """
        normed_input = F.layer_norm(input=input,normalized_shape=self.weight.shape, weight=self.weight,bias=self.bias)

        return normed_input
class MyLayerNorm(torch.nn.Module):
    """
    1. 参数初始化时,gamma=1,beta=0
    2. 一批数据，一条句子，所有token的feature都用共用同一个gamma和beta

        # 测试
        feature_dim = 5
    input_data = torch.randn(3, 4, feature_dim)  # batch_size=3, sequence_length=4, features=5
    layer_norm = MyLayerNorm(feature_dim)  # 对最后一维进行Layer Norm
    output = layer_norm(input_data)
    print(output)
    """
    def __init__(self, ndim, bias:bool=True,eps=1e-6):
        super(MyLayerNorm, self).__init__()
        features = ndim
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features)) if bias else None
        self.eps = eps
        self.bias = bias

    def forward(self, x):
        """
        # 对最后一个维度做计算求均值和标准差
        # keepdim=True,保留dim=-1这个维度，计算后还是(3,4,1) False的话就变成了(3,4)
        input_ids （3，4，5）
        mean (3,4)
        做相减时，5和4对应不上
        input_ids （3，4，5）
        mean (3,4,1)
        5和1对应上，1自动做广播，复制为 (3,4,5) 就可以做减法了
        对应不上时报错如下
        RuntimeError: The size of tensor a (5) must match the size of tensor b (4) at non-singleton dimension 2
        """
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normed_x = (x - mean) / (std + self.eps)
        if self.bias:
            return self.gamma * normed_x + self.beta
        else:
            return self.gamma * normed_x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class MyRMSNorm(nn.Module):
    def __init__(self,feature_dim,eps=1e-6):
        super(MyRMSNorm,self).__init__()
        self.gamma = nn.Parameter(data=torch.ones(feature_dim))
        self.eps = eps
    def forward(self,x:torch.Tensor):
        """
        总结下：
        1. 为什么我的实现会出现nan值loss的情况呢？
            因为，sqrt操作会导致数值下溢，需要提前添加eps，
            类似地，除法会导致数值上溢，除一个接近0的数值会导致这个严重的问题
        2. 我的实现还有一个明显的缺陷，我对x的shape有要求，
            要求是(b,t,h)的形状，但是其实没必要啊，只对最后一个
            维度做归一化就可以了
        3. 我还需要除以特征维度的长度，计算均值，其实没有必要
            mean()操作本身就可以实现有效均值
        """
        # b,t,h = input_ids.size()
        mean_square = torch.mean(torch.pow(x,2),dim=-1,keepdim=True)
        normed_x = x * torch.rsqrt(mean_square + self.eps)
        return self.gamma * normed_x

        square_x = torch.square(x)
        sum_square_x = square_x.sum(dim=2,keepdim=True)/h
        # normed_x = input_ids * torch.rsqrt(sum_square_x + self.eps)
        rms = torch.sqrt(sum_square_x + self.eps)
        # print(f'开根号前:{sum_square_x[0,0,0]}')
        # """
        # 看来是在开根号之前就已经是nan了，开根号后肯定就更惨了，还是nan
        # 所以，
        # """
        # rms = torch.sqrt(sum_square_x)
        # print(f'开根号后:{rms[0, 0, 0]}')
        normed_x = x/(rms + self.eps)
        return self.gamma * normed_x

class GLU(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        h = config.n_embd
        biger_h = int(config.glu_multiply_scalar * h)
        self.up_weight = nn.Linear(in_features=h,out_features=biger_h)
        self.down_weight = nn.Linear(in_features=biger_h,out_features=h)
        self.up_weight4act = nn.Linear(in_features=h,out_features=biger_h)
        self.act = nn.Sigmoid()
    def forward(self,x):
        up_x = self.up_weight(x)
        gate_elementwise_soft_dropout = self.act(self.up_weight4act(x))
        perserve_x = gate_elementwise_soft_dropout * up_x
        down_x = self.down_weight(perserve_x)
        return down_x
class Moe(nn.Module):
    def __init__(self,experts:List[nn.Module],gate:nn.Module,num_experts_per_token:int):
        super().__init__()
        self.experts = experts # 这里面就是一个一个的专家，专家输入h输出h，不改变大小
        self.gate = gate # 一个linear网络，输入h，输出num_experts大小的logits向量，之后可以用softmax归一化
        self.num_experts_per_token = num_experts_per_token # 这个token需要几个专家来处理

    def forward(self, x):
        """
        先假设 input_ids 是 (b, t, h) 大小的，不用管 batch_size, seq_len, 稍后再来考虑
        """
        batch_size, seq_len, token_dim = x.size()

        # 先使用 gate 计算出来，选择哪几个专家，专家的概率大小
        all_expert_prob_logits = self.gate(x)

        # 在分别把每个专家的输出做加权求和
        output = torch.zeros_like(x)

        for t in range(seq_len):
            # 对每个时间步应用相同的逻辑
            topk_prob_logits, topk_expert_indexs = torch.topk(input=all_expert_prob_logits[:, t, :],
                                                              k=self.num_experts_per_token, dim=-1)
            topk_prob = F.softmax(topk_prob_logits, dim=-1)

            for i in range(batch_size):
                for j in range(self.num_experts_per_token):
                    prob = topk_prob[i, j]
                    expert_index = topk_expert_indexs[i, j]
                    expert = self.experts[expert_index]
                    expert_output = expert(x[i, t, :])  # Unsqueeze to add a batch dimension
                    output[i, t, :] += prob * expert_output  # Squeeze to remove the added batch dimension

        return output

class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts_per_token: int):
        """
        初始化Mixture of Experts（MoE）层

        参数：
            experts (List[nn.Module]): 专家模块的列表，每个专家负责处理输入的一部分
            gate (nn.Module): 用于生成门控信号的模块
            num_experts_per_token (int): 每个 token 选择的专家数量
        """
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert_per_token = num_experts_per_token

        # 初始化专家选择次数的字典
        self.expert_selection_counts = {i: 0 for i in range(len(experts))}

    # 计算路由器z-loss
    def compute_router_z_loss(self, gate_logits):
        # 假设使用topk函数选择了前k个专家
        _, selected_experts = torch.topk(gate_logits, k=self.num_expert_per_token, dim=-1, largest=True)
        # 计算z_loss，这里使用的是简单的L1正则化
        z_loss = torch.sum((gate_logits - selected_experts).abs())
        return z_loss

    def compute_router_kl_loss(self,gate_logits,dtype):
        # ----------------------------------------做均衡-------------
        # 计算每个专家选择分布
        gate_prob = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(dtype)
        expert_selection_distribution = gate_prob.mean(dim=(0, 1))
        uniform_distribution = torch.ones_like(expert_selection_distribution) / len(expert_selection_distribution)
        # 计算KL散度
        kl_divergence_loss = F.kl_div(torch.log(expert_selection_distribution), uniform_distribution,
                                      reduction='batchmean')
        # ----------------------------------------做均衡-------------
        return kl_divergence_loss

    def forward(self, inputs: torch.Tensor):
        """
        前向传播过程

        参数：
            inputs (torch.Tensor): 输入张量，大小为 (batch_size, seq_len, dim)，包含了输入序列的特征表示

        返回：
            torch.Tensor: 输出张量，大小与输入张量相同，包含了经过专家处理后的序列特征表示
        """
        # 获取门控信号
        gate_logits = self.gate(inputs)
        # TODO 这部分还需要仔细思考下，到底该怎么实现负载均衡
        kl_divergence_loss = self.compute_router_kl_loss(gate_logits,dtype=inputs.dtype)

        # 使用topk函数获取每个token选择的专家及其权重
        weights, selected_experts = torch.topk(gate_logits, self.num_expert_per_token)
        # weights 大小：(batch_size, seq_len, num_experts_per_token)
        # selected_experts 大小：(batch_size, seq_len, num_experts_per_token)

        # 对权重进行softmax操作，得到每个专家的选择概率
        prob = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)
        # prob 大小：(batch_size, seq_len, num_experts_per_token)



        # 初始化结果张量
        results = torch.zeros_like(inputs)
        # results 大小：(batch_size, seq_len, dim)



        # 对于每个专家
        for i, expert in enumerate(self.experts):
            # 找到选择当前专家的 token 的位置
            sentence_idx4current_expert, token_idx4current_expert, prob_idx4current_expert = torch.where(
                selected_experts == i)
            self.expert_selection_counts[i] += sentence_idx4current_expert.size(0)
            """
            为什么用where操作？(为什么从expert角度出发？)
            上述两个问题是一个问题，从专家角度出发，把需要专家处理的token都处理好
            相比从token角度出发，效率更高，减少循环
            sentence_idx4current_expert, 
            token_idx4current_expert, 
            prob_idx4current_expert
            三个大小都是 (num_selected_experts_for_expert_i,)
            从大到小定位到需要被专家i处理的token的位置，哪个句子，哪个token，对应到哪个概率
            """

            # 获取对应的选择概率
            prob4expert = prob[sentence_idx4current_expert, token_idx4current_expert, prob_idx4current_expert]
            # prob4expert 大小：(num_selected_experts_for_expert_i,)

            # 获取对应的输入
            x4expert = inputs[sentence_idx4current_expert, token_idx4current_expert, :]
            # x4expert 大小：(num_selected_experts_for_expert_i, dim)

            # 获取当前专家的输出
            expert_output = expert(x4expert)
            # expert_output 大小：(num_selected_experts_for_expert_i, dim)
            """
            为什么要做prob4expert.unsqueeze(-1)？
                从 (num_selected_experts_for_expert_i,)
                转 (num_selected_experts_for_expert_i,1)
                就可以和 expert_output做对应位置的 element-wise乘法
                expert_output=(num_selected_experts_for_expert_i,dim)
            """
            # 将概率乘以专家输出，得到加权输出
            prob_multiply_expert_output = prob4expert.unsqueeze(-1) * expert_output
            # prob_multiply_expert_output 大小：(num_selected_experts_for_expert_i, dim)

            # 将加权输出加到结果张量中相应的位置
            results[sentence_idx4current_expert, token_idx4current_expert, :] += prob_multiply_expert_output

        return results, kl_divergence_loss

class CausalSelfAttention(nn.Module):
    """自注意力机制模块，实现了因果自注意力，确保模型在预测时只能使用之前的上下文信息，而不能使用未来的信息。

    Args:
        config (obj): 包含模型配置的对象，如嵌入维度、头部数、dropout概率等。
    """
    def __init__(self, config:GPTConfig):
        """
        如果不对父类进行初始化，就会提示
        AttributeError: cannot assign module before Module.__init__() call
        """
        super().__init__()

        # 确保嵌入维度可以被头部数整除，否则无法平均分配到各个头部
        assert config.n_embd % config.n_head == 0, '特征维度必须能整除头的个数n_head！'

        self.config = config
        h = config.n_embd
        # 通过一个线性层将输入投影到三个不同的向量，分别用于key、query和value
        # self.wqkv = nn.Linear(in_features=h,out_features=3*h,bias=config.bias)
        self.q_proj = nn.Linear(in_features=h,out_features=h,bias=config.bias)
        self.k_proj = nn.Linear(in_features=h, out_features=h, bias=config.bias)
        self.v_proj = nn.Linear(in_features=h, out_features=h, bias=config.bias)

        # 用于输出的线性层，将注意力机制的输出映射回原维度
        self.w_proj = nn.Linear(in_features=h,out_features=h,bias=config.bias)
        # 注意力机制中使用的dropout，用于防止过拟合
        self.attn_dropout = nn.Dropout(p=config.dropout)
        # 残差连接中使用的dropout
        self.res_dropout = nn.Dropout(p=config.dropout)
        # 头部的数量，用于分割key、query和value
        self.nh = config.n_head
        # 头部的特征维度
        self.hs = int(config.n_embd / config.n_head)

        # 嵌入维度，即模型输入的特征维度
        self.h = config.n_embd

        # dropout概率，用于正则化
        self.dropout_p = config.dropout

        # 检查PyTorch版本是否支持Flash Attention，这是一种更快的自注意力计算方式
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # T = config.block_size
            # mask = torch.tril(torch.ones(T,T))
            # mask = mask.view(1,1,T,T)
            # mask不需要grad，作为模型的一部分，保存和加载时，需要把mask保存到模型中
            # 使用的时候，直接使用self.mask 就可以获取到
            # self.register_buffer('mask',mask)

        if self.config.use_rope_pos_emb:
            self.roper = Rope(base_value=10000)

        if config.use_kvcache:
            self.k_cache = CacheLowFrequecyAllocateMem(config.cache_max_batch_size,config.block_size,self.nh,self.hs)
            self.v_cache = CacheLowFrequecyAllocateMem(config.cache_max_batch_size,config.block_size,self.nh,self.hs)

    def forward(self, input_ids:torch.Tensor, attention_mask=None,*args,**kwargs):
        if kwargs.get('attention_mask',None) is not None:
            attention_mask = attention_mask

        # 输入张量x的维度为(批次大小 B, 序列长度 T, 嵌入维度 C)
        B,T,h = input_ids.size()

        """
            使用自注意力机制计算注意力权重，并将输入x分别投影为query(q)、key(k)和value(v)
            使用split函数将输出张量沿着特定维度分割成三个张量，提示使用split函数
            使用view函数对张量进行形状重组，提示使用view函数
            使用transpose函数对张量进行维度转置，提示使用transpose函数
            split 函数可以把3h 切割成3个h
            view函数可以自动变换维度 ，把一个h 变成 (num_head ,head_size)只要给出其中一个尺寸，另一个就可以自动算出来
            transpose函数把需要交换的维度写进去就可以了 B,nh,T,hs -> B,T,nh,hs
            就只需要写交换T,nh两个维度 transpose(1,2) 不要写成transpose(2,1)
        """
        # 使用自注意力机制计算注意力权重，并将输入x分别投影为query(q)、key(k)和value(v)
        # xqkv:torch.Tensor = self.wqkv(input_ids)
        # xqkv (B,T,3h)
        # q, k, v = xqkv.split(split_size=h, dim=-1)
        # 改成这样是为了方便使用peft包的lora，这个包要求有q_proj，v_proj这样的参数变量
        q = self.q_proj(input_ids)
        k = self.k_proj(input_ids)
        v = self.v_proj(input_ids)


        # q,k,v 分别都是 (B,T,h) 大小
        # 将key(k)重新组织为形状为(B, nh, T, hs)，其中B是batch size，T是时间步长，nh是头数，hs是每个头的特征维度
        q = q.view(B,T,self.nh,self.hs).transpose(1,2)
        # 将query(q)重新组织为形状为(B, nh, T, hs)，其中B是batch size，T是时间步长，nh是头数，hs是每个头的特征维度
        k = k.view(B, T, self.nh, self.hs).transpose(1, 2)
        # 将value(v)重新组织为形状为(B, nh, T, hs)，其中B是batch size，T是时间步长，nh是头数，hs是每个头的特征维度
        v = v.view(B, T, self.nh, self.hs).transpose(1, 2)
        # 控制是否使用旋转位置编码
        if self.config.use_rope_pos_emb:
            q = self.roper.apply_rotatory_emb(q)
            k = self.roper.apply_rotatory_emb(k)
        # -----------------------------旋转之后，缓存起来--------------
        # 为什么是旋转之后缓存起来呢？
        """
        因为，我们只需要直接从缓存中取出来，计算attention就可以了，喂给attention的qkv就已经是旋转过的
        不然，难道每次从cache取出来之后再进行旋转重复计算，不就是冗余计算了吗，没有必要
        """

                    
        # 实现cachekv，要记住就是主要针对inference场景就可以，每次都实现单独一个token的解码
        # 1. 把token的kv存到cachekv中
        # 2. 把从cachekv中把历史的kv和最新的kv一起取出来
        """
        是否需要增加模型的training还是inference的状态判断呢？
            其实，并不需要，我们只需要根据use_kvcache是否为true判断即可
            在训练时，设定use_kvcache=False
            在inference是，设定 use_kvcache=True就可以了 
            在mistral的实现中，是通过forward函数中传入了一个cache，通过cache是否为None来决定是否使用cache
            还进一步区分了prefile时针对prompt文本使用chunk+cache的方式做cache初始化
            单个token decode阶段则使用cache进行attn计算
            ----
            在llama中则是通过startpos来控制是否为0来控制缓存的使用，训练时固定为0就可以了
        """
        if self.config.use_kvcache and not self.training:
            # 将value(v)重新组织为形状为(B, T,nh, hs)，其中B是batch size，T是时间步长，nh是头数，hs是每个头的特征维度
            k = k.transpose(1,2)
            v = v.transpose(1, 2)
            self.k_cache.save_to_cache(k)
            self.v_cache.save_to_cache(v)
            k = self.k_cache.get_all_from_cache()
            v = self.v_cache.get_all_from_cache()
            # 将value(v)重新组织为形状为(B, nh,T, hs)，其中B是batch size，T是时间步长，nh是头数，hs是每个头的特征维度
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)


        # ----------------------------------------------------------


        # 重新形状和转置以便进行注意力计算，将头部前移以成为批次维度
        # causal self-attention; Self-attend: (B, nh, T, hs) input_ids (B, nh, hs, T) -> (B, nh, T, T)
        # 因果自注意力计算
        """
        这里的感悟，目前我们基本就是使用flash attention，所以基本就是直接用torch.nn.functional.scaled_dot_product_attention
        这样就不需要自己写了，所以其实除非是类似GAU这种完全改进了，否则都不需要自己写，直接用底层实现就可以了        
        
        使用PyTorch内置的Flash Attention进行高效的注意力计算
        why mask = None？ we can use is_casual=True
        如果is_causal=True,默认就是下三角为1的mask了，不要再设置attention_mask，否则会报错
        用户如果传入了attention mask,就不要在设置is_casual=True了，传入用户给定的attention mask就好了
        self.training 是nn.Module中的属性
        """
        if self.flash:
            if attention_mask :
                output = torch.nn.functional.scaled_dot_product_attention(
                    query=q, key=k, value=v, attn_mask=attention_mask,is_causal=False,
                    dropout_p=self.dropout_p if self.training else 0)
            else:
                output = torch.nn.functional.scaled_dot_product_attention(
                    query=q, key=k, value=v, attn_mask=None, is_causal=True,
                    dropout_p=self.dropout_p if self.training else 0)
        else:
            # 如果不支持Flash Attention，则手动实现注意力计算
            # 使用query(q)与key(k)的转置进行点积操作，提示使用@符号
            attn = q @ k.transpose(-2,-1) # (B,nh,T,T)
            # 乘以缩放因子，该缩放因子是倒数为key(k)的特征维度的平方根，提示使用math.sqrt函数
            scalar = 1.0/math.sqrt(self.hs)
            attn = scalar * attn
            # 使用masked_fill函数将注意力权重中需要屏蔽的位置替换为负无穷，提示使用masked_fill函数
            # 如果传入了 attention_mask就用传入的，如果没有就用默认的，下三角为1的causal mask
            # mask = self.mask[:, :, :T, :T] == 0
            # 训练和全量推理时，都会传入下三角的attention mask
            # 此时，才需要应用mask，增量推理时，不需要填入mask
            if attention_mask is not None:
                mask = attention_mask == 0
                try:
                    attn = torch.masked_fill(input=attn, mask=mask, value=-1e9)
                except RuntimeError as e:
                    print("An error occurred while masking the tensor:", e)
                    print(f'input_ids : \n {input_ids.shape}\n attn:\n {attn.shape} \n mask:\n{mask}\n attention_mask:\n{attention_mask}\n')
            # 对注意力权重进行softmax操作，提示使用F.softmax函数
            attn = F.softmax(input=attn,dim=-1)
            # 对注意力权重进行dropout操作，提示使用attn_dropout函数
            attn = self.attn_dropout(attn) # attn (B,nh,T,T)

            if attention_mask is not None:
                attn = attn * attention_mask
            else:
                # 此时，没有attention_mask代表是推理
                # 如果没有用kvcache，跟训练时一样，采用的全量推理，而不是增量单token推理
                if not self.config.use_kvcache:
                    mask = torch.tril(torch.ones(T, T))
                    mask = mask.view(1, 1, T, T)
                    attn = attn * mask
                else:
                    # 用了缓存，则不需要再乘任何mask信息了，应该不会推理出pad_id，推理出eos就结束了
                    pass
            # 将注意力权重与value(v)进行加权求和操作，得到输出y，提示使用@符号
            # (B, nh, T, T) input_ids (B, nh, T, hs) -> (B, nh, T, hs)
            output = attn @ v


        #  why need contiguous()?
        """
        RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        因为，view操作调整张量形状时，是改变的张量维度的大小和内存地址的步长实现的
        隐含要求就是要求张量的内存是连续的，
        否则，无法通过仅仅改变步长就改变了形状
        因为，output不是内存连续的，所以执行view时候报错提示，至少一个维度 spans across 两个连续空间
        view操作是高效的，所以可以使用contiguous把空间变连续
        
        transpose不要求连续，因为transpose只是交换索引，不需要使用内存地址的步长
        """
        # 将所有头部的输出重新组装，用-3，-2，而不是用2，3是确保
        o = output.transpose(-3,-2).contiguous().view(B,T,h)

        # 通过输出投影层将注意力机制的输出映射回原维度
        new_x = self.w_proj(o) # new_x (B,T, h)
        return new_x


class MyDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, input):
        if self.training:
            mask = torch.rand_like(input) > self.p
            scale = 1 / (1 - self.p)
            return input * mask * scale
        else:
            return input

class MLP(nn.Module):
    """多层感知机模块"""

    def __init__(self, config:GPTConfig):
        super().__init__()
        h = config.n_embd
        h4 = config.ffn_multiply_scalar * config.n_embd
        # 全连接层，输入维度为 config.n_embd，输出维度为 4 * config.n_embd
        self.ffn_up = nn.Linear(in_features=h,out_features=h4,bias=config.bias)
        # GELU 激活函数，非线性变换
        self.act = nn.GELU()
        # 全连接层，输入维度为 4 * config.n_embd，输出维度为 config.n_embd
        self.ffn_down = nn.Linear(in_features=h4,out_features=h,bias=config.bias)
        # Dropout 层，防止过拟合，随机失活一部分神经元
        # TODO 到底参数设置多少比较好呢？
        # self.dropout = nn.Dropout(p=config.dropout)
        self.dropout = MyDropout(p=config.dropout)

    def forward(self, x:torch.Tensor):
        # 神经网络前向传播
        # 全连接层变换
        up_h = self.ffn_up(x)
        # GELU 激活函数
        up_h_act = self.act(up_h)
        # 全连接层变换
        down_h = self.ffn_down(up_h_act)
        # Dropout 随机失活
        down_h_dropped = self.dropout(down_h)
        # assert x.shape == down_h_dropped.shape, f'ffn层输出tensor的shape is {down_h_dropped.shape} input shape {x.shape}'
        return down_h_dropped

class Block(nn.Module):
    """Transformer 块"""
    def __init__(self, config:GPTConfig):
    # 调用父类构造函数初始化模块
        super(Block,self).__init__()

        if config.norm_type == 'layernorm':
            norm = LayerNorm(config.n_embd, bias=config.bias)
        elif config.norm_type == 'rmsnorm':
            norm = MyRMSNorm(config.n_embd)
        else:
            raise ValueError('norm_type should be in layernorm,rmsnorm!!')
    # 实例化 LayerNorm 层，用于对输入进行 Layer Normalization
        self.layer_norm4attn = norm
    # 实例化 CausalSelfAttention 注意力机制
        self.attention = CausalSelfAttention(config=config)
    # 实例化 LayerNorm 层，用于对注意力机制输出进行 Layer Normalization
        self.layer_norm4ffn = norm
    # 实例化 MLP 多层感知机模块
        if config.use_ffn:
            if config.use_moe:
                assert config.num_expert_per_token ,'num_expert_per_token is None'
                assert config.num_expert, 'num_expert is None'
                assert config.num_expert>=config.num_expert_per_token,'num_expert_per_token must <= num_expert'

                if config.expert_type == 'glu':
                    experts = [GLU(config) for _ in range(config.num_expert)]
                elif config.expert_type == 'mlp':
                    experts = [MLP(config) for _ in range(config.num_expert)]
                gate = nn.Linear(config.n_embd,out_features=config.num_expert_per_token)
                self.ffn = MoeLayer(experts,gate,config.num_expert_per_token)
            elif config.use_mlp_only:
                self.ffn = MLP(config=config)
            elif config.use_glu_only:
                self.ffn = GLU(config)
        else:
            self.ffn = None

        self.config = config


    def forward(self, input_ids, attention_mask):
        """
        原则1. 在使用x之前先做norm
        原则2. 残茶链接相加路径上，保持残差干净
        """
        # layer norm for attention
        normed_x = self.layer_norm4attn(input_ids)
        # 输入张量通过第一个注意力层
        attention_x = self.attention(input_ids=normed_x,attention_mask=attention_mask)
        # residual connection # 对第一个注意力层的输出进行 Layer Normalization，并与原始输入相加
        # h = attention_x + normed_x 这是错误的姿势
        if self.config.residual: #用来做实验，是否使用残差
            h = input_ids + attention_x # 这是正确的姿势
        else:
            h = attention_x
        # layyer norm for ffn
        normed_h = self.layer_norm4ffn(h)
        # 输入相加后再通过 MLP 多层感知机模块
        if self.config.use_moe:
            ffn_h,expert_gate_loss = self.ffn(normed_h)
            # TODO 可能还要一层一层地返回到最后的model
        else:
            ffn_h = self.ffn(normed_h)
        # residual modual
        # attn_output = ffn_h + normed_h #这是错误的姿势
        if self.config.residual:
            attn_output = ffn_h + h # 这是正确的姿势
        else:
            attn_output = ffn_h
        # 返回最终结果
        return attn_output


class GPT(nn.Module):
    """GPT 模型"""

    def __init__(self, config:GPTConfig):
        # 调用父类构造函数初始化模块
        super().__init__()
        # 断言确保词汇表大小和块大小已经设置
        self.config = config
        """
                    nn.Embedding也更加高效，因为它可以通过索引直接查找嵌入向量，而不需要进行矩阵乘法运算。
                    看来，embedding是做了高效处理，的确是，采用linear的话，还需要和一个onehot矩阵做乘法运算
                    embedding这种利用索引直接查找出权重向量的做法就不需要做矩阵乘法了
                    """
        # 判断是否使用ffn层
        myBlock = Block if config.use_ffn else CausalSelfAttention
        if config.use_abs_sin_pos_emb:
            base_value = 10000
            seq_len = config.block_size
            self.sin_pos_emb = compute_sin_pos_emb(seq_len, config.n_embd, base_value=base_value)

        self.transformer = nn.ModuleDict(dict(
            # 词嵌入层，将输入标记映射到嵌入维度的向量空间
            # TODO 错误姿势
            # word_emb = nn.Linear(in_features=config.vocab_size,out_features=config.n_embd)
            word_emb = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd),
            # 位置嵌入层，将输入位置编码映射到嵌入维度的向量空间
            # pos_emb = nn.Linear(in_features=config.block_size,out_features=config.n_embd)
            pos_emb = nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd) if config.use_abs_int_pos_emb else None,
            # Dropout 层，用于防止过拟合
            emb_dropout = nn.Dropout(p=config.dropout),
            # 多头自注意力机制的层列表
            attn=nn.ModuleList([myBlock(config) for _ in range(config.n_layer)]),
            # 输出的 LayerNorm 层，用于对最终输出进行归一化
            layer_norm4last = LayerNorm(config.n_embd,bias=config.bias),
        ))

        # 线性层，用于预测下一个标记
        self.lm_head = nn.Linear(in_features=config.n_embd,out_features=config.vocab_size,bias=config.bias)
        # 使用权重绑定技术，将词嵌入层的权重和线性层的权重绑定在一起
        self.transformer.word_emb.weight = self.lm_head.weight
        """
        原来如此，解码时候所用词编码和训练时所用的词编码，时一样的，想想也是合理的，
        理应二者采用都是一模一样的，但是意思这种权重绑定技术，在未来的版本中将会被去掉        
        """
        # 初始化所有权重
        # 根据 GPT-2 论文对残差投影进行特殊的缩放初始化
        for param_name, param in self.named_parameters():
            if param_name.endswith('w_proj.weight'):
                torch.nn.init.normal_(param,mean=0,std=0.02/math.sqrt(2*config.n_layer))
        """
        当然，这种初始化的目的肯定是为了训练的稳定性，层越大，梯度消失越严重，所以要控制初始化
        但是，我不明白的是，两点
        第一，为啥要针对proj矩阵呢？
        第二，要标准差越来越小呢
        """
        # 报告参数数量
        n_params = self.get_num_params()
        print(f'参数量：{n_params}')

    def get_num_params(self, non_embedding=True):
        param_num = 0
        """返回模型中的参数数量。默认情况下不包括嵌入层的参数。"""
        # 统计模型中所有参数的数量
        param_num = 0
        for param_name,param in self.named_parameters():
            param_num += param.numel()

        # 如果 non_embedding 为 True，则不包括嵌入层的参数数量
        if non_embedding:
            param_num -= self.transformer.word_emb.weight.numel()
        return param_num

    def _init_weights(self, module):
        """初始化权重"""
        """
        如何判断当前模块是什么模块？
        就需要利用isinstance(module,nn.Linear)，isinstance(module,nn.Embedding)
        这种方式进行判断
        """
        # 如果当前模块是线性层
        if isinstance(module,nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为0.02
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            if module.bias is not None:
                # module.bias = 0 TODO 错误姿势
                """
                不能直接使用 module.bias = 0
                而是要采用 torch.nn.init.zeros_(module.bias)
                """
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,0,std=0.02)
        # 如果模块有偏置，则将偏置初始化为零
        # 如果当前模块是嵌入层
        # 使用正态分布初始化权重，均值为0，标准差为0.02


    def forward(self, input_ids=None, target_ids=None, attention_mask:torch.Tensor=None, *args, **kwargs):
        """前向传播"""
        logits, loss = None,None
        """
        看来是利用idx来判断设备信息的，决定采用cude还是cpu来计算，idx的device信息决定了在什么设备上进行计算
        """
        # 确定输入张量的设备
        device = input_ids.device
        # 获取输入张量的批次大小和序列长度
        # b,t,h = input_ids.size()
        b, t = input_ids.size()
        """
        input_ids.size() input_ids.shape有什么区别呢？
        input_ids 只需记录batch_size，seq_len 来记录token的索引就可以，h是通过embeding查找出来的emb
        """
        # 断言序列长度不超过块大小
        # assert t<=self.config.block_size,'序列长度不能超过训练时blocksize大小'
        # 生成位置张量，形状为 (t)，用于表示输入序列中每个位置的位置编码
        pos = torch.arange(0,t)
        # 前向传播 GPT 模型本身
        # 获取令牌嵌入和位置嵌入，并进行相加
        word_emb = self.transformer.word_emb(input_ids)

        if self.config.use_abs_int_pos_emb:
            pos_emb = self.transformer.pos_emb(pos)
            x  = word_emb + pos_emb
        elif self.config.use_abs_sin_pos_emb:
            x = word_emb + self.sin_pos_emb
        else:
            x = word_emb


        # 将令牌和位置嵌入输入到模型中，依次经过每个 Transformer 块
        for attn_layer in self.transformer.attn:
            x = attn_layer(input_ids=x,attention_mask=attention_mask)

        # 对最终输出进行 Layer Normalization
        x = self.transformer.layer_norm4last(x)
        self.last_hidden_state = x

        if target_ids is not None:
        # 如果给定了一些目标，还计算损失
        # 获取最终输出并通过线性层生成预测 logits
            logits = self.lm_head(x)
        # 将 logits 重塑为二维张量
            logits_for_loss = logits.view(-1,logits.size(-1))
        # 将目标重塑为一维张量
            target_ids = target_ids.view(-1)

        # 计算交叉熵损失
            loss = F.cross_entropy(input=logits_for_loss, target=target_ids, ignore_index=self.config.ignore_sepicial_target_id)
        else:
        # 获取最后一个位置的输出并通过线性层生成预测 logits
        # 注意：使用列表 [-1] 以保留时间维度
        # 推理时的小优化：仅在最后一个位置上前向传播 lm_head
        #     logits = self.lm_head(input_ids[:,-1,:]) # TODO 这是错误的姿势
            need_infer_x = x[:,-1, :]
            logits = self.lm_head(need_infer_x)
        # 损失设置为 None
            loss = None
            """
            为什么不是用self.lm_head(input_ids[:,-1,:]) 
            而是用 self.lm_head(input_ids[:[-1],:]) ？？？？                        
            """
        """
        为什么要这么做呢？
        没有targets时，也就不需要计算全部的loss，而只是做一次前向推理，但是为什么只对序列中最后一个token做一次前向计算呢？
        仅仅只计算最后一个token的logits，是为什么呢？
        """
        return logits,loss

    def crop_block_size(self, block_size):
        """模型修剪以减小块大小"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """从预训练模型加载 GPT 模型"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # 默认为空字典
        # 只有 dropout 可以被覆盖，有关更多信息，请参见下面的注释
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("从预训练 GPT 加载权重：%s" % model_type)

        # n_layer、n_head 和 n_embd 是根据 model_type 确定的
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M 参数
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M 参数
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M 参数
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M 参数
        }[model_type]
        print("强制 vocab_size=50257，block_size=1024，bias=True")
        config_args['vocab_size'] = 50257 # GPT 模型检查点的 vocab_size 总是 50257
        config_args['block_size'] = 1024 # GPT 模型检查点的 block_size 总是 1024
        config_args['bias'] = True # GPT 模型检查点的 bias 总是 True
        # 如果需要，我们可以覆盖 dropout 率
        if 'dropout' in override_args:
            print(f"覆盖 dropout 率为 {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # 创建一个从头初始化的 minGPT 模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 丢弃此遮罩/缓冲区，不是参数

        # 初始化 huggingface/transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制并确保所有参数在名称和形状上对齐并匹配
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略这些，只是一个缓冲区
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 同样，只是遮罩（缓冲区）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上，openai 检查点使用了 "Conv1D" 模块，但我们只想使用普通的 Linear
        # 这意味着在导入它们时，我们必须转置这些权重
        assert len(sd_keys_hf) == len(sd_keys), f"不匹配的键：{len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对 Conv1D 权重进行特殊处理，我们需要转置它们
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay=1e-1, learning_rate=6e-4, betas=(0.9,0.95), device_type='cpu'):

        """配置优化器"""
        """
        这里所做的优化，主要就是，找出需要做权重衰减的参数，进行参数更新
        如果不做的话，优化器肯定也是知道哪些参数需要梯度
        只对需要梯度的参数做参数更新
        但是优化器应该默认会对所有参数都做权重衰减，即便是bias也会做的，但其实没必要，
        仅仅对维度大于2的权重做权重衰减才有必要，这么做是因为什么？
        可以节省计算资源？
        """
        optimizer = None
        # 开始时有所有候选参数
        # 过滤出不需要梯度的参数
        # 创建优化器组。任何 2D 的参数都将被进行权重衰减，否则不会。
        # 也就是说，所有在矩阵乘法中的权重 + 嵌入层会进行衰减，所有偏置和层归一化不会。
        decay_params = []
        no_decay_params = []
        for param_name,param in self.named_parameters():
            # if param.shape >=2: # TODO 错误的姿势
            if param.requires_grad:
                if param.dim() >= 2  : #需要梯度并且有两个维度以上的矩阵才需要更新参数，其他是不需要更新的
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        optim_groups = [
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':no_decay_params,'weight_decay':0.0}
        ]

        # 打印衰减参数张量数量和非衰减参数张量数量
        print(f'need decay params number is {sum([param.numel() for param in decay_params])}'
         f'\n no need decay params number is {sum([param.numel() for param in no_decay_params])}')
        # 打印是否使用融合 AdamW
        """
        什么是融合，为何需要融合？
        
        """
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # 创建 AdamW 优化器并在可用时使用融合版本
        optimizer = torch.optim.AdamW(params=optim_groups, lr=learning_rate, betas=betas, weight_decay=weight_decay,**extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """估算模型 FLOPS 利用率（MFU），以 A100 bfloat16 峰值 FLOPS 为单位"""
        # 首先估算每个前向传播的 FLOPS
        n_params = self.get_num_params()
        n_multadds_per_param = 6  # 论文中提到的每个参数的 FLOPS（乘法加法）
        flops_per_fwd = n_params * n_multadds_per_param
        # 然后估算在给定的时间步长内可以运行多少次前向传播
        max_fwd_per_sec = 1 / (fwdbwd_per_iter * dt)
        max_fwd_per_timestep = max_fwd_per_sec * dt
        # 最后，计算利用率
        mfu = flops_per_fwd * max_fwd_per_timestep
        return mfu

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": None,  # needs to be passed to make Keras.layer.__call__ happy
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "use_cache": use_cache,
        }



if __name__ == '__main__':

    # 测试 GPT 模型
    model = GPT.from_pretrained('gpt2')
    print(model)

    # 估计 MFU
    fwdbwd_per_iter = 10 # 每次前向传播/反向传播迭代的数量
    dt = 5e-3 # 每次迭代的时间步长，单位：秒
    mfu = model.estimate_mfu(fwdbwd_per_iter, dt)
    print(f"模型 FLOPS 利用率（MFU）：{mfu:.2f} AFLOPS (A100 bfloat16)")

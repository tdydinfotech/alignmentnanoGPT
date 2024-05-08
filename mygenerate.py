import torch
import torch.nn.functional as F
from model import GPT,GPTConfig

import os
import pickle
import tiktoken
unk_token_id = 3

def load_model_from_dir(out_dir,device='cpu'):
    if not out_dir :
        out_dir = 'pretrain_checkpoint'
    # 从特定目录中的模型中初始化
    ckpt_path = os.path.join(out_dir, 'myckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    # gptconf = GPTConfig(**checkpoint['config'])
    config:GPTConfig = checkpoint['config']
    config.use_kvcache = True
    # config.block_size = 5
    model = GPT(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model,config

def load_tokenizer(tokenizer_dir=None):
    """
    加载编码器和解码器函数。

    参数:
    tokenizer_dir (str, optional): 包含 meta.pkl 文件的目录路径。如果不提供,则默认为"pretrain_data"。

    返回:
    encode (function): 将字符串编码为数字列表的函数。
    decode (function): 将数字列表解码为字符串的函数。
    """
    if not tokenizer_dir:
        # 如果未提供 tokenizer_dir,则尝试从当前脚本所在目录中找到 meta.pkl
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_dir = os.path.join(script_dir, 'data', 'pretrain_data')

    meta_path = os.path.join(tokenizer_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)

    if load_meta:
        print(f"从 {meta_path} 加载元数据...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO: 使这更通用以适用于任意的编码器/解码器方案
        stoi, itos = meta['stoi'], meta['itos']

        # 如果 meta 中没有 'unk_token' 键,则使用一个默认的未知字符
        unk_token = meta.get('unk_token', '<unk>')

        unk_id = stoi.get(unk_token, unk_token_id)

        encode = lambda s: [stoi.get(c, unk_id) for c in s]
        decode = lambda l: ''.join([itos[i] if i in itos else unk_token for i in l])
    else:
        # 假设默认为 GPT-2 编码
        print("未找到 meta.pkl 文件,假设为 GPT-2 编码...")
        enc = tiktoken.get_encoding("gpt2")
        unk_token = enc.special_tokens["unk_token"]
        unk_id = enc.encode(unk_token)[0]

        encode = lambda s: enc.encode(s, allowed_special={unk_token})
        decode = lambda l: enc.decode(l, skip_special_tokens=False)

    return encode, decode

# @torch.no_grad()
def generate(model, config:GPTConfig, input_ids, max_new_tokens, temperature=1.0, top_k=None):
    """
    采用一个索引序列 input_ids（形状为 (b,t) 的 LongTensor），完成序列 max_new_tokens 次，
    每次将预测结果反馈回模型。在此操作中最好确保处于 model.eval() 模式下。
    """
    model.eval()

    # -------全量解码，cache prefill阶段------------

    # idx就直接是prompt部分的所有token转换成token_ids
    # 如果序列上下文过长，必须将其裁剪为 block_size
    # input_ids_clip_length = input_ids if input_ids.size(1) <= config.block_size else input_ids[:, -config.block_size:]

    # prompt_token_ids = input_ids_clip_length
    prompt_token_ids = input_ids
    pad_mask = torch.ones_like(input_ids)
    from utils.basic_utils import convert_1d_pad_mask_to_2d_attention_mask
    attention_mask = convert_1d_pad_mask_to_2d_attention_mask(pad_mask)

    # 前向传播模型以获取序列中索引的对数概率
    with torch.no_grad():
        last_token_logits, _ = model(input_ids=prompt_token_ids,attention_mask=attention_mask)
    last_token_logits = last_token_logits[:, :] / temperature
    # 应用 softmax 将对数概率转换为（归一化的）概率
    probs = F.softmax(last_token_logits, dim=-1)
    # 从分布中进行采样
    next_token_id = torch.multinomial(probs, num_samples=1)

    # --------增量解码，逐个token生成回复阶段--------------

    # 这里是，单个token的增量解码过程，逐个token生成，每次输入的token_id都是当前上一次生成的token_id，迭代进行
    current_input_token_id = next_token_id
    for _ in range(max_new_tokens):
        """
         其实，在llm场景下，每次都是传入一个token！！
         因为训练时是针对固定长度的样本进行的训练，所以这里就涉及到了长度外推
         当然也有streamingllm这种，就不用担心cachekv过长，可以动态地只保留临近的llm
         同时保留作为attention sink的第一token，这样cache就不需要很大了，理论上就可以无限长
        """
        # 前向传播模型以获取序列中索引的对数概率
        with torch.no_grad():
            last_token_logits, _ = model(current_input_token_id)
        # 提取最后一步的对数概率并按所需温度进行缩放
        # 在模型中已经做了，就没必要在做一次了，冗余啊
        last_token_logits = last_token_logits[:, :] / temperature
        # 可选地将对数概率裁剪为仅包含前 k 个选项
        if top_k is not None:
            v, _ = torch.topk(last_token_logits, min(top_k, last_token_logits.size(-1)))
            last_token_logits[last_token_logits < v[:, [-1]]] = -float('inf')
        # 应用 softmax 将对数概率转换为（归一化的）概率
        probs = F.softmax(last_token_logits, dim=-1)
        # 从分布中进行采样
        idx_next = torch.multinomial(probs, num_samples=1)
        # 将采样的索引附加到运行序列并继续
        input_ids = torch.cat((input_ids, idx_next), dim=1)
        # 直接更新为下一个token的id，每次喂给模型进行预测时只需要当前最新的token id
        current_input_token_id = idx_next
    # 生成结束后，需要的是把缓存清理掉，不然下一批次的样本会复用这批数据的cache

    return input_ids

from torch.nn.utils.rnn import pad_sequence
from typing import Callable

def convert_prompts_to_token_ids_with_batch(prompt_list,pad_value=5,encode_func:Callable=None):
    # 对每个 prompt 进行编码，并将它们存储为 PyTorch tensor 列表
    encoded_texts = [torch.tensor(encode_func(prompt)) for prompt in prompt_list]
    # 使用 pad_sequence 将这些 tensor 填充到相同的长度
    padded_input_ids = pad_sequence(encoded_texts, batch_first=True, padding_value=pad_value)
    return padded_input_ids

if __name__ == '__main__':
    device = 'cpu'

    # model_dir = 'pretrain_checkpoint'
    model_dir = 'sft_merge_checkpoint'

    dataset = 'data/pretrain_data'
    # tokenizer_dir = os.path.join('data', dataset)

    model,config = load_model_from_dir(model_dir)

    encode,decode = load_tokenizer(tokenizer_dir=dataset)

    prompt_list = [
        "飞燕真的可以飞吗？可以游泳吗？飞燕真的可以飞吗？可以游泳吗？",
        "诗歌是什么？你说呢, 天空不做美，飞燕，飞鸿，明月清风照我心，天河留下不同",
        "梅花",

    ]
    # kvcache的batch_size大小
    prompt_list =  prompt_list[:config.batch_size]

    # 基于huggingface的tokenizer的解码器实现
    # from mytokenize.load_my_tokenizer import get_my_tokenizer,replace_pad_token
    # tokenizer = get_my_tokenizer()
    # res = tokenizer(prompt_list, max_length=30, padding='max_length', return_tensors='pt')
    # input_ids = res['input_ids']
    # attention_mask = res['attention_mask']

    input_ids = convert_prompts_to_token_ids_with_batch(prompt_list=prompt_list,encode_func=encode)

    print(f'input_ids shape is {input_ids.shape}')
    print(
        f"input_ids with batch \n {input_ids}"
    )

    num_samples = 3
    max_new_tokens = 100
    temperature = 0.95
    top_k = 5

    generated_token_ids_with_batch = generate(model,config,input_ids, max_new_tokens, temperature=temperature, top_k=top_k)
    for generated_token_ids in generated_token_ids_with_batch:
        # 完全自己写的朴素解码器
        gen_text = decode(generated_token_ids.tolist())
        # 使用huggingface的 toknizer自己训练的分词器，做解码
        # gen_text = tokenizer.decode(generated_token_ids)
        print(gen_text.replace('\n',''))
        print('---------------')


import math

import torch
import pickle
from mygenerate import load_tokenizer
from model import GPTConfig
EOS_token_id = 0
pad_token_id = 5
separate_token_id = 4
IGNORE_INDEX = -100
# 将数据转换为token ids的函数
def convert_data_to_token_ids(data, encode_func, config):
    tokenized_data = []
    for item in data:
        prompt_text = item["prompt"]
        response_text = item["response"]

        # 编码提示文本和回复文本
        prompt_token_ids = encode_func(prompt_text)
        response_token_ids = encode_func(response_text)

        # 获取分隔符的 token ids
        separate_token_ids = [separate_token_id]

        # 构建输入和目标序列
        input_ids = prompt_token_ids + separate_token_ids + response_token_ids
        target_ids = [IGNORE_INDEX] * len(prompt_token_ids) + separate_token_ids + response_token_ids[1:] + [EOS_token_id]

        # 检查输入序列长度是否超出 config.block_size
        if len(input_ids) > config.block_size:
            # 如果超出,则对输入和目标序列进行截断
            input_ids = input_ids[:config.block_size]
            target_ids = target_ids[:config.block_size]

        # 计算需要填充的 token 数量
        need_pad_token_length = config.block_size - len(input_ids)

        # 构建填充部分
        # 输入部分就是正常添加pad_id
        pad_ids4input = [pad_token_id] * need_pad_token_length
        # 输出部分的pad_id就用ignore_index，这样在计算loss时，就会直接忽略pad，pad不参与loss计算
        pad_ids4target = [IGNORE_INDEX] * need_pad_token_length

        # 构建注意力掩码
        attention_mask = [1] * len(input_ids) + [0] * need_pad_token_length

        # 添加填充部分
        input_ids_with_pad = input_ids + pad_ids4input
        target_ids_with_pad = target_ids + pad_ids4target

        tokenized_data.append((input_ids_with_pad, target_ids_with_pad, attention_mask))

    return tokenized_data


# 读取本地数据文件并加载数据
def load_data_from_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


# 划分数据集为训练集和测试集
def split_dataset(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


# 将数据保存成文件
def save_data_to_file(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


import json
import os


def read_local_json(file_path, max_samples=None):
    """
    读取本地 JSON 文件,并将键名进行转换。

    参数:
    file_path (str): 本地 JSON 文件的路径。
    max_samples (int, optional): 最大读取的数据条数。如果不提供,则读取全部数据。

    返回:
    data (list): 转换后的数据列表。
    """
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在.")

    data = []
    with open(file_path, "rank", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line)
                sample["prompt"] = sample.pop("instruction")
                sample["response"] = sample.pop("output")
                data.append(sample)

                # 如果达到最大读取条数,则停止
                if max_samples is not None and len(data) >= max_samples:
                    break
            except json.JSONDecodeError as e:
                print(f"跳过无法解析的第 {i+1} 行: {e}")

    return data

if __name__ == '__main__':

    config = GPTConfig(block_size=100)
    tokenizer_dir = "pretrain_data"
    encode,decode = load_tokenizer()
    train_sample_num = 50000
    train_ratio = 0.9
    val_sample_num = (1-train_ratio) * (train_sample_num / train_ratio)
    val_sample_num = math.ceil(val_sample_num)

    max_samples = train_sample_num + val_sample_num

    # 加载数据
    # data = load_data_from_file("your_data_file.pkl")
    local_data_path = 'data/train_0.5M_CN.json'
    data = read_local_json(file_path=local_data_path,max_samples=max_samples)

    # 将数据转换为token ids
    tokenized_data = convert_data_to_token_ids(data, encode, config)

    # 划分数据集为训练集和测试集
    train_data, test_data = split_dataset(tokenized_data, train_ratio=0.9)

    # 我们在mac上做测试，每次评估val loss时，为了加快速度，我们把val_sample_num设成一个固定值小值
    val_sample_num4speedup_valadation = 100
    test_data = test_data[:val_sample_num4speedup_valadation]

    # 保存训练集和测试集数据到文件
    save_data_to_file(train_data, "data/sft_train.bin")
    save_data_to_file(test_data, "data/sft_val.bin")




"""
可以使用
encode,decode = load_tokenizer(tokenizer_dir=dataset)

input_text = '天下'
input_ids = encode(input_text)

输入数据

data = [
{"prompt":"三月北京适合去哪里玩？",
"response":"适合去颐和园西堤"}]

seperate_text = '#回复#'
EOS_token_id = 0
pad_token_id = 1
seperate_token_ids = encode(seperate_text)
prompt_token_ids
response_token_ids
input_ids = prompt_token_ids + seperate_token_ids + response_token_ids


target_ids = [-100]*(len(prompt_token_ids)) + response_token_ids[1:] + [EOS_token_id]
need_pad_token_length = config.block_size - len(input_ids)
pad_ids = [pad_token_id] *  need_pad_token_length
attention_mask = [1]*len(input_ids) + [0] * need_pad_token_length
input_ids_with_pad = input_ids + pad_ids
target_ids_with_pad = pad_ids + pad_ids
return input_ids_with_pad,target_ids_with_pad,attention_mask


sft数据格式是json其中有两个字段，prompt，response
1.把这样的数据添加分隔符#response_start#做拼接，成input_text
2.把input_text转换成 input_ids,target_ids,attention_mask
3.attention_mask中1代表有效token，0代表添加了pad的token
可以使用
encode,decode = load_tokenizer(tokenizer_dir=dataset)

input_text = '天下'
input_ids = encode(input_text)

4.再把target_ids转换成 prompt部分的token id全部替换为 config.ignore_sepicial_target_id


"""
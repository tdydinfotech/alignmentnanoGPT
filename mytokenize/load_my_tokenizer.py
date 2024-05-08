import os
import tokenizers
from transformers import PreTrainedTokenizerFast
def get_my_tokenizer():
    """
        wrapped_tokenizer = get_my_tokenizer()
        prompts = [
            "给我做一首诗歌",
            "一岁一枯荣",
        ]
        res = wrapped_tokenizer(prompts,max_length=10,padding='max_length',return_tensors='pt')
        for k,v in res.items():
            print(f"{k} : \n {v}")
        这个tokenizer就包含了encode ，decode方法了
        另外，也包含了 pad_token, unk_token, eos_token
        足够满足使用了，并且可以还是自己训练的，相当于是
        tokenizer = Autotokenizer.from_pretrained()
    """
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 tokenizer.json 文件的路径
    tokenizer_path = os.path.join(current_dir, 'tokenize_model/tokenizer.json')

    # 从文件加载 tokenizer
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)

    # 构建 PreTrainedTokenizerFast 对象
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token = '<pad>',
        unk_token='<unk>',
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        padding_side='right'
    )


    return wrapped_tokenizer


import torch


def replace_pad_token(input_ids, pad_token_id, replacement_token_id):
    """
        这在构造sft数据时需要的，而且只针对target_ids需要把pad_token_id 变成 ignore_index
        并不需要针对，input_ids来做这个操作，input_ids原本是什么就做什么就好了
    """
    # 创建一个新的 tensor 来存储替换后的 input_ids
    replaced_ids = input_ids.clone()

    # 找到所有等于 pad_token_id 的位置
    pad_indices = torch.where(input_ids == pad_token_id)

    # 将这些位置的值替换为 replacement_token_id
    replaced_ids[pad_indices] = replacement_token_id

    return replaced_ids



if __name__ == '__main__':

    # tokenizer = tokenizers.Tokenizer.from_file('tokenize_model/tokenizer.json')
    #
    # encoding_obj = tokenizer.encode("天外飞仙")
    #
    # print(f"encode_ids {encoding_obj.ids}"
    #       f"tokens {encoding_obj.tokens}"
    #       f"word_ids {encoding_obj.word_ids}"
    #       f"attention_mask {encoding_obj.attention_mask}")
    #
    # decode_text = tokenizer.decode(ids=[3009, 192, 75, 490])
    #
    # print(f"decoded text {decode_text}")

    wrapped_tokenizer = get_my_tokenizer()
    print(wrapped_tokenizer.vocab_size)
    print(wrapped_tokenizer.eos_token, wrapped_tokenizer.eos_token_id)
    print(wrapped_tokenizer.pad_token, wrapped_tokenizer.pad_token_id)
    print(wrapped_tokenizer.unk_token, wrapped_tokenizer.unk_token_id)
    prompts = [
        "给我做一首诗歌",
        "一岁一枯荣",
    ]
    res = wrapped_tokenizer(prompts, max_length=20, padding='max_length', return_tensors='pt',truncation=True)
    for k, v in res.items():
        print(f"{k} : \n {v}")

    input_ids = res['input_ids']
    input_ids = replace_pad_token(input_ids, pad_token_id=wrapped_tokenizer.pad_token_id, replacement_token_id=-100)
    print(f"replace pad with -100 {input_ids}")

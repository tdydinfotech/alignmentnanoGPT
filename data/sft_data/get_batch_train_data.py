import torch
import pickle
from model import GPTConfig

def get_sft_data_batch_genrator(data_path, split, config):
    if split == 'train':
        file_path = 'sft_train.bin'
    elif split == 'val':
        file_path = 'sft_val.bin'
    else:
        raise ValueError("Invalid split parameter. Please use 'train' or 'val'.")

    with open(data_path + file_path, 'rb') as f:
        data = pickle.load(f)

    batch_input_ids = []
    batch_target_ids = []
    batch_attention_mask = []

    for input_ids, target_ids, attention_mask in data:
        batch_input_ids.append(input_ids)
        batch_target_ids.append(target_ids)
        batch_attention_mask.append(attention_mask)

        if len(batch_input_ids) == config.batch_size:
            yield (
                torch.tensor(batch_input_ids, dtype=torch.long).to(config.device),
                torch.tensor(batch_target_ids, dtype=torch.long).to(config.device),
                torch.tensor(batch_attention_mask, dtype=torch.long).to(config.device)
            )
            batch_input_ids = []
            batch_target_ids = []
            batch_attention_mask = []

    # Yield the last batch if it's not a full batch
    if batch_input_ids:
        yield (
            torch.tensor(batch_input_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_target_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_attention_mask, dtype=torch.long).to(config.device)
        )



import random
import pickle
import torch
# 新增一个控制长度
from typing import List
def is_all_question_part(target_ids:List)->bool:
    # true 代表这个样本全都是question部分
    from mygenerate import unk_token_id
    from data.sft_data.convert_sft_data_to_token_ids import EOS_token_id,pad_token_id,separate_token_id
    new_target_ids = target_ids.copy()
    special_token_ids = [unk_token_id, EOS_token_id, pad_token_id, separate_token_id, -100]
    removed_special = [target_id for target_id in target_ids if target_id not in special_token_ids]
    return len(removed_special) == 0
    # return sum(target_ids) == -100 * len(target_ids)

import torch
import pickle
import random

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_data = self.load_and_shuffle_data('train')
        self.val_data = self.load_and_shuffle_data('val')

    def load_and_shuffle_data(self, split):
        if split == 'train':
            file_path = 'sft_train.bin'
        elif split == 'val':
            file_path = 'sft_val.bin'
        else:
            raise ValueError("Invalid split parameter. Please use 'train' or 'val'.")

        with open(self.data_path + file_path, 'rb') as f:
            data = pickle.load(f)

        # 随机打乱数据
        random.shuffle(data)

        return data

    def get_batch(self, split, config):
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            raise ValueError("Invalid split parameter. Please use 'train' or 'val'.")

        batch_input_ids = []
        batch_target_ids = []
        batch_attention_mask = []
        # 生成随机的start_pos
        start_pos = random.randint(0, len(data) - config.batch_size*3)

        sample_num = 0
        for input_ids, target_ids, attention_mask in data[start_pos:]:
            target_ids = target_ids[:config.block_size]
            if not is_all_question_part(target_ids):
                batch_input_ids.append(input_ids[:config.block_size])
                batch_target_ids.append(target_ids)
                batch_attention_mask.append(attention_mask[:config.block_size])
                sample_num += 1
                # 加满了
                if sample_num == config.batch_size:
                    break

        return (
            torch.tensor(batch_input_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_target_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_attention_mask, dtype=torch.long).to(config.device)
        )





def test_get_sft_data_batch_genrator():
    # 调用示例
    config = GPTConfig(block_size=100)
    data_path = 'data/'
    split = 'train'
    data_generator = get_sft_data_batch_genrator(data_path, split, config)
    for batch in data_generator:
        input_ids, target_ids, attention_mask = batch
        print("Input IDs shape:", input_ids.shape)
        print("Target IDs shape:", target_ids.shape)
        print("Attention Mask shape:", attention_mask.shape)
        # 在这里进行模型训练

def test_get_sft_data_batch():
    # 调用示例
    # 不支持比train.bin还长，train.bin最大长度是100
    config = GPTConfig(block_size=100)
    data_path = 'data/'
    split = 'train'
    data_loader = DataLoader(data_path)
    batch = data_loader.get_batch(split, config)
    # batch = get_sft_data_batch(data_path, split, config)
    input_ids, target_ids, attention_mask = batch
    print("Input IDs shape:", input_ids.shape)
    print("Target IDs shape:", target_ids.shape)
    print("Attention Mask shape:", attention_mask.shape)
        # 在这里进行模型训练

if __name__ == '__main__':
    # test_get_sft_data_batch_genrator()
    test_get_sft_data_batch()




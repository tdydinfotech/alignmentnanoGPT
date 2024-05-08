from model import GPTConfig
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



class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_data = self.load_and_shuffle_data('train')
        self.val_data = self.load_and_shuffle_data('val')

    def load_and_shuffle_data(self, split):
        if split == 'train':
            file_path = 'dpo_train.bin'
        elif split == 'val':
            file_path = 'dpo_val.bin'
        else:
            raise ValueError("Invalid split parameter. Please use 'train' or 'val'.")

        with open(self.data_path + file_path, 'rb') as f:
            data = pickle.load(f)

        # 随机打乱数据
        random.shuffle(data)

        return data

    def get_batch(self, split, config):
        assert config.batch_size % 2 ==0,"batch_size 必须是偶数，从上往下相邻的两个是一对"
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
        # batch2_input_ids, batch2_target_ids, batch2_attention_mask = data[start_pos]
        # 增加了一个维度， (batch_size ,2,seq_len)
        sample_num = 0
        for batch2_input_ids, batch2_target_ids, batch2_attention_mask in data[start_pos:]:
            batch2_target_ids = batch2_target_ids
            # if not is_all_question_part(target_ids):
            batch_input_ids.extend(batch2_input_ids)
            batch_target_ids.extend(batch2_target_ids)
            batch_attention_mask.extend(batch2_attention_mask)
            sample_num += 2
            # 加满了
            if sample_num == config.batch_size:
                break

        return (
            torch.tensor(batch_input_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_target_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_attention_mask, dtype=torch.long).to(config.device)
        )
    def get_batch_for_ppo_reward_model(self, split, config):
        assert config.batch_size % 2 ==0,"batch_size 必须是偶数，从上往下相邻的两个是一对"
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            raise ValueError("Invalid split parameter. Please use 'train' or 'val'.")

        batch_chosen_input_ids = []
        batch_chosen_target_ids = []
        batch_chosen_attention_mask = []

        batch_rejected_input_ids = []
        batch_rejected_target_ids = []
        batch_rejected_attention_mask = []

        # 生成随机的start_pos
        start_pos = random.randint(0, len(data) - config.batch_size*3)
        # batch2_input_ids, batch2_target_ids, batch2_attention_mask = data[start_pos]
        # 增加了一个维度， (batch_size ,2,seq_len)
        sample_num = 0
        for batch2_input_ids, batch2_target_ids, batch2_attention_mask in data[start_pos:]:
            chosen_input_ids, rejected_input_ids  = batch2_input_ids
            chosen_target_ids, rejected_target_ids  = batch2_target_ids
            chosen_attention_mask, rejected_attention_mask  = batch2_attention_mask

            batch_chosen_input_ids.append(chosen_input_ids)
            batch_chosen_target_ids.append(chosen_target_ids)
            batch_chosen_attention_mask.append(chosen_attention_mask)

            batch_rejected_input_ids.append(rejected_input_ids)
            batch_rejected_target_ids.append(rejected_target_ids)
            batch_rejected_attention_mask.append(rejected_attention_mask)

            sample_num += 2
            # 加满了
            if sample_num == config.batch_size:
                break

        return (
            torch.tensor(batch_chosen_input_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_chosen_target_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_chosen_attention_mask, dtype=torch.long).to(config.device),

            torch.tensor(batch_rejected_input_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_rejected_target_ids, dtype=torch.long).to(config.device),
            torch.tensor(batch_rejected_attention_mask, dtype=torch.long).to(config.device),
        )
def test_get_data_batch():
    # 调用示例
    # 不支持比train.bin还长，train.bin最大长度是100
    config = GPTConfig(block_size=100,batch_size=12)
    data_path = 'data/'
    split = 'train'
    data_loader = DataLoader(data_path)
    batch = data_loader.get_batch(split, config)
    # batch = get_sft_data_batch(data_path, split, config)
    input_ids, target_ids, attention_mask = batch
    print("Input IDs shape:", input_ids.shape)
    print("Target IDs shape:", target_ids.shape)
    print("Attention Mask shape:", attention_mask.shape)

def test_get_data_batch_for_reward_model():
    # 调用示例
    # 不支持比train.bin还长，train.bin最大长度是100
    config = GPTConfig(block_size=100, batch_size=12)
    data_path = 'data/'
    split = 'train'
    data_loader = DataLoader(data_path)
    batch = data_loader.get_batch_for_ppo_reward_model(split, config)
    # batch = get_sft_data_batch(data_path, split, config)
    (chosen_input_ids, chosen_target_ids, chosen_attention_mask ,
     rejected_input_ids, rejected_target_ids, rejected_attention_mask)= batch
    print("Input IDs shape:", chosen_input_ids.shape)
    print("Target IDs shape:", chosen_target_ids.shape)
    print("Attention Mask shape:", chosen_attention_mask.shape)

    print("Input IDs shape:", rejected_input_ids.shape)
    print("Target IDs shape:", rejected_target_ids.shape)
    print("Attention Mask shape:", rejected_attention_mask.shape)
        # 在这里进行模型训练

if __name__ == '__main__':
    # test_get_sft_data_batch_genrator()
    # test_get_data_batch()
    test_get_data_batch_for_reward_model()

"""
这里chosen和rejected两个样本组成了一个batch，batch size=2
所以，看起来是限制了batch size 的大小
但是其实，不是的，可以把按照
[
chosen
rejected
chosen
rejected
...
chosen
rejected
]
这样的顺序组合一个batch，然后一对一对的取出来来计算dpo loss
也可以
增加一个维度 
变成  (batch_size,pair_size,seq_len) 这样的形状
固定pair_size=2
但是，增加一个维度的话，模型不一定支持，尤其对于attention module
所以，还是做成两两一对的形式，对 model的兼容性更好，不需要改模型
"""


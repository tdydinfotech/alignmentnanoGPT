from datasets import load_dataset

data_dir = 'data/train_0.5M_CN.json'
if __name__ == '__main__':
    # 加载数据集
    dataset = load_dataset('json', data_files=data_dir)
    chosen_data = []
    rejected_data = []
    for record in dataset['train']:
        prompt = record['prompt']
        chosen = record['chosen']
        rejected = record['rejected']
        chosen_data.append(
            dict(prompt=prompt, response=chosen)
        )
        rejected_data.append(
            dict(prompt=prompt, response=rejected)
        )
    from data.sft_data.convert_sft_data_to_token_ids import convert_data_to_token_ids,save_data_to_file,split_dataset
    from model import GPTConfig
    from mygenerate import load_tokenizer
    encode, decode = load_tokenizer()
    config = GPTConfig(block_size=100)
    chosen_tokenized_data = convert_data_to_token_ids(data=chosen_data,encode_func=encode,config=config)
    rejected_tokenized_data = convert_data_to_token_ids(data=rejected_data, encode_func=encode, config=config)
    res = []
    for chosen,rejected in zip(chosen_tokenized_data,rejected_tokenized_data):
        chosen_input_ids ,chosen_target_ids,chosen_attention_mask = chosen
        rejected_input_ids, rejected_target_ids, rejected_attention_mask = rejected
        batch2_input_ids = [chosen_input_ids,rejected_input_ids]
        batch2_target_ids = [chosen_target_ids,rejected_target_ids]
        batch2_attention_mask = [chosen_attention_mask,rejected_attention_mask]
        res.append(
            (
                batch2_input_ids,batch2_target_ids,batch2_attention_mask
            )
        )
    train_data, test_data = split_dataset(res, train_ratio=0.9)
    # 保存训练集和测试集数据到文件
    save_data_to_file(train_data, "data/dpo_train.bin")
    save_data_to_file(test_data, "data/dpo_val.bin")










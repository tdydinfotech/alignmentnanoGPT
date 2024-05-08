from data.sft_data.download_data import load_belle_dataset
if __name__ == '__main__':
    load_belle_dataset(local_data_dir='data/dpo_data/data',
                       dataset_name='liyucheng/zhihu_rlhf_3k',force_download=True)

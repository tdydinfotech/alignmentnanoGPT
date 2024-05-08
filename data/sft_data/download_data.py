import os
from datasets import load_dataset


def load_belle_dataset(local_data_dir, dataset_name,force_download=False):
    """
    从本地或 Hugging Face Hub 加载 BelleGroup/train_0.5M_CN 数据集。

    参数:
    local_data_dir (str): 本地数据集存储路径。
    force_download (bool, 可选): 如果为 True,则强制从 Hugging Face Hub 下载数据集,忽略本地缓存。默认为 False。

    返回:
    dataset (datasets.Dataset): 加载的 BelleGroup/train_0.5M_CN 数据集。
    """
    # 构建本地数据集路径
    local_dataset_path = os.path.join(local_data_dir, "train_0.5M_CN.json")

    # 检查本地是否存在数据集文件
    if os.path.exists(local_dataset_path) and not force_download:
        # 如果存在且 force_download 为 False,则直接从本地加载
        print("Loading dataset from local path:", local_dataset_path)
        dataset = load_dataset("json", data_files=local_dataset_path)
    else:
        # 如果不存在或 force_download 为 True,则从 Hugging Face Hub 下载
        print("Dataset not found locally or force_download is True, downloading from Hugging Face Hub...")
        dataset = load_dataset(dataset_name)

        # 创建本地数据目录(如果不存在)
        os.makedirs(local_data_dir, exist_ok=True)

        # 将下载的数据集保存到本地
        dataset["train"].to_json(local_dataset_path)
        dataset["train"].to_json(local_dataset_path)
        print("Dataset saved to local path:", local_dataset_path)

    return dataset
if __name__ == '__main__':
    dataset = load_belle_dataset(local_data_dir='data/',dataset_name='"BelleGroup/train_0.5M_CN"')

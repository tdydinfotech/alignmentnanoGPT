import tokenizers
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

def get_training_corpus(file_path='/Users/dino/Documents/code/alignmentnanoGPT/data/pretrain_data/chinese_poetry40k.txt', chunk_size=10):
    """
    生成器函数,从文件中每次读取chunk_size行作为语料库数据

    Args:
        file_path (str): 训练语料库文件路径
        chunk_size (int): 每次读取的行数

    Yields:
        list: 长度为chunk_size的文本行列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = []
        for line in f:
            lines.append(line.strip())
            if len(lines) == chunk_size:
                yield lines
                lines = []
        if lines:
            yield lines

def train_new_tokenizer_model():
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())

    # 预标记化
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 训练tokenizer
    trainer = trainers.BpeTrainer(vocab_size=6859, special_tokens=["<|endoftext|>","<unk>"])

    # 从迭代器训练
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # 或者从文件训练
    # tokenizer.model = models.BPE()
    # tokenizer.train(["wikitext-2.txt"], trainer=trainer)

    # 后处理
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # 解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 保存tokenizer

    # 保存tokenizer
    tokenizer.save("tokenize_model/tokenizer.json")


if __name__ == '__main__':
    train_new_tokenizer_model()


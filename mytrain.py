import torch
import matplotlib.pyplot as plt
from model import GPT,GPTConfig
import os,numpy as np
from typing import Callable,List,Dict,Optional
# 学习率衰减调度器（余弦退火加热）
import math
import os
import torch
import numpy as np

dataset = 'pretrain_data'
data_dir = os.path.join('data', dataset)
def get_batch(split,config:GPTConfig):
    # 我们每个批次都重新创建 np.memmap，以避免内存泄漏，参考：
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    block_size = config.block_size
    device = config.device
    batch_size = config.batch_size
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if config.device == 'cuda':
        # 将数组 input_ids、y 固定，这样我们可以将它们异步移动到 GPU（non_blocking=True）
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    attention_mask = torch.ones(batch_size,block_size)
    return x, y, attention_mask

class CosineAnnealingWithWarmupScheduler:
    """
    余弦退火学习率调度器,包含预热阶段。

    参数:
    learning_rate (float): 初始学习率。
    min_lr (float): 最小学习率。
    lr_decay_iters (int): 学习率衰减迭代次数。
    warmup_iters (int): 预热迭代次数。
    """

    def __init__(self, learning_rate, min_lr, lr_decay_iters, warmup_iters):
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.lr_decay_iters = lr_decay_iters
        self.warmup_iters = warmup_iters

    def get_lr(self, iteration):
        """
        根据当前迭代次数计算学习率。

        参数:
        iteration (int): 当前迭代次数。

        返回:
        float: 当前学习率。
        """
        # 1) warmup_iters 步的线性预热
        if iteration < self.warmup_iters:
            # 避免iteration=0，那样的话学习率就是0
            return self.learning_rate * (iteration + 1) / self.warmup_iters

        # 2) 如果 iteration > lr_decay_iters，则返回最小学习率
        if iteration > self.lr_decay_iters:
            return self.min_lr

        # 3) 在此之间，使用余弦衰减到最小学习率
        decay_ratio = (iteration - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff 范围 0 到 1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    """
    # 创建调度器实例
    scheduler = CosineAnnealingWithWarmupScheduler(
        learning_rate=1e-4,
        min_lr=6e-5,
        lr_decay_iters=600000,
        warmup_iters=2000
    )

    # 获取学习率
    current_lr = scheduler.get_lr(iteration=10000)
    print(f"Current learning rate: {current_lr:.6f}")
    """

@torch.no_grad()
def estimate_loss(model,get_batch:Callable,eval_iters = 10)->float:
    out = None
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y,attention_mask = get_batch()
        attention_mask = convert_1d_pad_mask_to_2d_attention_mask(attention_mask)
        logits, loss = model(input_ids = X, target_ids=Y,attention_mask=attention_mask)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

def save_checkpoint(model, optimizer, config:GPTConfig, iter_num, best_val_loss, out_dir):
    """
    保存模型检查点函数

    参数:
        model: 训练的模型
        optimizer: 优化器
        config: 模型配置
        iter_num: 当前迭代次数
        best_val_loss: 当前最佳的验证损失
        checkpoint_save_path: 检查点保存的目录
    """
    # 从配置中获取模型参数
    model_args = {
        'n_layer': config.n_layer,
        'n_head': config.n_head,
        'n_embd': config.n_embd,
        'block_size': config.block_size,
        'bias': config.bias,
        'vocab_size': config.vocab_size,  # 这个需要根据具体情况赋值
        'dropout': config.dropout,
    }

    # 构建检查点
    checkpoint = {
        'model': model.state_dict(),  # 模型状态字典
        'optimizer': optimizer.state_dict(),  # 优化器状态字典
        'model_args': model_args,  # 模型参数
        'iter_num': iter_num,  # 当前迭代次数
        'best_val_loss': best_val_loss,  # 当前最佳验证损失
        'config': config,  # 模型配置
    }

    # 打印保存路径并保存检查点
    print(f"保存检查点到 {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'myckpt.pt'))

def load_checkpoint(checkpoint_dir,model=None):
    """
    加载模型检查点函数

    参数:
        model: 待加载的模型
        optimizer: 待加载的优化器
        model_dir: 检查点保存的目录
    返回:
        model: 加载后的模型
        optimizer: 加载后的优化器
        iter_num: 加载后的迭代次数
        best_val_loss: 加载后的最佳验证损失
        config: 加载后的模型配置
    """
    # 确保检查点文件存在
    checkpoint_path = os.path.join(checkpoint_dir, 'myckpt.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint file not found")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    if not model:
        model = GPT(config)
    # 加载模型参数
    model.load_state_dict(checkpoint['model'])

    # optimizer = model.configure_optimizers()
    optimizer = None
    # 加载优化器状态
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # 加载其他信息
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']


    return model, optimizer, iter_num, best_val_loss, config

# 使用示例
# 实例化模型、优化器和配置
# model = MyModel(...)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# config = GPTConfig(...)

# 加载检查点
# model, optimizer, iter_num, best_val_loss, config = load_checkpoint(model, optimizer, model_dir)

from utils.basic_utils import convert_1d_pad_mask_to_2d_attention_mask

def train_model(model, optimizer, num_epochs, num_batches, get_batch, need_save=True,
                checkpoint_save_path ='pretrain_checkpoint', save_checkpoint_interval_iter_num = 10, scheduler:CosineAnnealingWithWarmupScheduler=None,config:GPTConfig=None):
    device = next(model.parameters()).device
    losses = []
    iter_num = 0 # 能整除save_checkpoint_interval_iter_num
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx in range(num_batches):
            input_ids, target_ids, pad_mask = get_batch('train')
            try:
                assert input_ids is not None and target_ids is not None and pad_mask is not None
            except Exception as e:
                print(f"数据异常 input_ids or target_ids or pad_mask 为空")
                continue
            input_ids, target_ids ,attention_mask = input_ids.to(device), target_ids.to(device), pad_mask.to(device)
            attention_mask = convert_1d_pad_mask_to_2d_attention_mask(attention_mask)

            lr = scheduler.get_lr(iteration=iter_num) if config.decay_lr else config.learning_rate

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            logits, loss = model(input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iter_num += 1
        print(f"步骤 {iter_num}：训练损失 {loss:.4f}")


        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        if need_save:
            # 或者到迭代次数的最后了也要判断下
            max_item_num = num_epochs * num_batches
            if iter_num % save_checkpoint_interval_iter_num ==0 or iter_num==max_item_num:
                val_loss = estimate_loss(model,get_batch=lambda :get_batch(split='val'),eval_iters=5)
                if val_loss<best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, config, iter_num, best_val_loss, out_dir=checkpoint_save_path)
                    print(f'保存第{iter_num}轮的模型checkpoint到{checkpoint_save_path}')
                print(f"步骤 {iter_num}：训练损失 {loss:.4f}，验证损失 {val_loss:.4f}")

    # 在每个 epoch 结束后打印 moe 层的选择次数
    if config.use_moe:
        print("专家选择次数:", model.transformer.attn[0].ffn.expert_selection_counts)
    plot_loss = True
    if plot_loss:
        # 绘制损失曲线
        losses = moving_average(losses,window_size=5)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)),losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison (Smoothed)')
        plt.show()
    return losses

# 比较不同模型的训练损失

from utils.basic_utils import moving_average
def compare_models(models, num_epochs, num_batches, get_batch, window_size=10):
    import time

    all_losses = {}

    for model_name, (model, optimizer) in models.items():
        print(f"Training {model_name}...")
        start_time = time.time()  # 记录训练开始时间
        losses = train_model(model, optimizer, num_epochs, num_batches, get_batch,need_save=False)
        smoothed_losses = moving_average(losses, window_size)
        end_time = time.time()  # 记录训练结束时间
        training_time = end_time - start_time  # 计算训练耗时
        all_losses[model_name] = smoothed_losses
        print(f"Training of {model_name} completed in {training_time:.2f} seconds.")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    for model_name, losses in all_losses.items():
        plt.plot(losses, label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison (Smoothed)')
    plt.legend()
    plt.show()

def get_vocab_size():
    import pickle
    # 尝试从数据集中推断出 vocab_size
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"找到 vocab_size = {meta_vocab_size}（在 {meta_path} 中）")
        return meta_vocab_size

if __name__ == '__main__':
    from model import GPT

    vocab_size = get_vocab_size()
    batch_size = 12

    config = GPTConfig(vocab_size=vocab_size, block_size=5,
                       n_embd=16, n_head=4,
                       n_layer=1, dropout=0.1,
                       bias=False, ffn_multiply_scalar=1,
                       device='cpu', batch_size=batch_size,
                       decay_lr=True, residual=False,
                       use_moe=False, use_ffn=True, use_glu_only=False, use_mlp_only=True,
                       norm_type='rmsnorm', use_abs_int_pos_emb=False,
                       use_abs_sin_pos_emb=False, use_rope_pos_emb=True,cache_max_batch_size=32)

    model = GPT(config)
    beta1 = 0.9
    beta2 = 0.95
    learning_rate = 6e-4  # 最大学习率
    weight_decay = 1e-1
    # 创建调度器实例
    scheduler = CosineAnnealingWithWarmupScheduler(
        learning_rate=1e-4,
        min_lr=6e-5,
        lr_decay_iters=600000,
        warmup_iters=2000
    )
    device_type = 'cpu'
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)


    num_epochs = 400
    num_batches = 20
    my_get_batch = lambda split: get_batch(split,config)

    losses:List = train_model(model,optimizer,num_epochs,num_batches,my_get_batch,
                              scheduler=scheduler,
                              checkpoint_save_path='pretrain_checkpoint',
                              save_checkpoint_interval_iter_num=200,config=config)

    # -------------------------------------------------------------------
    #          下面是，可以做各种对比实验，
    #          比如
    #          使用残差前后对比，
    #          使用GLU前后对比，
    #          使用moe前后对比，
    # -------------------------------------------------------------------
    # -----------------------在不使用ffn的情况下，看看多头vs单头的影响-------------------------------------------
    # config_no_ffn_single_head = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                     n_embd=64, n_head=1,
    #                     n_layer=1, dropout=0.1,
    #                     bias=False, ffn_multiply_scalar=4,
    #                     device='cpu', batch_size=batch_size, decay_lr=True,use_ffn=True)
    #
    # config_no_ffn_multi_head = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                     n_embd=64, n_head=16,
    #                     n_layer=1, dropout=0.1,
    #                     bias=False, ffn_multiply_scalar=4,
    #                     device='cpu', batch_size=batch_size, decay_lr=True,use_ffn=True)
    # ------------------------------------------------------------------

    # config1 = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                    n_embd=64, n_head=1,
    #                    n_layer=1, dropout=0.1,
    #                    bias=False, ffn_multiply_scalar=1,
    #                    device='cpu',batch_size=batch_size,decay_lr=True)
    #
    # config2 = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                    n_embd=64, n_head=16,
    #                    n_layer=1, dropout=0.1,
    #                    bias=False, ffn_multiply_scalar=1,
    #                    device='cpu',batch_size=batch_size,decay_lr=True)
    # config3 = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                     n_embd=64, n_head=16,
    #                     n_layer=1, dropout=0.1,
    #                     bias=False, ffn_multiply_scalar=4,
    #                     device='cpu', batch_size=batch_size, decay_lr=True)
    # config4 = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                     n_embd=64, n_head=16,
    #                     n_layer=1, dropout=0.1,
    #                     bias=False, ffn_multiply_scalar=4,
    #                     device='cpu', batch_size=batch_size, decay_lr=True,residual=False)
    # ------------------对比看下层数对效果的影响-------------------------
    # single_layer = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                     n_embd=16, n_head=4,
    #                     n_layer=1, dropout=0.1,
    #                     bias=False, ffn_multiply_scalar=1,
    #                     device='cpu', batch_size=batch_size, decay_lr=True,residual=True)
    #
    # multi_layer = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                     n_embd=16, n_head=4,
    #                     n_layer=4, dropout=0.1,
    #                     bias=False, ffn_multiply_scalar=1,
    #                     device='cpu', batch_size=batch_size, decay_lr=True, residual=True)
    # useffn = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                     n_embd=16, n_head=4,
    #                     n_layer=1, dropout=0.1,
    #                     bias=False, ffn_multiply_scalar=1,
    #                     device='cpu', batch_size=batch_size,
    #                    decay_lr=True,residual=False,
    #                    use_moe=False)
    #
    # usemoe = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                          n_embd=16, n_head=4,
    #                          n_layer=1, dropout=0.1,
    #                          bias=False, ffn_multiply_scalar=1,
    #                          device='cpu', batch_size=batch_size,
    #                    decay_lr=True, residual=False,
    #                    use_moe=True,expert_type='mlp',num_expert=6,num_expert_per_token=6)
    # 对比 glu vs ffn
    # useffn = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                     n_embd=16, n_head=4,
    #                     n_layer=1, dropout=0.1,
    #                     bias=False, ffn_multiply_scalar=1,
    #                     device='cpu', batch_size=batch_size,
    #                    decay_lr=True,residual=False,
    #                    use_moe=False,use_ffn=True,use_glu_only=False,use_mlp_only=True)
    #
    # usemoe = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                          n_embd=16, n_head=4,
    #                          n_layer=1, dropout=0.1,
    #                          bias=False, ffn_multiply_scalar=1,
    #                          device='cpu', batch_size=batch_size,
    #                    decay_lr=True, residual=False,
    #                    use_moe=False,use_ffn=False,use_glu_only=True,use_mlp_only=False)
    # # 对比 layernorm vs rmsnorm
    # useffn = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                     n_embd=16, n_head=4,
    #                     n_layer=1, dropout=0.1,
    #                     bias=True, ffn_multiply_scalar=1,
    #                     device='cpu', batch_size=batch_size,
    #                    decay_lr=True,residual=False,
    #                    use_moe=False,use_ffn=True,use_glu_only=False,use_mlp_only=True,
    #                    norm_type='layernorm')
    #
    # usemoe = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                    n_embd=16, n_head=4,
    #                    n_layer=1, dropout=0.1,
    #                    bias=True, ffn_multiply_scalar=1,
    #                    device='cpu', batch_size=batch_size,
    #                    decay_lr=True, residual=False,
    #                    use_moe=False, use_ffn=True, use_glu_only=False, use_mlp_only=True,
    #                    norm_type='rmsnorm')

    # # 对比 layernorm vs rmsnorm
    # useffn = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                    n_embd=16, n_head=4,
    #                    n_layer=3, dropout=0.1,
    #                    bias=False, ffn_multiply_scalar=1,
    #                    device='cpu', batch_size=batch_size,
    #                    decay_lr=True, residual=False,
    #                    use_moe=False, use_ffn=True, use_glu_only=False, use_mlp_only=True,
    #                    norm_type='rmsnorm', use_abs_int_pos_emb=True)
    #
    # usemoe = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                    n_embd=16, n_head=4,
    #                    n_layer=3, dropout=0.1,
    #                    bias=False, ffn_multiply_scalar=1,
    #                    device='cpu', batch_size=batch_size,
    #                    decay_lr=True, residual=False,
    #                    use_moe=False, use_ffn=True, use_glu_only=False, use_mlp_only=True,
    #                    norm_type='rmsnorm', use_abs_int_pos_emb=False)
    #
    # 对比 abs_int_pos_emb vs rope
    # useffn = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                    n_embd=16, n_head=4,
    #                    n_layer=2, dropout=0.1,
    #                    bias=False, ffn_multiply_scalar=1,
    #                    device='cpu', batch_size=batch_size,
    #                    decay_lr=True, residual=False,
    #                    use_moe=False, use_ffn=True, use_glu_only=False, use_mlp_only=True,
    #                    norm_type='rmsnorm', use_abs_int_pos_emb=True,use_rope_pos_emb=False)
    #
    # usemoe = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                    n_embd=16, n_head=4,
    #                    n_layer=2, dropout=0.1,
    #                    bias=False, ffn_multiply_scalar=1,
    #                    device='cpu', batch_size=batch_size,
    #                    decay_lr=True, residual=False,
    #                    use_moe=False, use_ffn=True, use_glu_only=False, use_mlp_only=True,
    #                    norm_type='rmsnorm', use_abs_int_pos_emb=True,use_rope_pos_emb=True)
    #
    #
    # # 对比 abs_sin_pos_emb vs rope
    # useffn = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                    n_embd=16, n_head=4,
    #                    n_layer=2, dropout=0.1,
    #                    bias=False, ffn_multiply_scalar=1,
    #                    device='cpu', batch_size=batch_size,
    #                    decay_lr=True, residual=False,
    #                    use_moe=False, use_ffn=True, use_glu_only=False, use_mlp_only=True,
    #                    norm_type='rmsnorm', use_abs_int_pos_emb=False,
    #                    use_abs_sin_pos_emb=True,use_rope_pos_emb=False)
    #
    # usemoe = GPTConfig(vocab_size=vocab_size, block_size=32,
    #                    n_embd=16, n_head=4,
    #                    n_layer=2, dropout=0.1,
    #                    bias=False, ffn_multiply_scalar=1,
    #                    device='cpu', batch_size=batch_size,
    #                    decay_lr=True, residual=False,
    #                    use_moe=False, use_ffn=True, use_glu_only=False, use_mlp_only=True,
    #                    norm_type='rmsnorm', use_abs_int_pos_emb=False,
    #                    use_abs_sin_pos_emb=False,use_rope_pos_emb=True)
    #

    # 定义模型和优化器
    # model_useffn = GPT(useffn)
    # optimizer_useffn = model_useffn.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    #
    # model_usemoe = GPT(usemoe)
    # optimizer_usemoe = model_usemoe.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    # model_no_ffn_single_head = GPT(config_no_ffn_single_head)
    # optimizer_no_ffn_single_head = model_no_ffn_single_head.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    #
    # model_no_ffn_multi_head = GPT(config_no_ffn_multi_head)
    # optimizer_no_ffn_multi_head = model_no_ffn_multi_head.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    # --------------------------
    # model_no_ffn_single_head = GPT(single_layer)
    # optimizer_no_ffn_single_head = model_no_ffn_single_head.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    # model_no_ffn_multi_head = GPT(multi_layer)
    # optimizer_no_ffn_multi_head = model_no_ffn_multi_head.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    # -----------------------------------
    # model1 = GPT(config1)
    # optimizer1 = model1.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    #
    # model2 = GPT(config2)
    # optimizer2 = model2.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    #
    # model3 = GPT(config3)
    # optimizer3 = model3.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    #
    # model4 = GPT(config4)
    # optimizer4 = model4.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    # models = {
        # 'layer_1_head_4_ffn_1_residual_Yes_only_rotate_pos_emb': (model_usemoe, optimizer_usemoe),
        # 'layer_1_head_4_ffn_1_residual_Yes_only_abs_sin_pos_emb': (model_useffn, optimizer_useffn),
        # 'layer_1_head_4_ffn_1_residual_Yes': (model_no_ffn_single_head, optimizer_no_ffn_single_head),
        # 'layer_4_head_4_ffn_1_residual_Yes': (model_no_ffn_multi_head, optimizer_no_ffn_multi_head),
        # 'head_1_ffn_10_residual_Yes': (model0, optimizer0),
        # 'head_1_ffn_1_residual_Yes': (model1,optimizer1),
        # 'head_16_ffn_1_residual_Yes': (model2,optimizer2),
        # 'head_16_ffn_4_residual_Yes': (model3, optimizer3),
        # 'head_16_ffn_4_residual_No': (model4, optimizer4),
        # 添加更多模型...
    # }

    # 比较不同模型的损失
    # compare_models(models, num_epochs, num_batches,my_get_batch,window_size=4)


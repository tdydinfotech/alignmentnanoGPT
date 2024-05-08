"""
实现使用dpo方式对gpt进行微调
方式和sft基本一样，主要差别是
1. 模型增加了一个ref_model
2. 数据增加一个  chosen样本和一个rejecteded样本
也可以像qlora那样，model使用lora model
ref model使用量化model，因为ref模型不做训练只做推理
"""
import torch
from mytrain import CosineAnnealingWithWarmupScheduler,estimate_loss, save_checkpoint
from model import GPTConfig,GPT
from utils.basic_utils import moving_average, convert_1d_pad_mask_to_2d_attention_mask
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.nn.functional as F

from utils.basic_utils import get_even_odd_elements




def get_dpo_loss(model:GPT,ref_model:GPT,input_ids,target_ids,attention_mask,beta=1)->torch.Tensor:
    dpo_loss = None

    def get_model_chosen_and_rejected_logp(model,trainable:bool,input_ids,target_ids,attention_mask):
        log_prob_model_chosen = None
        log_prob_model_rejected = None
        # input_ids 和 target_ids 都是batch_size=2的，第一句子是被接受的，第二句子是被拒绝的
        # 大小都是(batch_size,seq_len)
        if trainable:
            logits,_ = model(input_ids=input_ids,target_ids=target_ids,attention_mask=attention_mask)
        else:
            with torch.no_grad():
                logits, _ = model(input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask)

        # logits [batch_size,seq_len,vocb_size]
        log_prob_with_vocab = torch.log_softmax(logits,dim=-1)
        target_ids4gater = target_ids.unsqueeze(-1).clone()  # 创建副本
        target_ids4gater[target_ids4gater == -100] = 0  # 在副本上进行修改
        log_prob_target_sentence = torch.gather(log_prob_with_vocab,index=target_ids4gater,dim=-1)
        log_prob_target_sentence = log_prob_target_sentence.squeeze(-1)
        log_prob_target_sentence = log_prob_target_sentence.sum(-1)

        # 构造示例输入序列
        index = torch.arange(log_prob_target_sentence.shape[0])

        # 调用函数获取偶数索引和奇数索引的元素
        even_indexs, odd_index = get_even_odd_elements(index)


        # batch中有两句，第一句是被接受的，第二句是被拒绝的
        log_prob_model_chosen = log_prob_target_sentence[even_indexs]
        log_prob_model_rejected = log_prob_target_sentence[odd_index]

        return log_prob_model_chosen, log_prob_model_rejected, log_prob_target_sentence

    log_prob_model_chosen, log_prob_model_rejected, log_prob = get_model_chosen_and_rejected_logp(model, True,input_ids,target_ids,
                                                                                      attention_mask)
    log_prob_ref_model_chosen, log_prob_ref_model_rejected,log_prob_ref = get_model_chosen_and_rejected_logp(ref_model, False,input_ids, target_ids,
                                                                                      attention_mask)

    chosen_infogain = log_prob_model_chosen - log_prob_ref_model_chosen
    rejected_infogain =  log_prob_model_rejected - log_prob_ref_model_rejected
    margin_logits = chosen_infogain - rejected_infogain
    dpo_loss = -F.logsigmoid(beta * margin_logits).mean()

    # dpo也可以计算出奖励
    reward = beta * (log_prob - log_prob_ref).detach()

    return dpo_loss


def dpo_train_model(model:GPT, optimizer, num_epochs, num_batches,
                get_batch,
                need_save=True,
                checkpoint_save_path ='pretrain_checkpoint',
                save_checkpoint_interval_iter_num = 10,
                scheduler:CosineAnnealingWithWarmupScheduler=None,
                config:GPTConfig=None,quantize_ref_model:bool=False):

    device = next(model.parameters()).device
    losses = []
    iter_num = 0 # 能整除save_checkpoint_interval_iter_num
    best_val_loss = float('inf')
    ref_model = deepcopy(model)
    # 可以选择是否要量化ref model
    if quantize_ref_model:
        ref_model = convert_model_to_quantized_model(ref_model)

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
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()

                loss = get_dpo_loss(model,ref_model,input_ids,target_ids,attention_mask)

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

from typing import Callable,List
def get_batch_function_for_dpo(config)->Callable:
    from data.dpo_data.get_batch_data import DataLoader
    data_path = 'data/dpo_data/data/'
    dataloader = DataLoader(data_path)
    my_get_batch = lambda split: dataloader.get_batch(split=split,config=config)
    return my_get_batch

def load_sft_model():
    from mytrain import load_checkpoint
    checkpoint_dir = 'sft_merge_checkpoint'
    # 我先不从ckpt加载，看看，原始模型能否训练
    config: GPTConfig
    model, optimizer, iter_num, best_val_loss, config = load_checkpoint(checkpoint_dir)
    config.model_type = 'mygpt'

    config.batch_size = 2  # 变小一点，微调时需要用更少的数据，更小的学习率
    config.block_size = 100

    model.config = config
    return model, config

def convert_model_to_quantized_model(model):
    from myquantization.myquantize import QuantizationHandler,QuantizationConfig,replace_linear_with_linearkbit
    quantize_config: QuantizationConfig = QuantizationConfig(precision=4, exclude_modules=['lm_head'])

    # model = set_parameters_requires_grad(model)

    quantizer = QuantizationHandler(min_val=-1, max_val=2, n_bits=8)
    quantized_model, has_been_replaced = replace_linear_with_linearkbit(model, config=quantize_config,
                                                                        quantizer=quantizer)
    return quantized_model

if __name__ == '__main__':
    from model import GPT, GPTConfig
    # 创建调度器实例
    scheduler = CosineAnnealingWithWarmupScheduler(
        learning_rate=1e-4, # 微调时把lr调小一点
        min_lr=6e-5, # 微调时把lr调小一点
        lr_decay_iters=600000,
        warmup_iters=2000
    )

    num_epochs = 1000
    num_batches = 10
    """
    1. 可以直接对pertrain model进行微调不做sft
    2. dpo可以先把model转为lora model或者qlora model进行微调
    3. ref model就可以直接转化为 quantized model只做推理    
    """
    # from mysft import load_pretrain_model
    # pertrain_model,config = load_pretrain_model()
    # model = pertrain_model

    sfted_model, config = load_sft_model()
    model = sfted_model

    config.batch_size = 4

    from mylora.minilora import MiniLoraConfig,get_mini_peft_model
    peft_config = MiniLoraConfig(
        rank=2,
        lora_alpha=2,
        target_modules=['q_proj', 'v_proj','k_proj','w_proj','ffn_up','ffn_down'],
        use_qlora=True,
        quantize_modules=[]

    )
    # 其实这里直接用的model，而不是quantized_model，只使用get_mini_peft_model就可以实现我们想要的量化目标了
    lora_model = get_mini_peft_model(model, peft_config)
    model = lora_model

    # 这里确保一致，是从要被训练的model获得的优化器
    optimizer = model.configure_optimizers()

    get_batch_fn = get_batch_function_for_dpo(config)


    losses: List = dpo_train_model(model, optimizer, num_epochs, num_batches, get_batch_fn,
                               scheduler=scheduler,
                               checkpoint_save_path='dpo_checkpoint',
                               save_checkpoint_interval_iter_num=800,config=config)


"""
相比，sft来说，训练更困难了，速度更慢，loss下降也更慢了
loss明显更加抖动
1. 把要使用dpo的model转为qlora model 进行训练是可以的
2. 把ref model转为 量化后的model也是可以的
节省显存
"""


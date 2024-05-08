import torch
from torch.nn import Module,Linear
import torch.nn.functional as F
from model import GPTConfig,GPT
from typing import Callable,List

class RewardModel(Module):
    def __init__(self,model:GPT,config:GPTConfig,**kwargs):
        super().__init__()
        self.model = model
        self.reward_predict_linear = Linear(in_features=config.n_embd,
                                            out_features=1,bias=False,device=config.device)
        self.loss_balance_factor = kwargs.get('loss_balance_factor',1)

    def reward_forward(self, input_ids=None,target_ids=None,attention_mask=None):
        """
         只需要传入当前token_id计算出一个reward就行了
        """
        logits,loss = self.model.forward(input_ids=input_ids,target_ids=target_ids,attention_mask=attention_mask)
        # last_hidden_state is shape like (batch_size,seq_len,hidden_dim)
        reward_predict = self.reward_predict_linear(self.model.last_hidden_state)
        return reward_predict

    def forward(self, prompt_positive_response_token_ids,
                attention_mask_prompt_positive_response,
                prompt_negative_response_token_ids,
                attention_mask_prompt_negative_response,
                labels,
                prompt_ids, lm_attn_mask, target_response_ids, **kargs):
        """
        prompt_positive_response_token_ids: 输入词元和正例输出词元拼接后的标号序列。
        attention_mask_prompt_positive_response: prompt_positive_response_token_ids 对应的注意力掩码。
        prompt_negative_response_token_ids: 输入词元和负例输出词元拼接后的标号序列。
        attention_mask_prompt_negative_response: prompt_negative_response_token_ids 对应的注意力掩码。
        labels: 正例输出所在的序列（均为 0，表示正例在 prompt_positive_response_token_ids 中）。
        prompt_ids: 输入词元和正例输出词元拼接后的标号序列。
        lm_attn_mask: prompt_ids 对应的注意力掩码。
        target_response_ids: 计算交叉熵损失时目标的标号序列。
        **kargs: 其他可选参数。
        """
        positive_response_reward = self.reward_forward(input_ids=prompt_positive_response_token_ids,
                            attention_mask=attention_mask_prompt_positive_response)
        negitive_response_reward = self.reward_forward(input_ids=prompt_negative_response_token_ids,
                            attention_mask=attention_mask_prompt_negative_response)
        reward_margin_logits = positive_response_reward - negitive_response_reward
        reward_margin_logits = reward_margin_logits.squeeze(-1)
        # target_response_margin_logits = torch.gather(reward_margin_logits,dim=1,index=labels.unsqueeze(-1))
        # loss = - F.logsigmoid(target_response_margin_logits).mean()
        torch.nn.BCEWithLogitsLoss()
        loss = F.binary_cross_entropy_with_logits(input=reward_margin_logits,target=labels)

        _, generate_loss = self.model.forward(input_ids=prompt_ids,target_ids=target_response_ids,attention_mask=lm_attn_mask)
        penalty_loss = generate_loss

        total_loss = loss + self.loss_balance_factor * penalty_loss
        return total_loss


def get_loss(model:RewardModel,
             chosen_input_ids,chosen_target_ids,chosen_attention_mask,
             rejected_input_ids,rejected_target_ids,rejected_attention_mask)->torch.Tensor:
    loss = None
    loss = model.forward(
        prompt_positive_response_token_ids=chosen_input_ids,
        attention_mask_prompt_positive_response=chosen_attention_mask,
        prompt_negative_response_token_ids=rejected_input_ids,
        attention_mask_prompt_negative_response=rejected_attention_mask,
        labels=torch.zeros_like(chosen_input_ids,dtype=torch.float32),
        prompt_ids = chosen_input_ids,
        lm_attn_mask = chosen_attention_mask,
        target_response_ids = chosen_target_ids
    )
    return loss

@torch.no_grad()
def estimate_loss(model,get_batch:Callable,eval_iters = 10)->float:
    out = None
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        batch_data = get_batch()
        # batch_data_with_device = [tensor_input.to(device) for tensor_input in batch_data]
        chosen_input_ids, chosen_target_ids, chosen_pad_mask, \
            rejected_input_ids, rejected_target_ids, rejected_pad_mask = batch_data

        from utils.basic_utils import convert_1d_pad_mask_to_2d_attention_mask
        chosen_attention_mask = convert_1d_pad_mask_to_2d_attention_mask(chosen_pad_mask)
        rejected_attention_mask = convert_1d_pad_mask_to_2d_attention_mask(rejected_pad_mask)

        loss = model.forward(
            prompt_positive_response_token_ids=chosen_input_ids,
            attention_mask_prompt_positive_response=chosen_attention_mask,
            prompt_negative_response_token_ids=rejected_input_ids,
            attention_mask_prompt_negative_response=rejected_attention_mask,
            labels=torch.zeros_like(chosen_input_ids, dtype=torch.float32),
            prompt_ids=chosen_input_ids,
            lm_attn_mask=chosen_attention_mask,
            target_response_ids=chosen_target_ids
        )

        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def train_model(model:GPT, optimizer, num_epochs, num_batches,
                get_batch,
                need_save=True,
                checkpoint_save_path ='pretrain_checkpoint',
                save_checkpoint_interval_iter_num = 10,
                scheduler=None,
                config:GPTConfig=None,quantize_ref_model:bool=False):

    device = next(model.parameters()).device
    losses = []
    iter_num = 0 # 能整除save_checkpoint_interval_iter_num
    best_val_loss = float('inf')
    # ref_model = deepcopy(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx in range(num_batches):
            batch_data = get_batch('train')
            batch_data_with_device = [tensor_input.to(device) for tensor_input in batch_data]
            chosen_input_ids, chosen_target_ids, chosen_pad_mask,\
            rejected_input_ids, rejected_target_ids, rejected_pad_mask = batch_data_with_device

            from utils.basic_utils import convert_1d_pad_mask_to_2d_attention_mask
            chosen_attention_mask = convert_1d_pad_mask_to_2d_attention_mask(chosen_pad_mask)
            rejected_attention_mask = convert_1d_pad_mask_to_2d_attention_mask(rejected_pad_mask)

            lr = scheduler.get_lr(iteration=iter_num) if config.decay_lr else config.learning_rate

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()

                loss = get_loss(model,chosen_input_ids,chosen_target_ids,chosen_attention_mask,
                                rejected_input_ids,rejected_target_ids,rejected_attention_mask)

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
                    from mytrain import save_checkpoint
                    save_checkpoint(model, optimizer, config, iter_num, best_val_loss, out_dir=checkpoint_save_path)
                    print(f'保存第{iter_num}轮的模型checkpoint到{checkpoint_save_path}')
                print(f"步骤 {iter_num}：训练损失 {loss:.4f}，验证损失 {val_loss:.4f}")

    plot_loss = True
    if plot_loss:
        # 绘制损失曲线
        from utils.basic_utils import moving_average
        import matplotlib.pyplot as plt
        losses = moving_average(losses,window_size=5)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(losses)),losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison (Smoothed)')
        plt.show()
    return losses


def get_batch_function_for_ppo_reward_model(config)->Callable:
    from data.dpo_data.get_batch_data import DataLoader
    data_path = 'data/dpo_data/data/'
    dataloader = DataLoader(data_path)
    my_get_batch = lambda split: dataloader.get_batch_for_ppo_reward_model(split=split,config=config)
    return my_get_batch


if __name__ == '__main__':
    from model import GPT, GPTConfig
    from mytrain import CosineAnnealingWithWarmupScheduler

    # 创建调度器实例
    scheduler = CosineAnnealingWithWarmupScheduler(
        learning_rate=1e-4,  # 微调时把lr调小一点
        min_lr=6e-5,  # 微调时把lr调小一点
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
    from dpo_training import load_sft_model
    sfted_model, config = load_sft_model()
    model = RewardModel(model=sfted_model,config=config)

    config.batch_size = 4

    # from mylora.minilora import MiniLoraConfig, get_mini_peft_model
    #
    # peft_config = MiniLoraConfig(
    #     rank=2,
    #     lora_alpha=2,
    #     target_modules=['q_proj', 'v_proj', 'k_proj', 'w_proj', 'ffn_up', 'ffn_down'],
    #     use_qlora=True,
    #     quantize_modules=[]
    #
    # )
    # # 其实这里直接用的model，而不是quantized_model，只使用get_mini_peft_model就可以实现我们想要的量化目标了
    # lora_model = get_mini_peft_model(model, peft_config)
    # model = lora_model

    # 这里确保一致，是从要被训练的model获得的优化器
    optimizer =  torch.optim.Adam(model.parameters(),lr=1e-5)

    get_batch_fn = get_batch_function_for_ppo_reward_model(config)

    losses: List = train_model(model, optimizer, num_epochs, num_batches, get_batch_fn,
                                   scheduler=scheduler,
                                   checkpoint_save_path='reward_model_checkpoint',
                                   save_checkpoint_interval_iter_num=800, config=config)



"""
支持正负回复样本对，进行训练的模型已经准备好了
准备好数据就可以了，先从最简单情况开始
在做dpo，训练时，我们已经有了正负样本训练数据，直接使用这个数据就可以训练了
训练数据跟dpo是一样的，只不过在dpo的样本准备中，采用了减少推理次数的方案，把正负样本放在一个batch中
正负样本的推理，一次性完成，减少推理次数
这里的reward model的实现方案，是把正负样本的推理分开两次完成！！！
这里也完全可以采用同样的策略，把正负样本放在一个batch中一次完成推理！！！
---------------
为了学习研究的多样性，这样就保持这种直接解法，更加符合公式！
仅仅需要基于dpo数据获取的get_batch函数，修改为pair形式的
"""
"""
目标：如果要针对训练好的llama使用quetion answer做监督微调，还不使用trl包
自己微调
------
目前我梳理的思路是
【数据】
把数据整理成这样的格式
input_ids
target_ids
跟训练语言模型的x，y保持一样
唯一的差别是target_ids中的question token部分用-100代替

【loss】
计算交叉熵损失时，忽略掉-100的的taget_ids
-100的token不计算tokenloss 做交叉熵累加时，token_id=-100的token做参与计算 token loss

但是是否有必要按照targte_id=-100的情况来确保grad不传递呢？
1.能否做到？
2.能做到的话是否有必要呢？
"""

"""
微调的训练和pretrain的训练基本完全一样
差别在
1. 数据获取
需要新写一个数据获取的函数，返回不再只是返回xy
input_ids, target_ids
还要增加一个 attention mask 

input_ids,target_ids,attention_mask =get_batch_for_sft(split='train')

2. 损失函数
损失函数增加一个 ignore index = -100
已经在原有的model文件中直接修改了
因为，loss不是定义在train函数中，而是直接定义在model文件中了
直接添加的ignore index并不影响pretrain，因为pretrain时并没有-100的target_id
这个特殊的标记id从config中获取
"""

def get_lora_model_with_perf(model):
    from peft import (
        LoraConfig,
        TaskType,
        AutoPeftModelForCausalLM,
        get_peft_model
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,  # config.n_emb是32， 则r比n_emb小，可以设置为8
        lora_alpha=4,  # 经验上设置为r的两倍
        lora_dropout=0.01,
        target_modules=['q_proj']
    )
    print("---------old model-------------")
    print(model)
    new_model = get_peft_model(model, peft_config=peft_config)
    print("---------------new model ----------------")
    print(new_model)
    return new_model


def get_lora_model_with_minilora(model):
    from mylora.minilora import MiniLoraConfig, get_mini_peft_model, LoRALinear
    peft_config = MiniLoraConfig(
        rank=10,
        lora_dropout=0.01,
        lora_alpha=20,
        # , 'v_proj', 'k_proj', 'w_proj', 'ffn_up', 'ffn_down'
        #     target_modules='all_linear'
        target_modules=['q_proj', 'v_proj']
    )
    print("---------old model-------------")
    print(model)
    new_model = get_mini_peft_model(model, peft_config=peft_config)
    print("---------------new model ----------------")
    print(new_model)
    return new_model

from myquantization.myquantize import *
from mylora.minilora import *
def get_qlora_model(model):
    module_names = [
        "word_emb",
        "emb_dropout",
        "layer_norm4attn",
        "q_proj",
        "k_proj",
        "v_proj",

        "attn_dropout",
        "res_dropout",
        "ffn_up",
        "act",
        "ffn_down",
        "dropout",
        "layer_norm4last",
        "lm_head"
    ]
    # ['lm_head']
    quantize_config:QuantizationConfig = QuantizationConfig(precision=4,exclude_modules=module_names)

    new_model = copy.deepcopy(model)

    new_model = set_parameters_requires_grad(new_model)
    quantizer = QuantizationHandler(min_val=-1,max_val=1,n_bits=8)
    quantized_model, has_been_replaced = replace_linear_with_linearkbit(new_model,config=quantize_config,quantizer=quantizer)
    print("------------------quantized model -----------------")
    print(quantized_model)
    for name,param in quantized_model.named_parameters():
        print(f"{param.data.dtype}## {param.requires_grad} ## {name}  ")


    peft_config = MiniLoraConfig(
        rank=2,
        lora_alpha=2,
        target_modules=['q_proj','v_proj'],
        use_qlora=True,
        quantize_modules=[]

    )
    # 其实这里直接用的model，而不是quantized_model，只使用get_mini_peft_model就可以实现我们想要的量化目标了
    lora_model = get_mini_peft_model(model, peft_config)
    print("------------------get lora and quantize model at the same time -----------------")
    print(lora_model)
    for name, param in lora_model.named_parameters():
        print(f"{param.data.dtype}## {param.requires_grad} ## {name}  ")

    return lora_model

from typing import List
from model import GPT,GPTConfig
from mytrain import get_vocab_size,CosineAnnealingWithWarmupScheduler,train_model

def get_new_model_optimizer():
    vocab_size = get_vocab_size()
    batch_size = 2

    config = GPTConfig(vocab_size=vocab_size, block_size=100,
                       n_embd=16, n_head=4,
                       n_layer=1, dropout=0.1,
                       bias=False, ffn_multiply_scalar=1,
                       device='cpu', batch_size=batch_size,
                       decay_lr=True, residual=False,
                       use_moe=False, use_ffn=True, use_glu_only=False, use_mlp_only=True,
                       norm_type='rmsnorm', use_abs_int_pos_emb=False,
                       use_abs_sin_pos_emb=False, use_rope_pos_emb=True, cache_max_batch_size=32)

    model = GPT(config)
    beta1 = 0.9
    beta2 = 0.95
    learning_rate = 6e-4  # 最大学习率
    weight_decay = 1e-1
    device_type = 'cpu'
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    return model,optimizer,config

def load_pretrain_model():
    from mytrain import load_checkpoint
    checkpoint_dir = 'pretrain_checkpoint'
    # 我先不从ckpt加载，看看，原始模型能否训练
    config: GPTConfig
    model, optimizer, iter_num, best_val_loss, config = load_checkpoint(checkpoint_dir)
    config.model_type = 'mygpt'

    config.batch_size = 2  # 变小一点，微调时需要用更少的数据，更小的学习率
    config.block_size = 100

    model.config = config
    return model, config

from data.sft_data.get_batch_train_data import DataLoader
from typing import Callable
def get_batch_function_for_sft()->Callable:
    data_path = 'data/sft_data/data/'
    dataloader = DataLoader(data_path)
    my_get_batch = lambda split: dataloader.get_batch(split=split,config=config)
    return my_get_batch

def merge_lora_and_save(lora_model, sfted_model_dir='sft_checkpoint', merged_model_dir='sft_merge_checkpoint'):
    from mytrain import load_checkpoint,save_checkpoint
    sfted_model, optimizer, iter_num, best_val_loss, config = load_checkpoint(checkpoint_dir=sfted_model_dir,model=lora_model)

    from mylora.minilora import merge_lora_back_to_pertrain_model
    merged_model = merge_lora_back_to_pertrain_model(sfted_model)
    # for name,param in sfted_model.named_parameters():
    #     print(f"name is {name} param is {param.shape}")
    save_checkpoint(merged_model, optimizer, config, iter_num, best_val_loss, out_dir=merged_model_dir)


def visual_backward_gpt(lora_model,config):
    # 可视化一下反向传播图
    input_ids = torch.randint(low=0, high=config.vocab_size-1, size=(1, config.block_size))
    from utils.basic_utils import visualize_model_with_tensorboard, visualize_model
    # 这个不好用
    # visualize_model_with_tensorboard(lora_model,input_ids)
    # 这个好用些，但是也分析不出来为什么
    visualize_model(lora_model,input_ids)

if __name__ == '__main__':
    # 创建调度器实例
    scheduler = CosineAnnealingWithWarmupScheduler(
        learning_rate=1e-4, # 微调时把lr调小一点
        min_lr=6e-5, # 微调时把lr调小一点
        lr_decay_iters=600000,
        warmup_iters=2000
    )

    num_epochs = 1000
    num_batches = 10
    # model,optimizer,config = get_new_model_optimizer()

    # pertrain_model,config = load_pretrain_model()
    # print(config)

    from model import GPT,GPTConfig

    config = GPTConfig(n_layer=1,n_embd=12,n_head=1,residual=True)
    pertrain_model = GPT(config)

    # lora_model = get_lora_model_with_minilora(model=pertrain_model)
    # lora_model = get_lora_model_with_perf(model=pertrain_model)
    lora_model = get_qlora_model(model=pertrain_model)


    lora_model_optimizer = lora_model.configure_optimizers()

    sft_get_batch_data_func = get_batch_function_for_sft()


    print("-------vanilla lora model---------")
    print(lora_model)
    for name, param in lora_model.named_parameters():
        print(f"param dtype : {param.data.dtype}## is_param_require_grad:{param.requires_grad} ## param_name:{name}  ")

    losses: List = train_model(lora_model, lora_model_optimizer, num_epochs, num_batches, sft_get_batch_data_func,
                               scheduler=scheduler,
                               checkpoint_save_path='sft_checkpoint',
                               save_checkpoint_interval_iter_num=800,config=config)

    # 如果用的lora就用下列合并方法，合并lora并且保存
    merge_lora_and_save(lora_model)

    # 如果是使用的qlora，就用下列合并方法
    merged_model = merge_lora_back_to_pertrain_model(lora_model, quantizer)
    # TODO 保存模型






"""
参数名称：base_model.model.transformer.word_emb.weight 
 参数尺寸大小：torch.Size([6860, 16])
参数名称：base_model.model.transformer.attn.0.layer_norm4attn.gamma 
 参数尺寸大小：torch.Size([16])
参数名称：base_model.model.transformer.attn.0.attention.q_proj.base_layer.weight 
 参数尺寸大小：torch.Size([16, 16])
参数名称：base_model.model.transformer.attn.0.attention.q_proj.lora_A.default.weight 
 参数尺寸大小：torch.Size([32, 16])
参数名称：base_model.model.transformer.attn.0.attention.q_proj.lora_B.default.weight 
 参数尺寸大小：torch.Size([16, 32])
参数名称：base_model.model.transformer.attn.0.attention.k_proj.base_layer.weight 
 参数尺寸大小：torch.Size([16, 16])
参数名称：base_model.model.transformer.attn.0.attention.k_proj.lora_A.default.weight 
 参数尺寸大小：torch.Size([32, 16])
参数名称：base_model.model.transformer.attn.0.attention.k_proj.lora_B.default.weight 
 参数尺寸大小：torch.Size([16, 32])
参数名称：base_model.model.transformer.attn.0.attention.v_proj.base_layer.weight 
 参数尺寸大小：torch.Size([16, 16])
参数名称：base_model.model.transformer.attn.0.attention.v_proj.lora_A.default.weight 
 参数尺寸大小：torch.Size([32, 16])
参数名称：base_model.model.transformer.attn.0.attention.v_proj.lora_B.default.weight 
 参数尺寸大小：torch.Size([16, 32])
参数名称：base_model.model.transformer.attn.0.attention.w_proj.base_layer.weight 
 参数尺寸大小：torch.Size([16, 16])
参数名称：base_model.model.transformer.attn.0.attention.w_proj.lora_A.default.weight 
 参数尺寸大小：torch.Size([32, 16])
参数名称：base_model.model.transformer.attn.0.attention.w_proj.lora_B.default.weight 
 参数尺寸大小：torch.Size([16, 32])
参数名称：base_model.model.transformer.attn.0.ffn.ffn_up.weight 
 参数尺寸大小：torch.Size([16, 16])
参数名称：base_model.model.transformer.attn.0.ffn.ffn_down.weight 
 参数尺寸大小：torch.Size([16, 16])
参数名称：base_model.model.transformer.layer_norm4last.weight 
 参数尺寸大小：torch.Size([16])
"""

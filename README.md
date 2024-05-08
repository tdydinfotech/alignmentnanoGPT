# alignmentnanoGPT

这是一个可以快速在mac上训练gpt并微调，对齐的项目，无需GPU！

### 有什么好处呢？

###### 快速验证新想法

- 如果新想法不需要涌现后的llm就可以验证，可以快速使用mac开发新想法做实验验证！
  
- 如果新想法需要涌现后的llm进行验证，可以快速使用mac开发，确保在小参数情况下，loss正确下降，然后再迁移到GPU上进行训练！
  

###### 学习成本低

由于项目没有使用高级封装的工具，比如huggingface的transformers或者trl高级库，主要使用pytorch实现，所以可以清晰看到llm构建的全流程代码，这样有利于学习，方便按照自己的新想法快速进行实验！包括预训练，微调和对齐环节！

## 安装

pip install torch numpy datasets tqdm

依赖项:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `datasets` 用于huggingface datasets <3 （如果您想下载+预处理OpenWebText）
- `tqdm` 用于进度条 <3

## 快速开始

首先，我们把用于训练文本文件中文文本，都映射成整数序列，这样可以作为模型的输入，进行计算，转化为整数的文件保存在train.bin和val.bin中；

train.bin用于训练，val.bin用计算验证集loss

计算资源：

**一台Macbook**（或其他pc电脑）。我建议获取最新的PyTorch nightly版本（在安装时[在这里选择](https://pytorch.org/get-started/locally/)），因为当前这个版本很可能会使您的代码更有效。

### 预训练

- **训练**：直接进入mytrain.py 就可以运行pretrian预训练一个基础小型llm
  

GPT模型定义，模型所需要参数定义GPTconfig都在model.py文件中

   `备注（设置device=cpu`和 compile=False`关闭PyTorch 2.0编译。其中config中block_size就是seq_len代表句子长度）`

- **推理**：进入mygenerate.py ，其中实现了推理的全量推理(kvcache预填充)和增量推理(使用kvcache做单token推理)
  
- **数据**：在data/pretrain_data/train.bin 和 val.bin； 把文本数据转化为训练所需的整数序列在prepare.py脚本中
  
- **模型** :训练好的模型保存在pretrain_checkpoint/myckpt.pt
  

### sft微调

- **训练**：进入mysft.py ，其中实现了lora微调和qlora微调
  
  - 其中所涉及的lora，是参考perf自己实现的简单lora，在mylora.minilora.py中
    
  - 其中qlora所涉及的量化，是参考bitsandbytes自己实现，在myquantization.myquantize.py中
    
- **数据**：微调所用数据，在data/sft_data/data中，把文本数据转为sft所需要的格式的脚本在data/sft_data/下的py脚本中
  
  - 数据所需格式 {"prompt":"你好啊","output":"见到你真好"}
    
  - 训练时，把数据转化成inputs = prompts + output
    
  - 我所用的示例数据是 BelleGroup/train_0.5M_CN 已经从hugging face下载好
    
- **模型**：训练好的模型保存在sft_checkpoint/myckpt.pt中
  

### DPO对齐

- **训练**：dpo_training.py 文件中，由于dpo的训练和sft类似都是对模型的微调，所以也可以使用lora，qlora进行训练，对比看效果
  
  - 需要使用两个模型，一个是需要微调的模型，一个是ref model
    
  - 可以用需要微调的模型的备份作为ref model，不需要训练仅需要做推理，也可以尝试对ref model做量化，加速训练
    
- **数据**：在data/dpo_data/data/dpo_train.bin 和 dpo_val.bin 数据处理脚本也在这个目录下；
  
  - dpo所需数据是
    
    {"prompt":"你好","chosen":"见到你真高兴","rejected" :"不想看见你"}
    
  - 训练时，可以把数据转化为两条数据，
    
    inputs_chosen = prompt+chosen
    
    inputs_rejected=prompt+rejected
    
  - 我所使用的示例是知乎的在huggingface上的数据，已经下载下来了，zhihu_rlhf_3k
    
- **模型**：保存在dpo_checkpoint/myckpt.pt中
  

### PPO对齐

- **奖励模型**
  
  - **数据**：依然使用dpo中所用的相同数据；但为了学习需要，在数据处理上和dpo略有不同，比如在dpo中把需要ref model推理的 接受和被拒绝的样本放在一个batch中，通过奇数偶数index做区分，这样做是为了提高效率；而在reward model这里，是分两次做推理，所以把被接受的input放在一起，被拒绝的inputs放在一起；在data/dpo_data/get_batch_data.py中有个get_batch_for_ppo_reward_model函数，是为reward model训练获取数据服务的！
    
  - **训练**：使用reward_model_training.py 训练奖励模型！需要注意的是其中为了减少过拟合问题，同时增加了chosen回复生成的loss作为平衡
    
- **数据**：数据是通过agent与env做交互，采样生成得到的，不同于sft和dpo的监督学习需要额外的数据，ppo的训练所需的数据是需要对齐的模型生成出来的！
  
- **训练**：在ppo_training.py文件中，定义了CriticModel价值模型和EnvWithRewardModel环境
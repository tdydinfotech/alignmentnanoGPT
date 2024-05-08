"""
ppo training
"""
import torch
from utils import rl_train_proccess # 假设这是一个自定义的实用程序模块
import copy
import torch.nn.functional as F
from model import GPT,GPTConfig
from torch.nn import Linear,Module
from reward_model_training import RewardModel


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, policy_model,ref_model,critic_model, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        # 初始化actor和critic网络
        self.actor = policy_model
        self.critic = critic_model
        self.ref_model = ref_model
        # 定义actor和critic的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        # 设置超参数
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 每个序列的训练轮数
        self.eps = eps  # PPO截断参数
        self.device = device

    def take_action(self, state):
        # 将状态转换为张量并移动到设备上
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        state = state.to(self.device)

        # 从actor网络获取动作概率
        action_logits,_ = self.actor(state)
        action_prob = torch.nn.functional.softmax(action_logits,dim=-1)
        # 从分布中采样动作
        action_dist = torch.distributions.Categorical(probs=action_prob)
        action = action_dist.sample()
        # 还是一样的，这个依然是一个随机性策略，采样出来一个动作，而不是取概率最大的策略
        action = action.unsqueeze(-1)
        return action

    def update(self, transition_dict):
        # 将转换字典转换为张量并移动到设备上
        # states中每个state元素是变长的，batch_size,seq_len 这里的seq_len是变长的
        # 需要把他们变成 一个新的 batch_size,seq_len的tensor，那就需要在batch_size
        # 维度上做拼接，
        data = transition_dict
        states = rl_train_proccess.pad_variable_length_sequences(states=data['states'])
        next_states = rl_train_proccess.pad_variable_length_sequences(states=data['next_states'])
        actions = rl_train_proccess.reshape_reward_and_action(data['actions'])
        rewards = rl_train_proccess.reshape_reward_and_action(data['rewards'])
        dones = rl_train_proccess.reshape_reward_and_action(data['dones'])

        # 计算TD_target, TD_error之前，需要先计算V(s_t) 和 V(S_t+1)
        # 为什么要先提前计算呢？因为还可以在critic net中复用呢，计算出critic loss
        reward_predict_of_s_t = self.critic(states)
        reward_predict_of_s_t_plus_one = self.critic(next_states)
        # 计算TD目标
        # 这里也很重要，这样做是为了防止在一回合结束时，溢出，1-done就可以避免溢出了
        td_target= rewards + self.gamma * reward_predict_of_s_t_plus_one * (1-dones)
        # 计算优势估计
        td_delta = td_target - reward_predict_of_s_t
        advantage = rl_train_proccess.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        # 计算动作的旧对数概率
        # 问题：如果这类用当前actor网络作为旧网络，把是否还做更新呢？难道永远不做更新吗？
        # 这是怎么回事呢？我觉得这个网络在策略网络更新了一段时间后可以更新一下啊
        # 并不是，因为在训练过程中，每次调用update函数，都会先使用当前动作网络函数计算动作的对数概率
        # 然后，再去更新网络，所以新旧动作网络的差别只在于是否使用了最新的采集而来的数据做参数更新
        # -------------------------------
        # old_log_probs = self.actor(states).log()
        # 这么写是不对的，是要所采取动作的，动作概率，要从动作概率分布中先查找出动作的概率值p再取log
        with torch.no_grad():
            old_probs_logits,_ = self.ref_model(states)
            old_probs = torch.nn.functional.softmax(old_probs_logits,dim=-1)
            old_action_prob = old_probs.gather(1,actions)
            old_log_prob =  old_action_prob.log()


        """
        为啥单独写一个old_log_prob.detach()就无法训练呢？
        因为我们要求，旧的策略是作为参考，不做更新的，detach()函数会返回一个新的tensor，一个脱了了计算图的tensor
        因为，可以做到不做梯度更新，只做前向计算，但是我实现时，没有用到这个脱离了计算图的tensor，还是会做梯度更新，
        错误的做法是，单独写一个语句 old_log_prob_tmp.detach()
        正确的做法是，old_log_prob = old_log_prob_tmp.detach()
        这个是真正的原因，而不是a.log VS torch.log(a)
        这二者之间没有差别，都不会阻断梯度的
        """

        # 这一步也超级关键，我们仅仅需要这个q分布做重要性采样，不需要更新梯度，所以detach()放在这里太关键了
        # old_log_prob.detach()

        # old_log_prob = torch.log(self.actor(states).gather(1, actions)).detach()
        """
        # 我就先尝试下，不复用数据的情况
        对比的结论显而易见，没有复用数据的方式，奖励上升速度明显慢很多
        即便是反复复用数据，不但没有导致效果变差，反而极大的提升了收敛速度
        所以，我猜测，off-policy是完全可以的
        """
        for _ in range(self.epochs):
            logits,_ = self.actor(states)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            specific_action_probs = probs.gather(1, actions)
            # 计算动作的新对数概率
            log_probs = torch.log(specific_action_probs)
            # 计算重要性采样比率
            ratio = torch.exp(log_probs - old_log_prob)
            # 计算替代损失
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio,1-self.eps,1+self.eps) * advantage
            # 注意既然变成了loss就要改成负数形式，目标是要让reward越来越大，这个公式本质是计算的reward期望最大化
            loss = -torch.min(surr1,surr2)
            # 计算actor的PPO损失
            actor_loss = torch.mean(loss)
            # 计算critic损失
            # 注意这里很重要，mse后也要mean
            critic_loss = torch.mean(F.mse_loss(input=self.critic(states),target=td_target.detach()))
            # print(f"actor loss {actor_loss},critic_loss {critic_loss}")
            # 清空梯度
            # 为什么这里要清掉梯度呢？而reinforce里不清零，反而累加呢？
            # 因为这里本来就是要用一个时间步数据更新参数，而不需要等到整个回合结束的，这也是优势
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # 反向传播并更新权重
            # 这里是loss执行backward，而不是用actor执行backward
            try:
                # 执行反向传播
                actor_loss.backward()
                critic_loss.backward()
            except RuntimeError as e:
                # 异常处理代码块
                print("---异常时的输入是：---------------------")
                self.actor(states)
                print(f'states:{states} \n action:{actions} \n next_states:{next_states} \n rewards:{rewards} ')
                raise e
                # 可以在这里添加你的处理逻辑，例如重新初始化网络参数、调整输入数据等

            # backward是网络本身的事，w=w+grad是优化器的事，
            # 优化器怎么会知道w在哪，grad又在哪？？？？
            # 其实是定义优化器时候已经把参数给到了优化器
            # print(self.critic.parameters())
            self.actor_optimizer.step()
            self.critic_optimizer.step()

class CriticModel(Module):
    def __init__(self,model:GPT,config:GPTConfig):
        super().__init__()
        self.model = model
        self.reward_predict_linear = Linear(in_features=config.vocab_size,
                                            out_features=1,bias=False)
    def forward(self, input_ids=None, target_ids=None, attention_mask:torch.Tensor=None, *args, **kwargs):
        logits,_ = self.model.forward(input_ids=input_ids,target_ids=target_ids,attention_mask=attention_mask)
        reward_predict = self.reward_predict_linear(logits)
        return reward_predict

class EnvWithRewardModel(object):
    def __init__(self,batch_size=2):
        self.reward_model = self.load_reward_model()
        self.batch_size = self.config.batch_size if not batch_size else batch_size
        self.next_state = torch.zeros(self.batch_size,0,dtype=torch.long)
    def load_reward_model(self):
        model,config = load_sft_model()
        reward_model = RewardModel(model=model,config=config)
        #  实现加载奖励模型，确保，输入action 输出reward预估值

        from mytrain import load_checkpoint
        checkpoint_dir = 'reward_model_checkpoint'
        # 我先不从ckpt加载，看看，原始模型能否训练
        config: GPTConfig
        model, optimizer, iter_num, best_val_loss, config = load_checkpoint(checkpoint_dir,model=reward_model)
        config.model_type = 'mygpt'

        # config.batch_size = 2  # 变小一点，微调时需要用更少的数据，更小的学习率
        # config.block_size = 100

        model.config = config

        self.config = config
        # reward_model = RewardModel(model,config)
        reward_model = model
        return reward_model

    def step(self,action):
        reward = self.reward_model.reward_forward(action)
        # TODO 从tokenizer里获取这个终止符号,并且应该逐句判断
        # action is (batch_size,1) shape so,we should judge action is done or not for every item in batch
        done = action == "EOS_TOKEN"
        done = self.next_state.shape[1] % 10 == 0
        if done:
            done = torch.ones_like(action)
        else :
            done = torch.zeros_like(action)
        info = ''
        self.next_state = torch.cat((self.next_state,action),dim=1)
        return self.next_state,reward, done, info
    def reset(self):
        # TODO 这里用该获得一个句子启始符号START_OF_SENTENCE的token_id
        # (batch_size,seq_len)
        seq_len = 1
        state = torch.randint(low=0,high=self.config.vocab_size,size=(self.batch_size,seq_len),dtype=torch.long)
        self.next_state = torch.cat((self.next_state,state),dim=1)
        return state


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    actor_lr = 1e-4
    critic_lr = 1e-4
    num_episodes = 80
    gamma = 0.98
    lmbda = 0.95
    num_epochs = 5
    eps = 0.4
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")


    from dpo_training import load_sft_model
    sfted_model,config = load_sft_model()
    policy_model = sfted_model
    critic_model = CriticModel(model=copy.deepcopy(sfted_model),config=config)
    ref_model = copy.deepcopy(policy_model)
    env = EnvWithRewardModel(batch_size=4)


    agent = PPO(policy_model,ref_model,critic_model, actor_lr, critic_lr, lmbda,
                num_epochs, eps, gamma, device)

    return_list = rl_train_proccess.train_on_policy_agent(env, agent, num_epochs,num_episodes)



    env_name = 'ppo-llm'
    rl_train_proccess.plot_reward_curve(return_list, env_name)

    print(policy_model)





    # policy model 的forward(states)->action动作空间概率分布
    # critic model 的forward(states)->reward返回未来奖励的预估值
    # env.step(action)->返回reard model的奖励值和下一个状态 next state

    # states就是当前生成的句子的token_ids，action就是所采用的token_id
    #  准备好下列四个模型，确保输入输出符合要求
    #  对于policy model需要确保输入states输出动作的logits
    #  对于critic_model需要确保输入states输出的是奖励值预估值，
    #  对于reward model同样需要确保输入action，输出奖励值
    """
    states is current input_ids
    action logits is next token logits 
    action is next token id 
    for cirtic model I just need to add one more linear to logits ,convert to 1 value
    for reward model I need to add one more linear to logits,reward model need
    action input, actually it is token_id as input, the output is one value
    我最大的困惑是这么做的话，按照逐个token去探索，不是太慢了吗，怎么能加快速度呢 ？         
    """
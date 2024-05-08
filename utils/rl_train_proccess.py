from tqdm import tqdm
import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

from ppo_training import PPO,EnvWithRewardModel
def train_on_policy_agent(env:EnvWithRewardModel, agent:PPO, num_epochs,num_episodes):
# def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(num_epochs):
        with tqdm(total=int(num_episodes / num_epochs), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / num_epochs)):
                episode_return = []
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = torch.tensor([0])
                while torch.any(done==0):
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward.sum(dim=-1).tolist()
                mean_reward = torch.tensor(episode_return).mean()
                return_list.append(mean_reward)
                agent.update(transition_dict)
                if (i_episode ) % 10 == 0:

                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % mean_reward.item()})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def plot_reward_curve(return_list,env_name):
    import matplotlib.pyplot as plt
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()


from torch.nn.utils.rnn import pad_sequence
def reshape_reward_and_action(actions):
    actions = torch.cat(actions,dim=0)
    actions = actions.view(-1,1)
    return actions

def pad_variable_length_sequences(states):
    # 找到所有序列中的最大长度
    max_seq_len = max([state.shape[1] for state in states])

    # 对每个序列进行填充，使其长度达到最大长度
    padded_states = [
        torch.cat([state, torch.zeros((state.shape[0], max_seq_len - state.shape[1]), dtype=state.dtype)], dim=1) for
        state in states]

    # 使用pad_sequence函数将填充后的序列组合成一个张量
    padded_states_tensor = pad_sequence(padded_states, batch_first=True)
    padded_states_tensor = padded_states_tensor.view(-1,max_seq_len)

    return padded_states_tensor

if __name__ == '__main__':
    import torch

    data = {
        'states': [torch.tensor([[4890], [3790]]),
                   torch.tensor([[4890, 2215], [3790, 3112]]),
                   torch.tensor([[4890, 2215, 454], [3790, 3112, 0]]),
                   torch.tensor([[4890, 2215, 454, 6827], [3790, 3112, 0, 6827]])],

        'actions': [torch.tensor([[2215], [3112]]),
                    torch.tensor([[454], [0]]),
                    torch.tensor([[6827], [6827]]),
                    torch.tensor([[1425], [2329]])],

        'next_states': [torch.tensor([[4890, 2215], [3790, 3112]]),
                        torch.tensor([[4890, 2215, 454], [3790, 3112, 0]]),
                        torch.tensor([[4890, 2215, 454, 6827], [3790, 3112, 0, 6827]]),
                        torch.tensor([[4890, 2215, 454, 6827, 1425], [3790, 3112, 0, 6827, 2329]])],

        'rewards': [torch.tensor([[-3.8349], [-4.8746]]),
                    torch.tensor([[-5.1317], [-5.0705]]),
                    torch.tensor([[-5.1066], [-4.8888]]),
                    torch.tensor([[-4.3419], [-4.5836]])],

        'dones': [False, False, False, True]
    }
    states = pad_variable_length_sequences(states=data['states'])
    next_states = pad_variable_length_sequences(states=data['next_states'])
    actions = reshape_reward_and_action(data['actions'])
    rewards = reshape_reward_and_action(data['rewards'])
    print(states)
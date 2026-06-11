import sys, os
# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取父目录的父目录（上两级目录）
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
import typing
import logging
import traceback
import numpy as np
import time

import torch 
from torch import multiprocessing as mp

from guandan_rlcard.baselines.danzero_plus.env_utils import Environment, action_vector
from guandan_rlcard.envs.guandan_env import GuandanEnv
from guandan_rlcard.baselines.danzero_plus.ppo_agent import PPOAgent

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_env(flags):
    return GuandanEnv(flags.objective)

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers # 此处 key 即'done', 'episode_return', 'target', 'obs_x'...
    }
    for m in indices:
        # ​​资源释放​​：将已使用的索引重新放入 free_queue，供后续数据填充时复用，形成循环缓冲区，避免频繁内存分配的开销
        free_queue.put(m)
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['0', '1', '2', '3']
    optimizers = {}
    # for position in positions:
    #     optimizer = torch.optim.RMSprop(
    #         learner_model.parameters(position),
    #         lr=flags.learning_rate,
    #         momentum=flags.momentum,
    #         eps=flags.epsilon,
    #         alpha=flags.alpha)
    #     optimizers[position] = optimizer

    for position in positions:
        optimizer = torch.optim.Adam(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            betas=(0.9, 0.999),  # Adam 特有的超参数
            eps=flags.epsilon,
            weight_decay=0.0  # 可选，L2 正则化
        )
        optimizers[position] = optimizer

    return optimizers

def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length # 时序展开长度，决定缓冲区时间步维度（如LSTM训练需展开多步状态）
    positions = ['0', '1', '2', '3']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 734
            # 所有字段以 T 为时间轴，支持时序模型训练
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 83), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 216), dtype=torch.float32),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    # ​torch.empty ​仅分配内存空间而不填充初始值​​，因此在需要快速分配内存且后续会覆盖数据的场景中性能更优；
                    # share_memory_ 共享内存允许进程/线程直接读写同一块物理内存，避免数据复制开销
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['0', '1', '2', '3']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        random = np.random.RandomState()
        # 自博弈训练
        # env.set_agents([DMCAgent(0, random, device=str(device)), DMCAgent(1, random, device=str(device)), DMCAgent(2, random, device=str(device)), DMCAgent(3, random, device=str(device))])
        env.set_agents([PPOAgent(0, random, device=str(device)), PPOAgent(1, random, device=str(device)), PPOAgent(2, random, device=str(device)), PPOAgent(3, random, device=str(device))])
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}


        position, obs, state, env_output = env.initial()
        position = str(position)

        while True:
            # 开一小局
            while True:
                # agent 决策动作
                action = env.env.agents[int(position)].step(state, model, training=True)
                

                if action != []: # action为[]是某agent即将接风的情况，该情况不记录
                    # ppo 的话，要不要考虑不记录只能过牌的场景呢？
                    # 尝试不记录只能过牌的场景

                    obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                    obs_z_buf[position].append(env_output['obs_z'])
                    obs_action_buf[position].append(action_vector(int(position), action))

                    size[position] += 1
                if len(obs_x_no_action_buf[position]) != len(obs_action_buf[position]) or len(obs_action_buf[position]) != len(obs_z_buf[position]):
                    raise ValueError('no_action != action')                    

            

                # 动作执行
                position, obs, state, env_output = env.step(action)
                position = str(position)
                
                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p]) # 用来将回报补充到和过程动作一样的长度
                        if diff > 0:

                            now_done = [False for _ in range(diff-1)]
                            now_done.append(True)

                            episode_return = env_output['episode_return'] if p in [position, str((int(position)+2)%4)] else -env_output['episode_return']
                            episode_return = episode_return / 3 

                            episode_returns = [0.0 for _ in range(diff-1)]
                            episode_returns.append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)]) # 每一步的target都为总回报

                            done_buf[p].extend(now_done)  
                            episode_return_buf[p].extend(episode_returns) 

                            # print(f"进程{i}采样一小局结束，小局结果{env.env.game.temp_result}， 本位置{p}奖励 {episode_return}")
                            if len(obs_x_no_action_buf[p]) != len(target_buf[p]) or len(obs_x_no_action_buf[p]) != len(target_buf[p]):
                                raise ValueError('no_action != target')
                    
                    break

            # 如果是大局结束，重置环境
            if env.is_over():
                # print('重置环境！')
                position, obs, state, env_output = env.initial()
                position = str(position)

            for p in positions:  
                # print(size)
                # exit()
                # 实际上每局数据相当于混杂在一起，因为每次取的 T 步数据可能跨越几局
                # （但是这样lstm的学习会不会有问题？因为状态在两局之间产生了突变）
                while size[p] > T: 
                    # 如果 free_queue为空，进程会阻塞在此步
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        # ... 表示​​自动补全所有未显式指定的维度
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]


                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T # 存到缓冲区去了，相当于消费掉了T步数据


    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


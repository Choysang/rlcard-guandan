import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_buffers, create_optimizers, act
import wandb
from compete_list import eval_agent_dict
from .eval import run_evaluation

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['0', '1', '2', '3']}

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss

def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """Performs a learning (optimization) step."""
    # batch = (batch_size, T, num_legal_actions, (54+54+...+17+15))
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    # flatten 将 batch_size 与其他维度合并，例如将 (batch, seq_len, features) 合并为 (batch*seq_len, features)，便于全连接层处理
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']] # 只取出几局的最后一步
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
        
    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return_'+position: torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_'+position: loss.item(),
        }
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def train(flags):  
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    
    # logger = wandb.init(
    #     project='guandan_RL_restart',
    #     name='dmc',
    # )
    
    checkpointpath = os.path.expandvars( # os.path.expandvars 用于替换路径中的环境变量引用：例如，若路径中包含环境变量 $SAVEDIR，则会被替换为其值
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar'))) # os.path.expanduser 用于展开路径中的 ~ 符号

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    # 主要用来存储索引，据此索引取共享缓冲区中的数据
    free_queue = {} # 空闲索引
    full_queue = {} # 存有未学习数据的索引
        
    for device in device_iterator:
        _free_queue = {'0': ctx.SimpleQueue(), '1': ctx.SimpleQueue(), '2': ctx.SimpleQueue(), '3': ctx.SimpleQueue()}
        _full_queue = {'0': ctx.SimpleQueue(), '1': ctx.SimpleQueue(), '2': ctx.SimpleQueue(), '3': ctx.SimpleQueue()}
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_0',
        'loss_0',
        'mean_episode_return_1',
        'loss_1',
        'mean_episode_return_2',
        'loss_2',
        'mean_episode_return_3',
        'loss_3',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'0':0, '1':0, '2':0, '3':0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for k in ['0', '1', '2', '3']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")
        print('从本地启动')

    # Starting actor processes
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        # nonlocal ​允许嵌套函数访问并修改其外层函数（非全局作用域）的局部变量​​。
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], flags, local_lock)
            # batch = (batch_size, T, num_legal_actions, (54+54+...+17+15))
            _stats = learn(position, models, learner_model.get_model(position), batch, 
                optimizers[position], flags, position_lock)  

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                logger.log(to_log, step=frames)
                # print(f'model[{position}] learned, wandb logged at frames: {frames}')
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator: # 缓冲区初始化为空
        for m in range(flags.num_buffers):
            free_queue[device]['0'].put(m)
            free_queue[device]['1'].put(m)
            free_queue[device]['2'].put(m)
            free_queue[device]['3'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {'0': threading.Lock(), '1': threading.Lock(), '2': threading.Lock(), '3': threading.Lock()}
    position_locks = {'0': threading.Lock(), '1': threading.Lock(), '2': threading.Lock(), '3': threading.Lock()}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['0', '1', '2', '3']:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i, device, position, locks[device][position], position_locks[position]))
                thread.start()
                threads.append(thread)
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)
     
        # 只存0号位模型参数
        model_weights_dir = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid, '0_weights_'+str(frames)+'.ckpt')))
        torch.save(learner_model.get_model('0').state_dict(), model_weights_dir)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        last_eval_time = timer() - flags.save_interval * 30
        
        level_list = ['0', '1', '2', '3'] # 对手池难度
        level_index = 2
        ep_win = [0, 0]
    
        while frames < flags.total_frames:
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60: # 每1个小时保存一次
                checkpoint(frames)
                last_checkpoint_time = timer()

            if timer() - last_eval_time > flags.save_interval * 30: # 每半个小时测试一次
                break
                last_eval_time = timer()
                print('开始测试')   # 和当前对手池每个规则打20局测胜率
                eval_agent_list = eval_agent_dict[level_list[level_index]] # 选定对手池
                
                for id, agent in enumerate(eval_agent_list):
                    print(agent.name)
                    win1, win2 = run_evaluation(agent, 20, learner_model.models['0'])
                    ep_win[id] = 0.2 * ep_win[id] + 0.8 * win1
                    # wandb.log({
                    #     f"eval_{level_list[level_index]}/ep_winrate_{agent.name}": win1,
                    #     f"eval_{level_list[level_index]}/game_winrate_{agent.name}": win2,
                    # }, step=frames)
                print(ep_win)
                
                if ep_win[0] > 0.85 and ep_win[1] > 0.85:
                    # 测试对手池难度升级
                    print('难度升级！')
                    level_index += 1
                    ep_win = [0, 0]

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()

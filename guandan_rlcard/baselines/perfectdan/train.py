"""
掼蛋PPO训练 - 文件系统版
使用文件系统共享数据，解决Windows多进程序列化问题
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
import threading
import multiprocessing as mp
import time
import logging
from datetime import datetime
import json

# 导入必要组件
#获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
#获取父目录的父目录(上两级目录)
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
from guandan_rlcard.envs.guandan_env import GuandanEnv
from guandan_rlcard.baselines.perfectdan.ppo_agent import PPOGuandanAgent, RewardShaper, PPOMemory
from guandan_rlcard.baselines.perfectdan.models import GuandanLSTMPolicyNetwork, GuandanValueNetwork, OptimizedGuandanStateEncoder

# 导入规则对手模型用于评估
from guandan_rlcard.baselines.random_agent import RandomAgent
from guandan_rlcard.baselines.rule_based.base1.base1_agent import Base1Agent
from guandan_rlcard.baselines.rule_based.base5.base5_agent import Base5Agent

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PPOTrainer")

# 全局停止信号
STOP_SIGNAL = False

# 独立进程函数 - 工作进程
def run_worker_standalone(worker_id, models_dir, experience_dir, config_path):
    """完全独立的工作进程函数"""
    # 配置日志
    logger = logging.getLogger(f"Worker-{worker_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 设置种子
    seed = config.get('seed', 42) + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建独立的工作进程实例
    worker = StandaloneWorker(
        worker_id=worker_id,
        models_dir=models_dir,
        experience_dir=experience_dir,
        config=config,
        logger=logger
    )
    
    # 运行工作循环
    try:
        worker.run()
    except Exception as e:
        logger.error(f"工作进程错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

# 独立进程函数 - 学习进程
def run_learner_standalone(learner_id, models_dir, experience_dir, config_path):
    """完全独立的学习者进程函数"""
    # 配置日志
    logger = logging.getLogger(f"Learner-{learner_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 设置种子
    seed = config.get('seed', 42) + 500 + learner_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建独立的学习者实例
    learner = StandaloneLearner(
        learner_id=learner_id,
        models_dir=models_dir,
        experience_dir=experience_dir,
        config=config,
        logger=logger
    )
    
    # 运行学习循环
    try:
        learner.run()
    except Exception as e:
        logger.error(f"学习进程错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

class StandaloneWorker:
    def __init__(self, worker_id, models_dir, experience_dir, config, logger=None):
        self.worker_id = worker_id
        self.models_dir = models_dir
        self.experience_dir = experience_dir
        self.config = config
        self.verbose = (worker_id == 0)
        
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(f"Worker-{worker_id}")
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"工作进程初始化中...")
        
        seed = config.get('seed', 42) + worker_id
        self.set_seed(seed)
        
        # 设置设备 - 直接使用分配到的设备
        if torch.cuda.is_available():
            # 由于 CUDA_VISIBLE_DEVICES, 'cuda:0' 会指向正确的物理GPU
            self.device = 'cuda:0' 
        else:
            self.device = 'cpu'
        self.logger.info(f"工作进程使用设备: {self.device} (物理GPU由CUDA_VISIBLE_DEVICES决定)")
        
        self.env = self.create_environment()
        self.agents = self.create_agents()
        self.env.set_agents(self.agents)
        
        self.current_version = -1
        self.stats = {"episodes": 0, "steps": 0, "wins_team0": 0, "wins_team1": 0}
        self.experience_batch_id = 0
    
    def set_seed(self, seed):
        """设置随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def create_environment(self):
        """创建掼蛋环境，添加observe方法支持"""
        env = GuandanEnv({
            'seed': self.config.get('seed', 42) + self.worker_id,
            'allow_step_back': True
        })
        
        # 修改环境的step方法，支持observe调用
        original_step = env.step
        
        def step_with_observe(self, action, use_raw=False):
            """Modified step method that supports observe call"""
            try:
                # Get current player
                player_id = self.game.get_player_id()
                agent = self.agents[player_id]
                
                # Execute action
                result = original_step(action, use_raw)
                
                # Handle the result based on the expected format
                if isinstance(result, tuple) and len(result) == 2:
                    next_state, next_player_id = result
                else:
                    self.logger.error(f"Unexpected result format from step: {result}")
                    return result
                
                # Notify the previous player to observe the new state
                if hasattr(agent, 'observe'):
                    agent.observe(next_state)
                
                return result
            except Exception as e:
                self.logger.error(f"Error in step_with_observe: {e}")
                # Return a sensible default or re-raise
                raise
        
        # 替换环境的step方法
        import types
        env.step = types.MethodType(step_with_observe, env)
        
        return env
    
    def create_agents(self):
        """创建智能体"""
        agents = []
        
        for player_id in range(4):
            agent = PPOGuandanAgent(
                player_id=player_id,
                np_random=np.random.RandomState(self.config.get('seed', 42) + self.worker_id + player_id),
                state_dim=673,
                action_dim=143,
                abstract_action_dim=336,
                perfect_info_dim=165,
                device=self.device,
                verbose=(self.worker_id == 0)
            )
            
            # 设置训练模式
            agent.train_mode = True
            
            # 设置奖励整形器
            agent.reward_shaper = RewardShaper(
                play_diff_factor=self.config.get('play_diff_factor', 0.1),
                win_reward=self.config.get('win_reward', 2.0),
                lose_reward=self.config.get('lose_reward', -2.0),
                fine_tuning=self.config.get('fine_tune', False),
                verbose=True  # 确保为True
            )
            
            agents.append(agent)
        
        # 输出调试信息
        if self.worker_id == 0:  # 只在worker 0上输出调试信息
            self.logger.info(f"Worker {self.worker_id} 创建了 {len(agents)} 个智能体")
            self.logger.info(f"Player 0 verbose: {agents[0].verbose}")
            self.logger.info(f"Player 0 train_mode: {agents[0].train_mode}")
            self.logger.info(f"Player 0 reward_shaper.verbose: {agents[0].reward_shaper.verbose}")
        
        return agents
    
    def update_agent_parameters(self):
        """更新智能体参数 - 从文件加载"""
        # 检查版本文件
        version_file = os.path.join(self.models_dir, "version.txt")
        if not os.path.exists(version_file):
            return False
        
        # 读取当前版本
        with open(version_file, 'r') as f:
            latest_version = int(f.read().strip())
        
        if latest_version > self.current_version:
            self.logger.debug(f"更新参数版本: {self.current_version} -> {latest_version}")
            
            # 加载模型
            models_path = os.path.join(self.models_dir, f"models_v{latest_version}.pt")
            if os.path.exists(models_path):
                try:
                    state_dict = torch.load(models_path, map_location=self.device)
                    
                    # 更新每个智能体
                    for agent in self.agents:
                        if f'policy_player{agent.player_id}' in state_dict:
                            agent.policy_network.load_state_dict(state_dict[f'policy_player{agent.player_id}'])
                        
                        if f'value_player{agent.player_id}' in state_dict:
                            agent.value_network.load_state_dict(state_dict[f'value_player{agent.player_id}'])
                    
                    self.current_version = latest_version
                    return True
                except Exception as e:
                    self.logger.error(f"加载模型时发生错误: {e}")
            
        return False
    
    def save_experiences(self, player_id, memory):
        """将一个PPOMemory对象的内容保存到文件"""
        if len(memory) == 0:
            return

        # 为每批经验创建唯一文件名
        timestamp = int(time.time() * 1000)
        # 文件名包含player_id，便于Learner区分
        file_name = f"exp_{self.worker_id}_{player_id}_{timestamp}_{self.experience_batch_id}.pt"
        file_path = os.path.join(self.experience_dir, file_name)

        # 从memory对象中提取所有数据
        data_to_save = {
            # 注意: memory.size 是实际存储数量
            'states': memory.states[:memory.size],
            'perfect_info': memory.perfect_info[:memory.size] if memory.perfect_info is not None else None,
            'actions': memory.actions[:memory.size],
            'log_probs': memory.log_probs[:memory.size],
            'vals': memory.vals[:memory.size],
            'rewards': memory.rewards[:memory.size],
            'dones': memory.dones[:memory.size],
            'raw_states': memory.raw_states[:memory.size],
            'action_features': [memory.action_features_data.get(i, []) for i in range(memory.size)]
        }

        # 保存到文件
        try:
            torch.save(data_to_save, file_path)
            self.experience_batch_id += 1
        except Exception as e:
            self.logger.error(f"保存经验时发生错误: {file_path}, {e}")
    
    def process_episode_experiences(self, state_histories, winner_team):
        """
        处理一局游戏的经验并保存
        
        参数:
        state_histories: 游戏状态历史
        winner_team: 获胜队伍
        
        返回:
        experiences_count: 处理的经验数量
        """
        # 收集所有玩家的经验
        all_experiences = []
        
        for agent in self.agents:
            experiences = agent.process_experiences(agent.step_history, winner_team)
            all_experiences.extend(experiences)
        
        # 保存经验到文件
        if all_experiences:
            self.save_experiences(all_experiences)
        
        return len(all_experiences)
    
    def run(self, num_episodes=None):
        """运行工作进程收集经验"""
        global STOP_SIGNAL  
        
        episode_count = 0
        while not STOP_SIGNAL:
            if num_episodes is not None and episode_count >= num_episodes:
                break
            
            self.update_agent_parameters()
            
            # 根据训练进度设置训练阶段 (此逻辑可以保留)
            if hasattr(self, 'current_version'):
                total_steps = self.config.get('total_steps', 1000000)
                for agent in self.agents:
                    if hasattr(agent, 'set_training_phase'):
                        agent.set_training_phase(self.current_version, total_steps)
            
            # 重置智能体
            for agent in self.agents:
                agent.reset()
                agent.train_mode = True

            episode_id = self.stats["episodes"] + 1
            # env.run 会处理整个回合的交互，并在内部调用 on_episode_end
            state_histories, reward, winner_team = self.env.run(episode_id, is_training=True)
            
            # --- 回合结束后，处理经验 ---
            all_agent_experiences = []
            for agent in self.agents:
                if hasattr(agent, 'step_history') and agent.step_history:
                    processed_exps = agent.process_experiences(agent.step_history, winner_team)
                    all_agent_experiences.extend(processed_exps)

            # 保存本回合收集到的所有经验
            if all_agent_experiences:
                self.save_experiences(all_agent_experiences)
            
            # 更新统计
            self.stats["episodes"] += 1
            self.stats["steps"] += len(all_agent_experiences)
            
            if winner_team is not None:
                if winner_team == 0:
                    self.stats["wins_team0"] += 1
                else:
                    self.stats["wins_team1"] += 1
            
            # 记录进度
            if episode_count % 10 == 0:
                total_games = self.stats["episodes"]
                win_rate_team0 = self.stats["wins_team0"] / total_games if total_games > 0 else 0
                win_rate_team1 = self.stats["wins_team1"] / total_games if total_games > 0 else 0
                
                self.logger.info(
                    f"Worker {self.worker_id} | Episodes: {total_games} | "
                    f"Steps: {self.stats['steps']} | "
                    f"Win Rate Team 0: {win_rate_team0:.4f} | "
                    f"Win Rate Team 1: {win_rate_team1:.4f}"
                )
            
            episode_count += 1
        
        return self.stats

class StandaloneLearner:
    """独立的学习者类，通过文件系统与其他进程通信"""
    def __init__(self, learner_id, models_dir, experience_dir, config, logger=None):
        try:
            self.learner_id = learner_id
            self.models_dir = models_dir
            self.experience_dir = experience_dir
            self.config = config
            
            if logger:
                self.logger = logger
            else:
                self.logger = logging.getLogger(f"Learner-{learner_id}") # 简化示例
        
            self.logger.info(f"Learner-{self.learner_id} __init__ started.")

            if self.config.get('device', 'auto') == 'cpu':
                self.device = 'cpu'
            elif torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
            self.logger.info(f"Learner-{self.learner_id} using device: {self.device}")

            # --- 网络创建 ---
            self.logger.info("Initializing networks...")
            self.policy_networks = {}
            self.value_networks = {}
            for player_id in range(4):
                self.policy_networks[player_id] = GuandanLSTMPolicyNetwork(
                    state_dim=673, action_feature_dim=143, abstract_action_dim=336,
                    lstm_hidden_dim=256, ff_hidden_dim=512, num_lstm_layers=2, device=self.device
                ).to(self.device)
                self.value_networks[player_id] = GuandanValueNetwork(
                    state_dim=673, perfect_info_dim=165,
                    lstm_hidden_dim=256, ff_hidden_dim=512, num_lstm_layers=2, device=self.device
                ).to(self.device)
            self.logger.info("Networks initialized successfully.")

            # --- 优化器创建 ---
            self.logger.info("Initializing optimizers...")
            lr = self.config.get('learning_rate', 3e-4)
            self.policy_optimizers = {}
            self.value_optimizers = {}
            self.policy_schedulers = {}
            self.value_schedulers = {}
            for player_id in range(4):
                self.policy_optimizers[player_id] = torch.optim.Adam(self.policy_networks[player_id].parameters(), lr=lr)
                self.value_optimizers[player_id] = torch.optim.Adam(self.value_networks[player_id].parameters(), lr=lr)
                if self.config.get('use_lr_scheduler', True):
                    self.policy_schedulers[player_id] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.policy_optimizers[player_id], mode='min', factor=0.5, patience=5, threshold=0.01, min_lr=1e-5)
                    self.value_schedulers[player_id] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.value_optimizers[player_id], mode='min', factor=0.5, patience=5, threshold=0.01, min_lr=1e-5)
            self.logger.info("Optimizers initialized successfully.")

            # --- 内存创建 ---
            self.logger.info("Initializing memories...")
            self.memories = {player_id: PPOMemory(
                batch_size=self.config.get('batch_size', 1024), max_size=4000,
                gamma=self.config.get('gamma', 0.99), gae_lambda=self.config.get('gae_lambda', 0.95),
                device=self.device
            ) for player_id in range(4)}
            self.logger.info("Memories initialized successfully.")
            
            # --- 统计和版本 ---
            self.stats = {"updates": 0, "samples_processed": 0, "policy_losses": [], "value_losses": [], "entropies": [], "learning_rates": []}
            self.current_version = 0
            
            # --- 加载模型 ---
            self.logger.info("Calling load_initial_models...")
            self.load_initial_models()
            self.logger.info("load_initial_models finished.")
            
            self.logger.info(f"Learner-{self.learner_id} __init__ finished successfully.")

        except Exception as e:
            # 捕获任何在 __init__ 期间发生的异常
            import traceback
            # 使用根日志记录器，以防 self.logger 还未初始化
            logging.error(f"FATAL ERROR in StandaloneLearner __init__: {e}")
            logging.error(traceback.format_exc())
            # 重新抛出异常，让进程崩溃，以便主进程能检测到
            raise
    def load_initial_models(self):
        """尝试加载初始模型"""
        version_file = os.path.join(self.models_dir, "version.txt")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                self.current_version = int(f.read().strip())
            
            # 尝试加载现有模型
            models_path = os.path.join(self.models_dir, f"models_v{self.current_version}.pt")
            if os.path.exists(models_path):
                self.logger.info(f"加载模型: {models_path}")
                
                try:
                    state_dict = torch.load(models_path, map_location=self.device)
                    for player_id in range(4):
                        if f'policy_player{player_id}' in state_dict:
                            self.policy_networks[player_id].load_state_dict(state_dict[f'policy_player{player_id}'])
                        
                        if f'value_player{player_id}' in state_dict:
                            self.value_networks[player_id].load_state_dict(state_dict[f'value_player{player_id}'])
                    
                    self.logger.info("模型加载成功")
                except Exception as e:
                    self.logger.error(f"加载模型时发生错误: {e}")
    
    def save_models(self):
        """保存模型到文件"""
        self.current_version += 1
        
        # 创建模型状态字典
        state_dict = {}
        for player_id in range(4):
            state_dict[f'policy_player{player_id}'] = self.policy_networks[player_id].state_dict()
            state_dict[f'value_player{player_id}'] = self.value_networks[player_id].state_dict()
        
        # 保存模型
        models_path = os.path.join(self.models_dir, f"models_v{self.current_version}.pt")
        torch.save(state_dict, models_path)
        
        # 更新版本文件
        version_file = os.path.join(self.models_dir, "version.txt")
        with open(version_file, 'w') as f:
            f.write(str(self.current_version))
        
        self.logger.info(f"模型已保存，版本: {self.current_version}")
        return self.current_version
    
    def collect_and_process_experiences(self):
        """从经验文件夹中收集经验并直接处理"""
        experience_files = [f for f in os.listdir(self.experience_dir) 
                        if f.startswith("exp_") and f.endswith(".pt")]
        
        if not experience_files:
            return 0
        
        # 随机打乱文件顺序
        random.shuffle(experience_files)
        
        total_processed = 0
        # 每次最多处理一批文件，避免内存爆炸
        for file_name in experience_files[:self.config.get('learner_file_batch_size', 32)]:
            file_path = os.path.join(self.experience_dir, file_name)
            try:
                # 从文件名中解析出 player_id
                parts = file_name.split('_')
                player_id = int(parts[2])
                
                # 加载数据
                data = torch.load(file_path, map_location='cpu') # 先加载到cpu
                
                # 获取对应agent的memory
                memory = self.agents[player_id].memory
                
                # 将加载的数据填充到memory中
                # 注意：这里是直接覆盖，因为我们假设worker发来的是完整的batch
                # 如果是增量式，逻辑会更复杂
                num_samples = len(data['actions'])
                if memory.position + num_samples > memory.max_size:
                    # 如果空间不足，可以先触发一次更新或直接丢弃
                    # 为简单起见，这里我们直接清空旧的
                    memory.clear()

                for i in range(num_samples):
                    # PPOMemory.store 需要 torch.Tensor
                    # action_features 需要特殊处理，它是一个列表的列表
                    action_features_tensors = [torch.from_numpy(feat) for feat in data['action_features'][i]]

                    memory.store(
                        raw_state=data['raw_states'][i],
                        state=torch.from_numpy(data['states'][i]).unsqueeze(0).to(self.device),
                        action=data['actions'][i],
                        log_prob=data['log_probs'][i],
                        val=data['vals'][i],
                        reward=data['rewards'][i],
                        done=data['dones'][i],
                        perfect_info=torch.from_numpy(data['perfect_info'][i]).unsqueeze(0).to(self.device) if data['perfect_info'] is not None else None,
                        action_features=action_features_tensors
                    )
                
                total_processed += num_samples
                os.remove(file_path) # 处理完后删除
            except Exception as e:
                self.logger.error(f"处理经验文件出错: {file_path}, {e}")
                # 如果文件有问题，也删除它，防止阻塞
                if os.path.exists(file_path):
                    os.remove(file_path)

        if total_processed > 0:
            self.stats['samples_processed'] += total_processed

        return total_processed
    
    def update_all_policies(self):
        """更新所有有足够经验的玩家的策略"""
        update_results = {}
        for player_id, agent in self.agents.items():
            # 检查内存中的样本是否足够
            if len(agent.memory) >= agent.steps_per_update:
                # 直接调用agent自身的更新方法
                # ppo_agent.py中的update_policy已经包含了所有复杂逻辑
                self.logger.info(f"Updating policy for player {player_id} with {len(agent.memory)} samples.")
                stats = agent.update_policy()
                if stats:
                    update_results[player_id] = stats
        
        if update_results:
            self.stats['updates'] += 1
            # 聚合统计数据
            self.stats['policy_losses'].append(np.mean([s['policy_loss'] for s in update_results.values()]))
            self.stats['value_losses'].append(np.mean([s['value_loss'] for s in update_results.values()]))
            self.stats['entropies'].append(np.mean([s['entropy'] for s in update_results.values()]))
            
            # 打印日志
            self.logger.info(
                f"Update #{self.stats['updates']}: "
                f"PLoss={self.stats['policy_losses'][-1]:.4f}, "
                f"VLoss={self.stats['value_losses'][-1]:.4f}, "
                f"Entropy={self.stats['entropies'][-1]:.4f}"
            )
    
    def run(self):
        """运行学习循环"""
        global STOP_SIGNAL
        
        last_save_time = time.time()
        save_interval = self.config.get('save_interval_seconds', 300)
        
        while not STOP_SIGNAL:
            # 1. 收集并处理经验
            samples_processed = self.collect_and_process_experiences()
            
            # 2. 如果收集到新经验，就尝试更新策略
            if samples_processed > 0:
                self.update_all_policies()

            # 3. 定期保存模型
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                self.save_models()
                last_save_time = current_time
            
            # 如果没有新经验，短暂休眠
            if samples_processed == 0:
                time.sleep(1)

class SimpleTrainingSystem:
    """训练系统实现 - 使用文件系统通信"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TrainingSystem")
        self.set_seed(config.get('seed', 42))
        self.processes = []
        
        # --- 解析设备分配 ---
        self.actor_gpu_ids = []
        self.learner_device_str = 'cpu'

        # 1. 解析 Learner 设备
        training_device_arg = str(self.config.get('training_device', 'cpu')).lower()
        if training_device_arg != 'cpu' and torch.cuda.is_available():
            self.learner_device_str = f"cuda:{training_device_arg}"
        else:
            self.learner_device_str = 'cpu'
        self.logger.info(f"Learner 将使用设备: {self.learner_device_str}")

        # 2. 解析 Actor 设备
        num_actor_devices = self.config.get('num_actor_devices', 0)
        if num_actor_devices > 0 and torch.cuda.is_available():
            all_gpu_devices_str = self.config.get('gpu_devices', '0').split(',')
            all_gpu_ids = [int(gid.strip()) for gid in all_gpu_devices_str]
            
            learner_gpu_id = -1
            if 'cuda' in self.learner_device_str:
                learner_gpu_id = int(self.learner_device_str.split(':')[-1])
            
            available_for_actors = [gid for gid in all_gpu_ids if gid != learner_gpu_id]
            
            if len(available_for_actors) < num_actor_devices:
                raise ValueError(
                    f"请求了 {num_actor_devices} 个 Actor GPU，但排除了 Learner GPU ({learner_gpu_id}) 后只剩下 "
                    f"{len(available_for_actors)} 个可用: {available_for_actors}。"
                )
            
            self.actor_gpu_ids = available_for_actors[:num_actor_devices]
            self.logger.info(f"Actors 将使用 {len(self.actor_gpu_ids)} 个 GPU: {self.actor_gpu_ids}")
        else:
            self.logger.info("Actors 将使用 CPU。")

        self.setup_wandb()
        self.logger.info("训练系统初始化完成")
    
    def set_seed(self, seed):
        """设置随机种子"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def setup_wandb(self):
        """设置wandb"""
        self.use_wandb = self.config.get('use_wandb', False)
        
        if self.use_wandb:
            # 尝试导入wandb
            try:
                import wandb
                
                # 准备wandb配置
                wandb_config = {
                    'project': self.config.get('wandb_project', 'guandan-ppo'),
                    'name': self.config.get('wandb_name', f"train-{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                    'tags': self.config.get('wandb_tags', '').split(',') if self.config.get('wandb_tags') else [],
                    'config': self.config
                }
                
                # 初始化wandb
                wandb.init(**wandb_config)
                self.logger.info(f"Wandb已初始化: {wandb_config['name']}")
                
                self.wandb = wandb
            except ImportError:
                self.logger.warning("Wandb不可用，将仅保存到本地文件")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def run_periodic_evaluation(self):
        """加载最新模型，并与基准对手进行快速评估"""
        self.logger.info("开始周期性评估...")
        
        # 找到最新的模型
        models_dir = os.path.join("shared_data", "models")
        version_file = os.path.join(models_dir, "version.txt")
        if not os.path.exists(version_file):
            self.logger.warning("评估失败：找不到模型版本文件。")
            return

        with open(version_file, 'r') as f:
            latest_version = int(f.read().strip())
        
        latest_model_path = os.path.join(models_dir, f"models_v{latest_version}.pt")
        if not os.path.exists(latest_model_path):
            self.logger.warning(f"评估失败：找不到模型文件 {latest_model_path}。")
            return

        try:
            # 导入评估脚本中的函数 (确保路径正确)
            from evaluate_guandan_ppo import load_ppo_agents, evaluate_against_opponent
            
            # 使用一个专用的评估GPU或CPU
            eval_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 加载智能体
            ppo_agents = load_ppo_agents(latest_model_path, eval_device)
            
            # 创建评估环境
            eval_env = GuandanEnv({'seed': 999}) # 使用固定种子
            
            # 与一个或多个基准对手进行快速评估
            opponent_to_test = "Base5" # 选择一个有代表性的对手
            num_eval_games = 50 # 评估局数不宜过多，以免阻塞训练
            
            results = evaluate_against_opponent(
                eval_env, ppo_agents, opponent_to_test, num_eval_games, team_mode=True
            )
            
            win_rate = results['win_rate']
            self.logger.info(f"评估完成 (模型版本 {latest_version}): 对抗 {opponent_to_test} 胜率 = {win_rate:.4f}")
            
            # 将评估结果记录到 wandb
            if self.use_wandb and self.wandb:
                self.wandb.log({
                    'eval/win_rate_vs_Base5': win_rate,
                    'model_version': latest_version # 使用 model_version 作为 x 轴
                })

        except Exception as e:
            self.logger.error(f"周期性评估期间发生错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def save_initial_model(self, models_dir):
        """保存初始模型到共享目录"""
        # 获取参数
        params = {}
        for player_id in range(4):
            # 策略网络
            # FIX: 提供所有必需的参数，与 Learner 和 Agent 中的定义保持一致
            policy_net = GuandanLSTMPolicyNetwork(
                state_dim=673,
                action_feature_dim=143,
                abstract_action_dim=336,
                lstm_hidden_dim=256,
                ff_hidden_dim=512,
                num_lstm_layers=2,
                device='cpu'  # 初始模型总是在CPU上创建
            )
            
            # 价值网络
            # FIX: 提供所有必需的参数
            value_net = GuandanValueNetwork(
                state_dim=673,
                perfect_info_dim=165,
                lstm_hidden_dim=256,
                ff_hidden_dim=512,
                num_lstm_layers=2,
                device='cpu'  # 初始模型总是在CPU上创建
            )
            
            params[f'policy_player{player_id}'] = policy_net.state_dict()
            params[f'value_player{player_id}'] = value_net.state_dict()
        
        # 加载预训练模型(如果指定)
        if self.config.get('load_model'):
            model_path = self.config.get('load_model')
            if os.path.exists(model_path):
                self.logger.info(f"加载预训练模型: {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    
                    # 检查是否是PPO检查点
                    if 'policy_network_state_dict' in checkpoint and 'value_network_state_dict' in checkpoint:
                        # 更新初始模型
                        for player_id in range(4):
                            if player_id == 0 or self.config.get('share_weights', False):
                                params[f'policy_player{player_id}'] = checkpoint['policy_network_state_dict']
                                params[f'value_player{player_id}'] = checkpoint['value_network_state_dict']
                    else:
                        self.logger.warning("无效的检查点格式，使用随机初始化")
                except Exception as e:
                    self.logger.error(f"加载预训练模型失败: {e}")
        
        # 保存模型
        models_path = os.path.join(models_dir, "models_v0.pt")
        torch.save(params, models_path)
        
        # 保存版本文件
        version_file = os.path.join(models_dir, "version.txt")
        with open(version_file, 'w') as f:
            f.write("0")
        
        self.logger.info("模型参数已初始化")
    
    def start_workers(self):
        """启动工作进程，并将其分配到指定的Actor GPU或CPU上"""
        num_actors = self.config.get('num_actors', 2)
        
        self.logger.info(f"启动 {num_actors} 个 Actor 进程...")
        
        models_dir = os.path.join("shared_data", "models")
        experience_dir = os.path.join("shared_data", "experience")
        config_path = os.path.join("shared_data", "config.json")
        
        for actor_id in range(num_actors):
            target_func = run_worker_standalone
            args = (actor_id, models_dir, experience_dir, config_path)
            
            # 如果有可用的 Actor GPU，则进行绑定
            if self.actor_gpu_ids:
                gpu_id = self.actor_gpu_ids[actor_id % len(self.actor_gpu_ids)]
                target_func = self.run_process_with_gpu_env
                args = (run_worker_standalone, args, gpu_id)

            actor_process = mp.Process(target=target_func, args=args, daemon=True)
            actor_process.start()
            self.processes.append(actor_process)

    def start_learner(self):
        """启动学习进程，并将其绑定到指定的训练设备上"""
        self.logger.info("启动 Learner 进程...")
        
        models_dir = os.path.join("shared_data", "models")
        experience_dir = os.path.join("shared_data", "experience")
        config_path = os.path.join("shared_data", "config.json")
        
        target_func = run_learner_standalone
        args = (0, models_dir, experience_dir, config_path)
        
        # 如果 Learner 使用 GPU，则进行绑定
        if 'cuda' in self.learner_device_str:
            gpu_id = int(self.learner_device_str.split(':')[-1])
            target_func = self.run_process_with_gpu_env
            args = (run_learner_standalone, args, gpu_id)

        learner_process = mp.Process(target=target_func, args=args, daemon=True)
        learner_process.start()
        self.processes.append(learner_process)

    def run_process_with_gpu_env(self, target_func, args, gpu_id):
        """通用包装函数，用于在启动目标函数前设置 CUDA_VISIBLE_DEVICES"""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # 注意：这里不能使用 self.logger，因为 logger 对象不可序列化
        print(f"进程 {target_func.__name__} (PID: {os.getpid()}) 正在启动，绑定到物理 GPU: {gpu_id}")
        target_func(*args)
    
    def log_metrics(self):
        """记录训练指标"""
        # 获取共享目录路径
        models_dir = os.path.join("shared_data", "models")
        experience_dir = os.path.join("shared_data", "experience")
        
        # 计算经验文件数量
        experience_count = len([f for f in os.listdir(experience_dir) if f.startswith("exp_") and f.endswith(".pt")])
        
        # 获取当前模型版本
        version = 0
        version_file = os.path.join(models_dir, "version.txt")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version = int(f.read().strip())
        
        # 记录指标
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experience_files': experience_count,
            'model_version': version
        }
        
        # 如果启用，记录到wandb
        if self.use_wandb and self.wandb:
            self.wandb.log(metrics)
        
        # 记录到控制台
        self.logger.info(
            f"指标 | 经验文件: {experience_count} | "
            f"模型版本: {version}"
        )
    
    def save_checkpoint(self, custom_path=None):
        """保存当前模型检查点到指定路径"""
        # 获取当前模型路径
        models_dir = os.path.join("shared_data", "models")
        version_file = os.path.join(models_dir, "version.txt")
        
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version = int(f.read().strip())
            
            models_src = os.path.join(models_dir, f"models_v{version}.pt")
            
            if os.path.exists(models_src):
                # 确定保存路径
                if custom_path is None:
                    checkpoint_dir = "checkpoints"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    custom_path = os.path.join(checkpoint_dir, f"checkpoint_v{version}.pt")
                else:
                    # 确保自定义路径的目录存在
                    checkpoint_dir = os.path.dirname(custom_path)
                    if checkpoint_dir:  # 如果路径包含目录部分
                        os.makedirs(checkpoint_dir, exist_ok=True)
                
                # 复制模型文件
                import shutil
                shutil.copy2(models_src, custom_path)
                
                self.logger.info(f"检查点已保存到 {custom_path}")
                
                # 如果启用，保存到wandb
                if self.use_wandb and self.wandb:
                    self.wandb.save(custom_path)
                
                return custom_path
        
        self.logger.warning("无法保存检查点，找不到当前模型")
        return None
    
    def run(self):
        """
        运行训练系统的主循环。
        协调启动 Actors 和 Learner，并负责监控、保存和评估。
        """
        global STOP_SIGNAL
        
        # 记录开始时间
        start_time = time.time()
        
        # 启动所有工作进程和学习进程
        self.start_workers()
        self.start_learner()
        
        # 创建检查点目录
        os.makedirs("checkpoints", exist_ok=True)
        
        # 从配置中获取训练参数
        save_interval_versions = self.config.get('save_interval', 1000)  # 按版本数保存的间隔
        log_interval_seconds = self.config.get('log_interval', 60)      # 日志打印的时间间隔 (秒)
        eval_interval_seconds = self.config.get('eval_interval', 3600)   # 实时评估的时间间隔 (秒)
        total_versions_target = self.config.get('total_steps', 1000000) # 目标模型版本数

        try:
            self.logger.info(f"开始训练，目标模型版本: {total_versions_target}")
            
            # 初始化循环变量
            last_checkpoint_version = 0
            last_log_time = time.time()
            last_eval_time = time.time()
            
            # 主监控循环
            while True:
                # 1. 检查所有子进程是否都还存活
                if not all(p.is_alive() for p in self.processes):
                    self.logger.error("一个或多个子进程已意外终止，停止训练。")
                    break
                
                # 2. 获取当前最新的模型版本号
                current_version = 0
                version_file = os.path.join("shared_data", "models", "version.txt")
                if os.path.exists(version_file):
                    try:
                        with open(version_file, 'r') as f:
                            content = f.read().strip()
                            if content:
                                current_version = int(content)
                    except (IOError, ValueError) as e:
                        self.logger.warning(f"读取版本文件时出错: {e}, 将使用版本 0。")
                
                # 3. 周期性打印系统日志
                current_time = time.time()
                if current_time - last_log_time >= log_interval_seconds:
                    self.log_metrics() # 打印经验池文件数等信息
                    last_log_time = current_time
                
                # 4. 周期性保存模型检查点
                if current_version - last_checkpoint_version >= save_interval_versions:
                    self.save_checkpoint()
                    last_checkpoint_version = current_version
                
                # 5. 周期性进行实时评估
                if current_time - last_eval_time >= eval_interval_seconds:
                    self.run_periodic_evaluation()
                    last_eval_time = current_time

                # 6. 检查是否达到训练目标
                if current_version >= total_versions_target:
                    self.logger.info(f"已达到目标模型版本: {total_versions_target}。训练完成。")
                    break
                
                # 休眠以避免主进程过度占用CPU
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("训练被用户手动中断。")
        
        finally:
            # 训练结束或中断后的清理工作
            self.logger.info("正在发送停止信号并清理子进程...")
            STOP_SIGNAL = True
            
            # 等待所有子进程终止
            for p in self.processes:
                p.join(timeout=10) # 等待10秒
                if p.is_alive():
                    self.logger.warning(f"进程 {p.pid} 未能在10秒内终止，将强制终止。")
                    p.terminate() # 强制终止
            
            # 保存最终的模型检查点
            self.logger.info("正在保存最终的模型检查点...")
            final_checkpoint_path = os.path.join("checkpoints", "final_checkpoint.pt")
            self.save_checkpoint(final_checkpoint_path)
            
            # 计算并打印总训练时长
            total_time_seconds = time.time() - start_time
            total_time_hours = total_time_seconds / 3600
            self.logger.info(
                f"训练结束，总用时: {total_time_hours:.2f} 小时 ({total_time_seconds:.0f} 秒)"
            )
            
            # 如果启用，确保 wandb 会话结束
            if self.use_wandb and self.wandb:
                self.wandb.finish()

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    """主函数，负责解析参数和启动训练系统"""
    parser = argparse.ArgumentParser(
        description='掼蛋PPO分布式训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- 系统与设备配置 ---
    parser.add_argument('--gpu-devices', type=str, default='0', 
                        help='可用于训练的所有GPU设备ID，逗号分隔 (例如: "0,1,2,3")')
    parser.add_argument('--num-actor-devices', type=int, default=0, 
                        help='用于运行Actor的GPU数量。如果为0或无可用GPU，则Actor使用CPU')
    parser.add_argument('--training-device', type=str, default='cpu', 
                        help='用于训练(Learner)的设备ID (例如: "0" 或 "cpu")')
    parser.add_argument('--num-actors', type=int, default=2, 
                        help='启动的Actor(worker)进程总数，将被均匀分配到Actor设备上')

    # --- 训练超参数 ---
    parser.add_argument('--total-steps', type=int, default=1000000, 
                        help='训练总步数 (以Learner更新次数为准)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    parser.add_argument('--batch-size', type=int, default=1024, 
                        help='Learner的训练批次大小')
    parser.add_argument('--ppo-epochs', type=int, default=4, 
                        help='每批数据在Learner上重复训练的轮数')
    parser.add_argument('--learning-rate', type=float, default=3e-4, 
                        help='学习率')
    
    # --- 模型与保存 ---
    parser.add_argument('--load-model', type=str, default=None, 
                        help='预训练模型路径 (用于初始化 version 0 模型)')
    parser.add_argument('--save-interval', type=int, default=1000, 
                        help='保存模型检查点的间隔 (按Learner更新次数)')

    # --- 日志与监控 ---
    parser.add_argument('--log-interval', type=int, default=60, 
                        help='打印系统日志的时间间隔 (秒)')
    parser.add_argument('--eval-interval', type=int, default=3600, 
                        help='进行实时评估的时间间隔 (秒)')
    parser.add_argument('--use-wandb', action='store_true', 
                        help='使用Weights & Biases记录训练')
    parser.add_argument('--wandb-project', type=str, default='guandan-ppo', 
                        help='Wandb项目名称')
    parser.add_argument('--wandb-name', type=str, default=None, 
                        help='Wandb运行名称 (默认为时间戳)')
    parser.add_argument('--wandb-tags', type=str, default=None, 
                        help='Wandb标签，逗号分隔')
    
    # 解析参数
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"参数解析错误: {e}")
        parser.print_help()
        return
    
    # 转换为配置字典
    config = vars(args)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保保存路径存在
    if args.model_path:
        dir_path = os.path.dirname(args.model_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    print(f"开始训练，使用设备: {args.device}, 工作进程数: {args.num_workers}")
    
    try:
        # 确保共享数据目录存在
        os.makedirs("shared_data", exist_ok=True)
        
        # 在Windows上使用spawn启动方法
        mp.set_start_method('spawn', force=True)
        
        # 创建并运行训练系统
        training_system = SimpleTrainingSystem(config)
        training_system.run()
    except KeyboardInterrupt:
        print("训练被用户中断")
        global STOP_SIGNAL
        STOP_SIGNAL = True
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Windows多进程特殊处理
    if os.name == 'nt':  # Windows系统
        mp.freeze_support()
    
    main()
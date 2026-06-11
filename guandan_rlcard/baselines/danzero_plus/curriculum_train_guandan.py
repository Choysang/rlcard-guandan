import os
import sys
import torch
import numpy as np
import argparse
import json
import logging
import random
from datetime import datetime, timedelta
from collections import deque
import matplotlib.pyplot as plt
import time

# 导入PPO智能体
from baselines.dmc.dmc_agent import DMCAgent

# 导入规则模型
from guandan_rlcard.baselines.random_agent import RandomAgent
from guandan_rlcard.baselines.rule_based.base1.base1_agent import Base1Agent
from guandan_rlcard.baselines.rule_based.base3.base3_agent import Base3Agent
from guandan_rlcard.baselines.rule_based.base4.base4_agent import Base4Agent
from guandan_rlcard.baselines.rule_based.base5.base5_agent import Base5Agent
from guandan_rlcard.baselines.rule_based.base6.base6_agent import Base6Agent
from guandan_rlcard.baselines.rule_based.base7.base7_agent import Base7Agent
from guandan_rlcard.baselines.rule_based.base8.base8_agent import Base8Agent

# 导入环境和工具函数
from guandan_rlcard.envs.guandan_env import GuandanEnv
from baselines.Guandanzero.utils import plot_training_progress, print_training_summary

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class TrainingStats:
    """
    训练统计管理器，用于记录和分析训练过程中的数据
    """
    def __init__(self, window_size=1000, save_dir='training_stats'):
        """
        初始化训练统计管理器
        
        参数:
        window_size: 滑动窗口大小，用于计算平均值和判断收敛
        save_dir: 统计数据保存目录
        """
        self.window_size = window_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 基本统计数据
        self.episodes = 0
        self.start_time = datetime.now()
        self.episode_rewards = []
        self.win_rates = []
        self.win_count = 0
        
        # 对各个规则模型的胜率统计
        self.rule_model_stats = {
            "Random": {"wins": 0, "games": 0, "history": []},
            "Base1": {"wins": 0, "games": 0, "history": []},
            "Base3": {"wins": 0, "games": 0, "history": []},
            "Base4": {"wins": 0, "games": 0, "history": []},
            "Base5": {"wins": 0, "games": 0, "history": []},
            "Base6": {"wins": 0, "games": 0, "history": []},
            "Base7": {"wins": 0, "games": 0, "history": []},
            "Base8": {"wins": 0, "games": 0, "history": []}
        }
        
        # 阶段性评估结果
        self.eval_results = []
        
        # 收敛检测
        self.win_rate_window = deque(maxlen=window_size)
        self.reward_window = deque(maxlen=window_size)
        self.convergence_metrics = []
        
        # 最佳模型记录
        self.best_win_rate = 0.0
        self.best_model_episode = 0
        
        # 进度记录
        self.progress_log = []
    
    def update(self, reward, win, opponent_names=None):
        """
        更新基本统计信息
        
        参数:
        reward: 当前回合奖励
        win: 是否获胜
        opponent_names: 对手模型名称列表
        """
        self.episodes += 1
        self.episode_rewards.append(reward)
        self.reward_window.append(reward)
        
        if win:
            self.win_count += 1
        
        current_win_rate = self.win_count / self.episodes
        self.win_rates.append(current_win_rate)
        self.win_rate_window.append(1 if win else 0)
        
        # 更新对手特定统计
        if opponent_names:
            for name in opponent_names:
                if name in self.rule_model_stats:
                    self.rule_model_stats[name]["games"] += 1
                    if win:
                        self.rule_model_stats[name]["wins"] += 1
                    
                    # 计算并记录当前胜率
                    current_model_win_rate = (
                        self.rule_model_stats[name]["wins"] / 
                        self.rule_model_stats[name]["games"]
                    )
                    self.rule_model_stats[name]["history"].append(
                        (self.episodes, current_model_win_rate)
                    )
        
        # 更新最佳模型记录
        if current_win_rate > self.best_win_rate:
            self.best_win_rate = current_win_rate
            self.best_model_episode = self.episodes
        
        # 每100局记录一次收敛进度
        if self.episodes % 100 == 0:
            window_win_rate = np.mean(self.win_rate_window) if self.win_rate_window else 0
            window_reward = np.mean(self.reward_window) if self.reward_window else 0
            
            self.convergence_metrics.append({
                "episode": self.episodes,
                "win_rate": current_win_rate,
                "window_win_rate": window_win_rate,
                "window_reward": window_reward,
                "elapsed_time": str(datetime.now() - self.start_time)
            })
            
            # 记录进度日志
            self.progress_log.append({
                "episode": self.episodes,
                "win_rate": current_win_rate,
                "window_win_rate": window_win_rate,
                "elapsed_time": str(datetime.now() - self.start_time),
                "rule_model_stats": {
                    name: {
                        "win_rate": stats["wins"] / stats["games"] if stats["games"] > 0 else 0,
                        "games": stats["games"]
                    } 
                    for name, stats in self.rule_model_stats.items() 
                    if stats["games"] > 0
                }
            })
    
    def add_evaluation_result(self, win_rates, avg_reward, breakdown=None):
        """
        添加评估结果
        
        参数:
        win_rates: 对各类对手的胜率
        avg_reward: 平均奖励
        breakdown: 对各个具体模型的胜率详细信息
        """
        eval_result = {
            "episode": self.episodes,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": str(datetime.now() - self.start_time),
            "win_rates": win_rates,
            "avg_reward": avg_reward,
            "breakdown": breakdown or {}
        }
        
        self.eval_results.append(eval_result)
        
        # 更新进度日志
        self.progress_log.append({
            "episode": self.episodes,
            "evaluation": True,
            "overall_win_rate": win_rates.get("overall", 0),
            "breakdown": breakdown or {},
            "elapsed_time": str(datetime.now() - self.start_time)
        })
    
    def is_converged(self, threshold=0.01, window_count=5):
        """
        检查模型是否收敛
        
        参数:
        threshold: 收敛阈值，连续几个窗口胜率变化小于此值视为收敛
        window_count: 需要多少个连续窗口满足条件
        
        返回:
        bool: 是否已收敛
        """
        if len(self.convergence_metrics) < window_count + 1:
            return False
        
        # 检查最近几个窗口的胜率变化
        for i in range(1, window_count + 1):
            idx = -i
            prev_idx = -(i + 1)
            
            win_rate_change = abs(self.convergence_metrics[idx]["window_win_rate"] - 
                              self.convergence_metrics[prev_idx]["window_win_rate"])
            
            if win_rate_change > threshold:
                return False
        
        return True
    
    def is_beating_all_models(self, win_rate_threshold=0.55, min_games=30):
        """
        检查是否已经击败所有规则模型
        
        参数:
        win_rate_threshold: 胜率阈值，高于此值视为击败
        min_games: 最少对战局数，确保有足够样本
        
        返回:
        bool: 是否已击败所有规则模型
        """
        for name, stats in self.rule_model_stats.items():
            if name == "Random":  # 忽略随机模型
                continue
                
            if stats["games"] < min_games:
                return False
                
            current_win_rate = stats["wins"] / stats["games"] if stats["games"] > 0 else 0
            if current_win_rate < win_rate_threshold:
                return False
        
        return True
    
    def save(self, filename="training_stats.json"):
        """保存统计数据到文件"""
        data = {
            "episodes": self.episodes,
            "training_time": str(datetime.now() - self.start_time),
            "win_rates": self.win_rates,
            "episode_rewards": self.episode_rewards,
            "rule_model_stats": {
                name: {
                    "wins": stats["wins"],
                    "games": stats["games"],
                    "history": stats["history"]
                }
                for name, stats in self.rule_model_stats.items()
            },
            "eval_results": self.eval_results,
            "convergence_metrics": self.convergence_metrics,
            "best_model": {
                "win_rate": self.best_win_rate,
                "episode": self.best_model_episode
            },
            "progress_log": self.progress_log
        }
        
        with open(os.path.join(self.save_dir, filename), 'w') as f:
            json.dump(data, f, indent=2)
    
    def plot_win_rates(self, save_path=None, show=False):
        """绘制胜率曲线"""
        plt.figure(figsize=(12, 8))
        
        # 绘制总体胜率
        episodes = list(range(1, self.episodes + 1))
        plt.plot(episodes, self.win_rates, 'b-', label='Overall Win Rate', alpha=0.7)
        
        # 绘制对各规则模型的胜率
        for name, stats in self.rule_model_stats.items():
            if stats["games"] > 0 and stats["history"]:
                history_episodes, history_win_rates = zip(*stats["history"])
                plt.plot(history_episodes, history_win_rates, '-', label=f'{name} Win Rate', alpha=0.6)
        
        # 绘制评估结果
        if self.eval_results:
            eval_episodes = [result["episode"] for result in self.eval_results]
            eval_win_rates = [result["win_rates"].get("overall", 0) for result in self.eval_results]
            plt.scatter(eval_episodes, eval_win_rates, color='red', marker='o', label='Evaluation', s=50)
        
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate')
        plt.title('Win Rate vs Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            
        if show:
            plt.show()
        
        plt.close()
    
    def print_progress(self):
        """打印当前训练进度"""
        elapsed_time = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n===== 训练进度 (回合 {self.episodes}) =====")
        print(f"已训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        print(f"总体胜率: {self.win_rates[-1]:.4f} ({self.win_count}/{self.episodes})")
        
        # 打印对各规则模型的胜率
        print("\n各模型胜率:")
        for name, stats in self.rule_model_stats.items():
            if stats["games"] > 0:
                win_rate = stats["wins"] / stats["games"]
                status = "✓" if win_rate >= 0.55 else "✗" if stats["games"] >= 30 else "?"
                print(f"{name}: {win_rate:.4f} ({stats['wins']}/{stats['games']}) {status}")
        
        # 打印收敛状态
        is_conv = self.is_converged()
        is_beating = self.is_beating_all_models()
        
        print("\n阶段性目标:")
        print(f"击败所有规则模型: {'是' if is_beating else '否'}")
        print(f"模型收敛: {'是' if is_conv else '否'}")
        print(f"最佳模型: 回合 {self.best_model_episode}，胜率 {self.best_win_rate:.4f}")
        
        if is_beating and not is_conv:
            print("\n当前状态: 已击败所有规则模型，正在进行自博弈以达到收敛")
        elif not is_beating and not is_conv:
            print("\n当前状态: 正在学习击败规则模型")
        elif is_conv:
            print("\n当前状态: 模型已收敛")
        
        print("=====================================\n")

class OpponentPoolManager:
    """
    对手池管理器，根据训练阶段动态选择对手
    """
    def __init__(self, random_state, max_model_pool_size=5):
        """
        初始化对手池管理器
        
        参数:
        random_state: 随机数生成器
        max_model_pool_size: 模型池最大容量
        """
        self.random_state = random_state
        self.max_model_pool_size = max_model_pool_size
        
        # 初始化规则模型（按强度降序排列）
        self.rule_models = {
            "strong": [
                (Base3Agent, "Base3"),
                (Base1Agent, "Base1")
            ],
            "medium": [
                (Base4Agent, "Base4"),
                (Base7Agent, "Base7")
            ],
            "weak": [
                (Base6Agent, "Base6"),
                (Base8Agent, "Base8"),
                (Base5Agent, "Base5")
            ]
        }
        
        # 所有规则模型的完整列表（用于单独评估）
        self.all_rule_models = [
            (Base3Agent, "Base3"),
            (Base1Agent, "Base1"),
            (Base4Agent, "Base4"),
            (Base7Agent, "Base7"),
            (Base6Agent, "Base6"),
            (Base8Agent, "Base8"),
            (Base5Agent, "Base5"),
            (RandomAgent, "Random")
        ]
        
        # 初始化模型池，存储历史表现良好的模型
        self.model_pool = deque(maxlen=max_model_pool_size)
        
        # 训练阶段标志
        self.stage = "learning"  # 可选值: "learning", "selfplay", "converged"
    
    def update_stage(self, is_beating_all, is_converged):
        """
        更新训练阶段
        
        参数:
        is_beating_all: 是否已击败所有规则模型
        is_converged: 是否已收敛
        """
        if is_converged:
            self.stage = "converged"
        elif is_beating_all:
            self.stage = "selfplay"
        else:
            self.stage = "learning"
    
    def add_model_to_pool(self, model_path, win_rate):
        """
        将表现良好的模型添加到模型池
        
        参数:
        model_path: 模型路径
        win_rate: 模型胜率
        """
        # 当池已满时，移除最旧的模型
        if len(self.model_pool) >= self.max_model_pool_size:
            self.model_pool.popleft()
        
        # 添加新模型
        self.model_pool.append((model_path, win_rate))
    
    def select_opponents(self, env, device, positions=(1, 3)):
        """
        根据当前训练阶段选择对手
        
        参数:
        env: 游戏环境
        device: 计算设备
        positions: 对手位置，默认为1和3号位（PPO智能体在0和2号位）
        
        返回:
        opponents: 选择的对手列表
        """
        opponents = []
        
        # 根据训练阶段确定不同类型对手的概率
        if self.stage == "learning":
            # 学习阶段：关注击败规则模型
            random_prob = 0.2
            weak_rule_prob = 0.2
            medium_rule_prob = 0.3
            strong_rule_prob = 0.2
            self_play_prob = 0.05
            model_pool_prob = 0.05
        elif self.stage == "selfplay":
            # 自博弈阶段：主要进行自我对弈，仍保留少量其他对手
            random_prob = 0.05
            weak_rule_prob = 0.0
            medium_rule_prob = 0.05
            strong_rule_prob = 0.1
            self_play_prob = 0.7
            model_pool_prob = 0.1
        else:  # converged
            # 收敛阶段：纯自博弈
            random_prob = 0.0
            weak_rule_prob = 0.0
            medium_rule_prob = 0.0
            strong_rule_prob = 0.0
            self_play_prob = 1.0
            model_pool_prob = 0.0
        
        # 为每个位置选择对手
        for position in positions:
            # 随机决定对手类型
            p = self.random_state.random()
            
            if p < random_prob:
                # 随机智能体 (这里已经传递了 np_random)
                opponents.append((RandomAgent(position, self.random_state), "Random")) # [cite: 1]
            elif p < random_prob + weak_rule_prob:
                # 弱规则智能体
                weak_models = self.rule_models["weak"] #
                selected_index = self.random_state.randint(len(weak_models))
                agent_class, name = weak_models[selected_index]
                # --- 修改：添加 np_random=self.random_state ---
                opponents.append((agent_class(position, np_random=self.random_state), f"{name}")) # 修改此行
            elif p < random_prob + weak_rule_prob + medium_rule_prob:
                # 中等规则智能体
                medium_models = self.rule_models["medium"] #
                selected_index = self.random_state.randint(len(medium_models))
                agent_class, name = medium_models[selected_index]
                 # --- 修改：添加 np_random=self.random_state ---
                opponents.append((agent_class(position, np_random=self.random_state), f"{name}")) # 修改此行
            elif p < random_prob + weak_rule_prob + medium_rule_prob + strong_rule_prob:
                # 强规则智能体
                strong_models = self.rule_models["strong"] #
                selected_index = self.random_state.randint(len(strong_models))
                agent_class, name = strong_models[selected_index]
                 # --- 修改：添加 np_random=self.random_state ---
                opponents.append((agent_class(position, np_random=self.random_state), f"{name}")) # 修改此行
            elif p < random_prob + weak_rule_prob + medium_rule_prob + strong_rule_prob + self_play_prob:
                # 自博弈 - 使用当前最新的PPO智能体
                ppo_opponent = PPOGuandanAgent(
                    player_id=position,
                    np_random=self.random_state,
                    state_dim=256,
                    action_dim=300,
                    device=device,
                    use_transformer=True  # 使用与训练模型相同的网络架构
                )
                # 关闭训练模式
                ppo_opponent.train_mode = False
                
                opponents.append((ppo_opponent, "SelfPlay"))
            else:
                # 从模型池中选择
                if self.model_pool:
                    # 从模型池选择也应该用索引
                    pool_index = self.random_state.randint(len(self.model_pool))
                    model_path, win_rate = self.model_pool[pool_index]
                    ppo_opponent = PPOGuandanAgent(
                        player_id=position,
                        np_random=self.random_state,
                        state_dim=256,
                        action_dim=300,
                        model_path=model_path,
                        device=device,
                        use_transformer=True  # 保持一致的网络架构
                    )
                    ppo_opponent.train_mode = False
                    
                    opponents.append((ppo_opponent, f"Pool(WR:{win_rate:.2f})"))
                else: # 模型池为空的情况
                    # 如果模型池为空，使用随机智能体 (这里已经传递了 np_random)
                    opponents.append((RandomAgent(position, self.random_state), "Random")) # [cite: 1]

        return opponents
    
    def create_evaluation_opponents(self, device):
        """
        创建用于评估的所有对手
        
        参数:
        device: 计算设备
        
        返回:
        eval_opponents: 用于评估的对手字典
        """
        eval_opponents = {}

        # 添加所有规则模型
        for agent_class, name in self.all_rule_models: # [cite: 1]
            eval_opponents[name] = []
            for position in [1, 3]: # [cite: 1]
                # --- 修改：添加 np_random=self.random_state ---
                # 注意：确保 self.random_state 在这个方法中可用
                # OpponentPoolManager 初始化时接收了 random_state，所以 self.random_state 可用
                eval_opponents[name].append(agent_class(position, np_random=self.random_state)) # 修改此行

        # 添加自博弈模型 (PPOGuandanAgent 不需要显式传递 np_random，它内部处理)
        eval_opponents["SelfPlay"] = [] # [cite: 1]
        for position in [1, 3]:
            ppo_opponent = PPOGuandanAgent(
                player_id=position,
                np_random=self.random_state, # PPOGuandanAgent 接收 np_random
                state_dim=256,
                action_dim=300,
                device=device,
                use_transformer=True
            )
            ppo_opponent.train_mode = False # [cite: 1]
            eval_opponents["SelfPlay"].append(ppo_opponent)

        # 如果有模型池，也添加到评估集 (PPOGuandanAgent 处理 np_random)
        if self.model_pool: # [cite: 1]
            # ... (这部分代码应该没问题，因为 PPOGuandanAgent 接收 np_random) ...
             eval_opponents["ModelPool"] = [] # [cite: 1]
             for position in [1, 3]:
                 model_path, _ = self.model_pool[-1] # [cite: 1]
                 ppo_opponent = PPOGuandanAgent(
                     player_id=position,
                     np_random=self.random_state, # [cite: 1]
                     state_dim=256, # [cite: 1]
                     action_dim=300, # [cite: 1]
                     model_path=model_path, # [cite: 1]
                     device=device, # [cite: 1]
                     use_transformer=True # [cite: 1]
                 )
                 ppo_opponent.train_mode = False # [cite: 1]
                 eval_opponents["ModelPool"].append(ppo_opponent) # [cite: 1]


        return eval_opponents

def train_until_convergence(env, ppo_agent_0, ppo_agent_2, opponent_manager, stats_manager,
                          max_episodes=10000000, save_path='models/ppo_guandan', 
                          save_interval=500, evaluate_interval=1000, 
                          evaluate_num=100, pool_threshold=0.8,
                          beating_threshold=0.85, convergence_threshold=0.01,
                          patience=3, checkpoint_dir='checkpoints'):
    """
    训练PPO智能体直至收敛
    
    参数:
    env: 游戏环境
    ppo_agent_0: 0号位PPO智能体
    ppo_agent_2: 2号位PPO智能体
    opponent_manager: 对手池管理器
    stats_manager: 统计管理器
    max_episodes: 最大训练回合数（防止无限训练）
    save_path: 模型保存路径
    save_interval: 保存间隔
    evaluate_interval: 评估间隔
    evaluate_num: 每次评估的局数
    pool_threshold: 添加到模型池的胜率阈值
    beating_threshold: 视为击败规则模型的胜率阈值
    convergence_threshold: 收敛判定阈值
    patience: 收敛确认所需的连续符合条件的评估次数
    checkpoint_dir: 检查点保存目录
    """
    # 创建检查点保存目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 设置训练模式
    ppo_agent_0.train_mode = True
    ppo_agent_2.train_mode = True
    
    # 收敛检测
    convergence_counter = 0
    is_beating_all = False
    is_converged = False
    
    # 记录训练开始时间
    start_time = datetime.now()
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"开始无限训练，直至收敛")
    print(f"使用动态训练策略，自动调整对手")
    
    # 配置常量
    TEAM_0_PLAYERS = [0, 2]  # 队伍0的玩家ID
    TEAM_1_PLAYERS = [1, 3]  # 队伍1的玩家ID
    
    # 训练循环
    episode = 0
    last_save_time = datetime.now()
    last_progress_print = datetime.now()
    
    try:
        while episode < max_episodes and not (is_converged and is_beating_all):
            episode += 1
            
            # 每30分钟保存一次进度数据，无论保存间隔如何
            if (datetime.now() - last_save_time).total_seconds() > 3600:  # 60分钟
                stats_manager.save(f"training_stats_autosave.json")
                stats_manager.plot_win_rates(save_path=os.path.join(stats_manager.save_dir, "win_rates.png"))
                last_save_time = datetime.now()
            
            # 每5分钟打印一次进度，无论其他间隔如何
            if (datetime.now() - last_progress_print).total_seconds() > 600:  # 10分钟
                stats_manager.print_progress()
                last_progress_print = datetime.now()
            
            # 更新训练阶段
            opponent_manager.update_stage(is_beating_all, is_converged)
            
            # 选择本局对手
            opponents = opponent_manager.select_opponents(env, ppo_agent_0.device)
            opponent_1, opponent_1_name = opponents[0]
            opponent_3, opponent_3_name = opponents[1]
            
            # 清空智能体的历史记录
            if hasattr(ppo_agent_0, 'step_history'):
                ppo_agent_0.step_history = []
            if hasattr(ppo_agent_2, 'step_history'):
                ppo_agent_2.step_history = []
            
            # 设置环境的智能体
            env.set_agents([ppo_agent_0, opponent_1, ppo_agent_2, opponent_3])
            
            # 打印当前对手信息（每500局）
            if episode % 500 == 0:
                print(f"\nEpisode {episode} - 对手: [{opponent_1_name}, {opponent_3_name}] - 阶段: {opponent_manager.stage}")
            
            try:
                # 运行一个完整的游戏回合
                trajectories, _, winner_team = env.run(episode, is_training=True)
                
                # 处理收集的经验，为每一步计算奖励并存储到缓冲区
                episode_reward_0 = process_agent_experience(ppo_agent_0, winner_team, 0)
                episode_reward_2 = process_agent_experience(ppo_agent_2, winner_team, 2)
                
                # 平均奖励
                avg_reward = (episode_reward_0 + episode_reward_2) / 2
                
                # 更新统计
                stats_manager.update(
                    reward=avg_reward, 
                    win=(winner_team == 0),  # 0队是玩家0和2
                    opponent_names=[opponent_1_name, opponent_3_name]
                )
                
            except Exception as e:
                print(f"回合 {episode} 运行出错: {e}")
                continue  # 跳过本回合的其余部分
            
            try:
                # 更新PPO智能体的策略 - 只在达到阈值时更新
                update_threshold = 1024  # 设置一个样本阈值，如1024个转移样本
                
                # 根据内存中样本数量决定是否更新
                if hasattr(ppo_agent_0, 'memory') and len(ppo_agent_0.memory) >= update_threshold:
                    ppo_agent_0.update_policy()
                    if episode % 100 == 0:
                        print(f"玩家0更新策略，共{len(ppo_agent_0.memory)}个样本")
                
                if hasattr(ppo_agent_2, 'memory') and len(ppo_agent_2.memory) >= update_threshold:
                    ppo_agent_2.update_policy()
                    if episode % 100 == 0:
                        print(f"玩家2更新策略，共{len(ppo_agent_2.memory)}个样本")
            except Exception as e:
                print(f"更新策略时出错: {e}")
            
            # 保存模型
            if episode % save_interval == 0:
                # 保存模型代码...
                ppo_agent_0_save_path = f"{save_path}_player0_episode{episode}.pt"
                ppo_agent_2_save_path = f"{save_path}_player2_episode{episode}.pt"
                ppo_agent_0.save_model(ppo_agent_0_save_path)
                ppo_agent_2.save_model(ppo_agent_2_save_path)
                print(f"模型已保存: {ppo_agent_0_save_path} 和 {ppo_agent_2_save_path}")
                
                # 保存检查点
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode{episode}.tar")
                torch.save({
                    'episode': episode,
                    'ppo_agent_0_state_dict': ppo_agent_0.network.state_dict(),
                    'ppo_agent_2_state_dict': ppo_agent_2.network.state_dict(),
                    'stats': stats_manager.progress_log,
                    'is_beating_all': is_beating_all,
                    'is_converged': is_converged
                }, checkpoint_path)
                
                # 更新统计信息
                stats_manager.save(f"training_stats_episode{episode}.json")
                stats_manager.plot_win_rates(save_path=os.path.join(stats_manager.save_dir, f"win_rates_episode{episode}.png"))
            
            # 评估智能体
            if episode % evaluate_interval == 0:
                try:
                    # 保存当前训练模式
                    original_train_mode_0 = getattr(ppo_agent_0, 'train_mode', True)
                    original_train_mode_2 = getattr(ppo_agent_2, 'train_mode', True)
                    
                    # 切换到评估模式
                    setattr(ppo_agent_0, 'train_mode', False)
                    setattr(ppo_agent_2, 'train_mode', False)
                    
                    # 创建评估对手集
                    eval_opponents = opponent_manager.create_evaluation_opponents(ppo_agent_0.device)
                    
                    # 详细评估结果
                    detailed_results = {}
                    total_games = 0
                    total_wins = 0
                    total_reward = 0
                    
                    # 对每种对手类型进行评估
                    for opponent_type, opponents_list in eval_opponents.items():
                        if not opponents_list or len(opponents_list) < 2:
                            continue  # 跳过无效对手类型
                            
                        # 确保有两个对手 (1号和3号位)
                        opponent_1 = opponents_list[0]
                        opponent_3 = opponents_list[1] if len(opponents_list) > 1 else opponents_list[0]
                        
                        # 设置环境
                        env.set_agents([ppo_agent_0, opponent_1, ppo_agent_2, opponent_3])
                        
                        # 本类型对手评估统计
                        wins = 0
                        games = 0
                        reward_sum = 0
                        
                        # 进行多局评估
                        num_games = evaluate_num // len(eval_opponents) + 1  # 确保每种对手至少一局
                        for _ in range(num_games):
                            # 重置环境和智能体
                            eval_state, _ = env.reset()
                            ppo_agent_0.reset()
                            ppo_agent_2.reset()
                            
                            eval_done = False
                            eval_reward = 0
                            while not eval_done:
                                eval_next_state, r, eval_done, eval_info = env.step(None)
                                eval_reward += r
                                eval_state = eval_next_state
                            
                            # 更新统计
                            games += 1
                            if eval_info and 'winners' in eval_info and any(p in eval_info['winners'] for p in TEAM_0_PLAYERS):
                                wins += 1
                            reward_sum += eval_reward
                        
                        # 计算本类型的胜率和平均奖励
                        win_rate = wins / games if games > 0 else 0
                        avg_reward = reward_sum / games if games > 0 else 0
                        
                        # 添加到详细结果
                        detailed_results[opponent_type] = {
                            'win_rate': win_rate,
                            'games': games,
                            'avg_reward': avg_reward
                        }
                        
                        # 更新总体统计
                        total_games += games
                        total_wins += wins
                        total_reward += reward_sum
                        
                        print(f"评估结果 - {opponent_type}: 胜率 = {win_rate:.4f} ({wins}/{games})")
                    
                    # 计算总体胜率和平均奖励
                    overall_win_rate = total_wins / total_games if total_games > 0 else 0
                    overall_avg_reward = total_reward / total_games if total_games > 0 else 0
                    
                    # 汇总结果
                    eval_win_rates = {
                        'overall': overall_win_rate,
                        'by_type': {k: v['win_rate'] for k, v in detailed_results.items()}
                    }
                    
                    # 添加评估结果到统计管理器
                    stats_manager.add_evaluation_result(
                        win_rates=eval_win_rates,
                        avg_reward=overall_avg_reward,
                        breakdown=detailed_results
                    )
                    
                    print(f"\n===== 评估总结 (回合 {episode}) =====")
                    print(f"总体胜率: {overall_win_rate:.4f} ({total_wins}/{total_games})")
                    print(f"平均奖励: {overall_avg_reward:.4f}")
                    print("=====================================\n")
                    
                    # 打印详细的训练进度
                    stats_manager.print_progress()
                    
                    # 如果胜率超过阈值，将模型添加到模型池
                    if overall_win_rate >= pool_threshold:
                        pool_model_path = f"{save_path}_pool_episode{episode}.pt"
                        ppo_agent_0.save_model(pool_model_path)
                        opponent_manager.add_model_to_pool(pool_model_path, overall_win_rate)
                        print(f"模型已添加到模型池，胜率: {overall_win_rate:.4f}")
                    
                    # 更新目标达成状态
                    old_beating = is_beating_all
                    is_beating_all = stats_manager.is_beating_all_models(win_rate_threshold=beating_threshold)
                    
                    if is_beating_all and not old_beating:
                        print(f"\n>>> 已达成第一阶段目标: 击败所有规则模型 <<<\n")
                        # 保存此阶段的模型
                        milestone_path = f"{save_path}_milestone_beating_all.pt"
                        ppo_agent_0.save_model(milestone_path)
                    
                    # 检查是否收敛
                    is_converged_now = stats_manager.is_converged(
                        threshold=convergence_threshold, 
                        window_count=patience
                    )
                    
                    if is_converged_now and not is_converged:
                        convergence_counter += 1
                        
                        if convergence_counter >= patience:
                            is_converged = True
                            print(f"\n>>> 模型训练已收敛! <<<\n")
                            # 保存收敛时的模型
                            convergence_path = f"{save_path}_converged.pt"
                            ppo_agent_0.save_model(convergence_path)
                    else:
                        convergence_counter = 0
                    
                    # 恢复训练模式
                    setattr(ppo_agent_0, 'train_mode', original_train_mode_0)
                    setattr(ppo_agent_2, 'train_mode', original_train_mode_2)
                except Exception as e:
                    print(f"执行评估时出错: {e}")
                    import traceback
                    traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n手动中断训练，保存当前状态...")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 训练结束，执行最终策略更新
    try:
        if hasattr(ppo_agent_0, 'memory') and len(ppo_agent_0.memory) > 0:
            ppo_agent_0.update_policy()
        if hasattr(ppo_agent_2, 'memory') and len(ppo_agent_2.memory) > 0:
            ppo_agent_2.update_policy()
        print("执行最终策略更新")
    except Exception as e:
        print(f"执行最终策略更新时出错: {e}")
    
    # 训练结束，保存最终模型
    final_save_path_0 = f"{save_path}_player0_final.pt"
    final_save_path_2 = f"{save_path}_player2_final.pt"
    ppo_agent_0.save_model(final_save_path_0)
    ppo_agent_2.save_model(final_save_path_2)
    print(f"最终模型已保存: {final_save_path_0} 和 {final_save_path_2}")
    
    # 保存最终训练统计数据
    stats_manager.save("training_stats_final.json")
    
    # 绘制最终训练曲线
    stats_manager.plot_win_rates(
        save_path=os.path.join(stats_manager.save_dir, "final_win_rates.png"),
        show=False
    )
    
    # 打印最终统计摘要
    total_time = datetime.now() - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n\n===== 训练完成 =====")
    print(f"总回合数: {episode}")
    print(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print(f"最终胜率: {stats_manager.win_rates[-1]:.4f}")
    print(f"击败所有规则模型: {'是' if is_beating_all else '否'}")
    print(f"模型收敛: {'是' if is_converged else '否'}")
    print(f"最佳模型: 回合 {stats_manager.best_model_episode}，胜率 {stats_manager.best_win_rate:.4f}")
    print("=====================")
    
    return {
        'episodes': episode,
        'training_time': str(total_time),
        'final_win_rate': stats_manager.win_rates[-1],
        'is_beating_all': is_beating_all,
        'is_converged': is_converged
    }

def process_agent_experience(agent, winner_team, player_id):
    """
    处理单个智能体的经验并计算总奖励
    
    参数:
    agent: PPO智能体对象
    winner_team: 获胜队伍ID
    player_id: 玩家ID
    
    返回:
    float: 回合总奖励
    """
    if not hasattr(agent, 'step_history') or not agent.step_history:
        return 0.0
    
    total_reward = 0.0
    
    # 为每一步计算奖励并存储到经验回放缓冲区
    for i in range(len(agent.step_history)):
        step_info = agent.step_history[i]
        
        # 获取下一步信息（如果存在）
        next_step_info = agent.step_history[i+1] if i+1 < len(agent.step_history) else None
        next_state = next_step_info['raw_state'] if next_step_info else None
        
        # 判断是否为终止状态
        done = (i+1 >= len(agent.step_history))
        
        # 计算奖励
        reward = agent.reward_shaper.calculate_reward(
            state=step_info['raw_state'],
            action=step_info['action'],
            next_state=next_state,
            done=done,
            info={'winner_team': winner_team} if done else None,
            player_id=player_id
        )
        
        total_reward += reward
        
        # 存储完整的经验（使用对数概率）
        try:
            agent.memory.store(
                step_info['state_tensor'],
                step_info['history_tensor'],
                step_info['action_idx'],
                step_info['log_prob'],  # 使用对数概率
                step_info['value'],
                reward,
                int(done),
                step_info['action_mask']
            )
        except Exception as e:
            print(f"存储玩家{player_id}的经验时出错: {e}")
    
    return total_reward

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='掼蛋PPO智能体持续训练直至收敛')
    parser.add_argument('--max-episodes', type=int, default=100000, help='最大训练回合数')
    parser.add_argument('--save-interval', type=int, default=500, help='保存模型的间隔回合数')
    parser.add_argument('--eval-interval', type=int, default=1000, help='评估的间隔回合数')
    parser.add_argument('--eval-num', type=int, default=100, help='每次评估的对局数')
    parser.add_argument('--model-path', type=str, default='models/ppo_guandan_conv', help='模型保存路径')
    parser.add_argument('--device', type=str, default='auto', help='计算设备: auto, cpu, cuda')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use-transformer', action='store_true', help='使用Transformer模型替代LSTM')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    parser.add_argument('--pool-threshold', type=float, default=0.6, help='添加到模型池的胜率阈值')
    parser.add_argument('--pool-size', type=int, default=5, help='模型池最大容量')
    parser.add_argument('--beating-threshold', type=float, default=0.55, help='击败规则模型的胜率阈值')
    parser.add_argument('--convergence-threshold', type=float, default=0.01, help='收敛判定阈值')
    parser.add_argument('--patience', type=int, default=3, help='收敛确认所需的连续评估次数')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--stats-dir', type=str, default='training_stats', help='统计数据保存目录')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    print(f"使用设备: {device}")
    print(f"网络模型: {'Transformer' if args.use_transformer else 'LSTM'}")
    
    # 设置日志
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # 创建RLCard环境
    config = {'seed': args.seed, 'allow_step_back': True}
    env = GuandanEnv(config)
    random_state = np.random.RandomState(args.seed)
    
    # 创建PPO智能体(0号和2号位)
    ppo_agent_0 = PPOGuandanAgent(
        player_id=0,
        np_random=random_state,
        state_dim=256,  # 更新为优化后的状态维度
        action_dim=300,  # action_dim设为300
        device=device,
        use_transformer=args.use_transformer,  # 设置是否使用Transformer
        verbose=args.verbose
    )
    
    ppo_agent_2 = PPOGuandanAgent(
        player_id=2,
        np_random=random_state,
        state_dim=256,  # 更新为优化后的状态维度
        action_dim=300,
        device=device,
        use_transformer=args.use_transformer,  # 设置是否使用Transformer
        verbose=args.verbose
    )
    
    # 创建统计管理器
    stats_manager = TrainingStats(
        window_size=100,
        save_dir=args.stats_dir
    )
    
    # 创建对手池管理器
    opponent_manager = OpponentPoolManager(
        random_state=random_state,
        max_model_pool_size=args.pool_size
    )
    
    # 是否从检查点恢复
    if args.resume:
        if os.path.exists(args.resume):
            print(f"从检查点 {args.resume} 恢复训练...")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 加载模型状态
            ppo_agent_0.network.load_state_dict(checkpoint['ppo_agent_0_state_dict'])
            ppo_agent_2.network.load_state_dict(checkpoint['ppo_agent_2_state_dict'])
            
            # 恢复统计数据
            if 'stats' in checkpoint:
                stats_manager.progress_log = checkpoint['stats']
                
            # 恢复目标达成状态
            is_beating_all = checkpoint.get('is_beating_all', False)
            is_converged = checkpoint.get('is_converged', False)
            
            print(f"已恢复到回合 {checkpoint.get('episode', 0)}")
        else:
            print(f"检查点 {args.resume} 不存在，从头开始训练")
    
    print(f"PPO智能体已创建，开始持续训练至收敛...")
    
    # 使用持续训练
    result = train_until_convergence(
        env=env,
        ppo_agent_0=ppo_agent_0,
        ppo_agent_2=ppo_agent_2,
        opponent_manager=opponent_manager,
        stats_manager=stats_manager,
        max_episodes=args.max_episodes,
        save_path=args.model_path,
        save_interval=args.save_interval,
        evaluate_interval=args.eval_interval,
        evaluate_num=args.eval_num,
        pool_threshold=args.pool_threshold,
        beating_threshold=args.beating_threshold,
        convergence_threshold=args.convergence_threshold,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print("训练已完成或中断!")
    print(f"总回合数: {result['episodes']}")
    print(f"总训练时间: {result['training_time']}")
    print(f"最终胜率: {result['final_win_rate']:.4f}")
    print(f"打败所有规则模型: {'是' if result['is_beating_all'] else '否'}")
    print(f"模型是否收敛: {'是' if result['is_converged'] else '否'}")

if __name__ == "__main__":
    main()
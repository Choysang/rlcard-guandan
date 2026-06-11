import os
import sys
# 添加路径修复导入问题
import torch
import numpy as np
import argparse
import json
from datetime import datetime
from .ppo_agent import PPOGuandanAgent
from .utils import plot_evaluation_results, plot_rule_models_comparison

from guandan_rlcard.game.player import GuandanPlayer
from guandan_rlcard.envs.guandan_env import GuandanEnv
# 导入模型和状态编码器 - 更新导入
from guandan_rlcard.baselines.perfectdan.models import GuandanLSTMPolicyNetwork, GuandanValueNetwork, OptimizedGuandanStateEncoder

# 导入规则模型
from guandan_rlcard.baselines.random_agent import RandomAgent
from guandan_rlcard.baselines.rule_based.base1.base1_agent import Base1Agent
from guandan_rlcard.baselines.rule_based.base3.base3_agent import Base3Agent
from guandan_rlcard.baselines.rule_based.base4.base4_agent import Base4Agent
from guandan_rlcard.baselines.rule_based.base5.base5_agent import Base5Agent
from guandan_rlcard.baselines.rule_based.base6.base6_agent import Base6Agent
from guandan_rlcard.baselines.rule_based.base7.base7_agent import Base7Agent
from guandan_rlcard.baselines.rule_based.base8.base8_agent import Base8Agent

def set_seed(seed):
    """设置随机种子确保实验可重复"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_rule_agent_by_name(name, player_id, random_state):
    """根据名称获取相应的规则智能体"""
    rule_agents = {
        "Random": RandomAgent,
        "Base1": Base1Agent,
        "Base3": Base3Agent,
        "Base4": Base4Agent,
        "Base5": Base5Agent,
        "Base6": Base6Agent,
        "Base7": Base7Agent,
        "Base8": Base8Agent
    }
    
    if name in rule_agents:
        return rule_agents[name](player_id, random_state)
    else:
        print(f"未知规则智能体: {name}，使用随机智能体代替")
        return RandomAgent(player_id, random_state)

def load_ppo_agents(model_path, device):
    """
    加载由 train.py 生成的 PPO 模型，并创建智能体。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    ppo_agents = []
    random_state = np.random.RandomState(42) # 评估时使用固定种子

    for i in range(4):
        # 使用与 train.py 完全一致的参数创建 Agent
        agent = PPOGuandanAgent(
            player_id=i,
            np_random=random_state,
            state_dim=673,
            action_dim=143,
            abstract_action_dim=336,
            perfect_info_dim=165,
            device=device,
            verbose=(i == 0) # 只让0号玩家打印详细日志
        )
        
        # 加载模型权重
        policy_key = f'policy_player{i}'
        value_key = f'value_player{i}'

        if policy_key in checkpoint:
            agent.policy_network.load_state_dict(checkpoint[policy_key])
        else:
            # 兼容旧的单模型文件
            if 'policy_network_state_dict' in checkpoint:
                agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            else:
                print(f"警告: 在 {model_path} 中找不到玩家 {i} 的策略网络权重。")

        if value_key in checkpoint:
            agent.value_network.load_state_dict(checkpoint[value_key])
        else:
            if 'value_network_state_dict' in checkpoint:
                agent.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            else:
                print(f"警告: 在 {model_path} 中找不到玩家 {i} 的价值网络权重。")
        
        # 设置为评估模式
        agent.train_mode = False
        agent.policy_network.eval()
        agent.value_network.eval()
        
        ppo_agents.append(agent)
        
    print("所有PPO智能体加载并设置完成。")
    return ppo_agents

def evaluate_round(env, agents):
    """运行一局完整的游戏并返回结果"""
    state, player_id = env.reset()
    
    # 重置所有智能体
    for agent in agents:
        if hasattr(agent, 'reset'):
            agent.reset()

    # 游戏循环
    while not env.is_over():
        current_agent = agents[player_id]
        action = current_agent.step(state)
        state, player_id = env.step(action)

    # 获取游戏结果
    payoffs = env.get_payoffs()
    winner_team = 0 if payoffs[0] > 0 else 1
    return payoffs, winner_team, len(env.game.trace)

def evaluate_against_opponent(env, ppo_agents, opponent_name, num_games=100, team_mode=True):
    """
    评估PPO智能体对抗特定对手的性能。
    """
    wins_team0 = 0
    total_rewards_team0 = 0
    game_steps_list = []
    
    random_state = np.random.RandomState(42)

    for game in range(num_games):
        # 根据团队模式配置当场游戏的智能体
        if team_mode:
            # 0, 2 是 PPO； 1, 3 是规则模型
            agents = [
                ppo_agents[0],
                get_rule_agent_by_name(opponent_name, 1, random_state),
                ppo_agents[2],
                get_rule_agent_by_name(opponent_name, 3, random_state)
            ]
        else:
            # 0 是 PPO； 1, 2, 3 是规则模型
            agents = [
                ppo_agents[0],
                get_rule_agent_by_name(opponent_name, 1, random_state),
                get_rule_agent_by_name(opponent_name, 2, random_state),
                get_rule_agent_by_name(opponent_name, 3, random_state)
            ]
        
        env.set_agents(agents)
        payoffs, winner_team, steps = evaluate_round(env, agents)
        
        if winner_team == 0:
            wins_team0 += 1
        
        total_rewards_team0 += payoffs[0]
        game_steps_list.append(steps)

        if (game + 1) % 10 == 0:
            print(f"  对抗 {opponent_name} 进度: {game + 1}/{num_games}, 当前胜率: {wins_team0 / (game + 1):.4f}")

    win_rate = wins_team0 / num_games
    avg_reward = total_rewards_team0 / num_games
    avg_steps = np.mean(game_steps_list)

    return {
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'total_wins': wins_team0,
        'total_games': num_games
    }

def evaluate_agent(env, ppo_agent, opponent_name, num_games=100, team_mode=True):
    """
    评估PPO智能体的性能
    
    参数:
    env: 游戏环境
    ppo_agent: PPO智能体列表 [player_0, player_2]
    opponent_name: 对手类型名称
    num_games: 评估游戏局数
    team_mode: 是否使用团队模式（0、2位置vs 1、3位置）
    
    返回:
    results: 评估结果
    """
    # 重置统计
    total_games = 0
    total_wins = 0
    rewards = []
    
    random_state = np.random.RandomState(42)
    
    # 根据团队模式设置智能体
    if team_mode:
        # 团队模式：0和2位置是PPO智能体，1和3位置是规则智能体
        ppo_agent_0 = ppo_agent[0]
        ppo_agent_2 = ppo_agent[1]
        
        opponent_1 = get_rule_agent_by_name(opponent_name, 1, random_state)
        opponent_3 = get_rule_agent_by_name(opponent_name, 3, random_state)
        
        # 设置环境的智能体
        env.set_agents([ppo_agent_0, opponent_1, ppo_agent_2, opponent_3])
    else:
        # 单人模式：只在0位置使用PPO智能体，其他位置使用对手智能体
        ppo_agent_0 = ppo_agent[0]
        
        opponent_1 = get_rule_agent_by_name(opponent_name, 1, random_state)
        opponent_2 = get_rule_agent_by_name(opponent_name, 2, random_state)
        opponent_3 = get_rule_agent_by_name(opponent_name, 3, random_state)
        
        # 设置环境的智能体
        env.set_agents([ppo_agent_0, opponent_1, opponent_2, opponent_3])
    
    # 评估循环
    for game in range(num_games):
        # 重置环境
        state, _ = env.reset()
        
        # 重置智能体
        if team_mode:
            ppo_agent_0.reset()
            ppo_agent_2.reset()
        else:
            ppo_agent_0.reset()
        
        # 游戏循环
        done = False
        game_reward = 0
        
        while not done:
            # 运行一步
            _, reward, done, info = env.step(None)
            game_reward += reward
        
        # 记录结果
        total_games += 1
        if team_mode:
            if 'winners' in info and (0 in info['winners'] or 2 in info['winners']):
                total_wins += 1
        else:
            if 'winners' in info and 0 in info['winners']:
                total_wins += 1
        
        rewards.append(game_reward)
        
        # 打印进度
        if (game + 1) % 10 == 0:
            print(f"评估进度 {game + 1}/{num_games}, "
                  f"当前胜率: {total_wins / total_games:.4f}")
    
    # 计算最终结果
    win_rate = total_wins / total_games
    avg_reward = sum(rewards) / total_games
    
    results = {
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'max_reward': max(rewards),
        'min_reward': min(rewards),
        'std_reward': np.std(rewards),
        'rewards': rewards,
        'total_games': total_games,
        'total_wins': total_wins
    }
    
    return results

def evaluate_against_all_rule_models(env, ppo_agent, num_games=100, team_mode=True):
    """
    评估PPO智能体对抗所有规则模型的性能
    
    参数:
    env: 游戏环境
    ppo_agent: PPO智能体列表 [player_0, player_2]
    num_games: 每种对手的评估游戏局数
    team_mode: 是否使用团队模式（0、2位置vs 1、3位置）
    
    返回:
    results: 评估结果字典
    """
    # 规则模型列表
    rule_models = [
        "Base3",
        "Base1",
        "Base4",
        "Base7",
        "Base6",
        "Base8",
        "Base5",
        "Random"
    ]
    
    # 设置智能体为评估模式
    for agent in ppo_agent:
        if hasattr(agent, 'train_mode'):
            agent.train_mode = False
    
    # 结果字典
    all_results = {}
    
    # 总体统计
    total_games = 0
    total_wins = 0
    all_rewards = []
    all_avg_steps = []
    all_inference_speeds = []
    
    # 对每种规则模型进行评估
    for model_name in rule_models:
        print(f"\n开始对抗 {model_name} 的评估...")
        
        # 计算所有性能指标
        metrics = calculate_performance_metrics(
            env=env,
            ppo_agent=ppo_agent,
            opponent_name=model_name,
            num_games=num_games,
            team_mode=team_mode
        )
        
        # 获取基本评估结果（为了兼容性）
        basic_results = evaluate_agent(
            env=env,
            ppo_agent=ppo_agent,
            opponent_name=model_name,
            num_games=num_games,
            team_mode=team_mode
        )
        
        # 合并结果
        combined_results = {**basic_results}
        combined_results.update(metrics)
        
        # 保存结果
        all_results[model_name] = combined_results
        
        # 更新总体统计
        total_games += basic_results['total_games']
        total_wins += basic_results['total_wins']
        all_rewards.extend(basic_results['rewards'])
        all_avg_steps.append(metrics['avg_steps'])
        all_inference_speeds.append(metrics['inference_speed'])
        
        # 打印当前规则模型的结果
        print(f"对抗 {model_name} 的评估结果:")
        print(f"  胜率 (WR): {metrics['win_rate']:.4f} ({basic_results['total_wins']}/{basic_results['total_games']})")
        print(f"  期望得分 (ES): {metrics['expected_score']:.4f}")
        print(f"  平均出牌步数 (AS): {metrics['avg_steps']:.2f}")
        print(f"  推理速度: {metrics['inference_speed']:.2f} 推理/秒")
    
    # 计算总体结果
    overall_win_rate = total_wins / total_games if total_games > 0 else 0
    overall_avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    overall_avg_steps = sum(all_avg_steps) / len(all_avg_steps) if all_avg_steps else 0
    overall_avg_inference_speed = sum(all_inference_speeds) / len(all_inference_speeds) if all_inference_speeds else 0
    
    # 添加总体结果
    all_results['overall'] = {
        'win_rate': overall_win_rate,
        'avg_reward': overall_avg_reward,
        'max_reward': max(all_rewards) if all_rewards else 0,
        'min_reward': min(all_rewards) if all_rewards else 0,
        'std_reward': np.std(all_rewards) if all_rewards else 0,
        'rewards': all_rewards,
        'total_games': total_games,
        'total_wins': total_wins,
        'avg_steps': overall_avg_steps,
        'inference_speed': overall_avg_inference_speed
    }
    
    return all_results

def calculate_performance_metrics(env, ppo_agent, opponent_name, num_games=100, team_mode=True):
    """
    计算四个关键性能指标
    
    参数:
    env: 游戏环境
    ppo_agent: PPO智能体列表 [player_0, player_2]
    opponent_name: 对手名称
    num_games: 评估游戏局数
    team_mode: 是否使用团队模式
    
    返回:
    metrics: 包含四个关键指标的字典
    """
    # 进行评估获取基本结果
    results = evaluate_agent(env, ppo_agent, opponent_name, num_games, team_mode)
    
    # 收集游戏历史记录
    game_steps_list = []
    total_steps = 0
    
    # 设置智能体为评估模式
    for agent in ppo_agent:
        if hasattr(agent, 'train_mode'):
            agent.train_mode = False
    
    random_state = np.random.RandomState(42)
    
    # 根据团队模式设置智能体
    if team_mode:
        # 团队模式：0和2位置是PPO智能体，1和3位置是规则智能体
        ppo_agent_0 = ppo_agent[0]
        ppo_agent_2 = ppo_agent[1]
        
        opponent_1 = get_rule_agent_by_name(opponent_name, 1, random_state)
        opponent_3 = get_rule_agent_by_name(opponent_name, 3, random_state)
        
        # 设置环境的智能体
        env.set_agents([ppo_agent_0, opponent_1, ppo_agent_2, opponent_3])
    else:
        # 单人模式：只在0位置使用PPO智能体，其他位置使用对手智能体
        ppo_agent_0 = ppo_agent[0]
        
        opponent_1 = get_rule_agent_by_name(opponent_name, 1, random_state)
        opponent_2 = get_rule_agent_by_name(opponent_name, 2, random_state)
        opponent_3 = get_rule_agent_by_name(opponent_name, 3, random_state)
        
        # 设置环境的智能体
        env.set_agents([ppo_agent_0, opponent_1, opponent_2, opponent_3])
    
    # 再运行几局游戏专门收集步数数据
    metric_games = min(10, num_games)  # 用较少的游戏计算步数，避免过长时间
    
    for game in range(metric_games):
        # 重置环境
        state, _ = env.reset()
        
        # 重置智能体
        if team_mode:
            ppo_agent_0.reset()
            ppo_agent_2.reset()
        else:
            ppo_agent_0.reset()
        
        # 游戏循环
        done = False
        steps = 0
        
        while not done:
            # 运行一步
            _, _, done, _ = env.step(None)
            steps += 1
        
        # 记录步数
        game_steps_list.append(steps)
        total_steps += steps
    
    # 计算平均步数
    avg_steps = total_steps / metric_games if metric_games > 0 else 0
    
    # 测量推理速度
    inference_speed = ppo_agent[0].measure_inference_speed(env)
    
    # 1. 胜率 (已经在results中)
    win_rate = results['win_rate']
    
    # 2. 期望得分 (已经在results中，为avg_reward)
    expected_score = results['avg_reward']
    
    # 组合所有指标
    metrics = {
        'win_rate': win_rate,
        'expected_score': expected_score,
        'avg_steps': avg_steps,
        'inference_speed': inference_speed,
        'game_steps': game_steps_list
    }
    
    return metrics    

def main():
    parser = argparse.ArgumentParser(description='掼蛋PPO智能体评估脚本')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径前缀，不含player0或player2')
    parser.add_argument('--games', type=int, default=100, help='每种对手的评估游戏局数')
    parser.add_argument('--device', type=str, default='auto', help='计算设备: auto, cpu, cuda')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--opponent', type=str, default='all', help='对手类型，all表示评估所有规则模型')
    parser.add_argument('--save-results', action='store_true', help='保存评估结果')
    parser.add_argument('--team-mode', action='store_true', help='使用团队模式（0、2位置vs 1、3位置）')
    parser.add_argument('--use-wandb', action='store_true', help='是否使用Weights & Biases记录评估数据')
    parser.add_argument('--wandb-project', type=str, default='guandan-ppo-eval', help='Weights & Biases项目名称')
    parser.add_argument('--distributed-model', action='store_true', help='评估分布式训练模型')
    parser.add_argument('--model-version', type=int, default=None, help='分布式模型版本号')
    
    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(
            entity="choysun9527-sun-yat-sen-university",
            project=args.wandb_project,
            name=f"eval_{os.path.basename(args.model_path)}",
            config=vars(args),
            tags=["evaluation"]
        )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 创建RLCard环境
    config = {'seed': args.seed, 'allow_step_back': True}
    env = GuandanEnv(config)
    random_state = np.random.RandomState(args.seed)
    
    # 创建PPO智能体并加载模型
    ppo_agents = []
    
    # 处理分布式训练模型
    if args.distributed_model:
        from distributed_training import load_distributed_model
        
        # 加载分布式模型参数
        print(f"加载分布式训练模型: {args.model_path}")
        model_params = load_distributed_model(args.model_path, args.model_version)
        
        # 创建并初始化智能体
        for i in range(4):
            agent = PPOGuandanAgent(
                player_id=i,
                np_random=random_state,
                state_dim=256,
                action_dim=300,
                device=device,
                verbose=(i == 0)
            )
            
            # 加载模型参数
            if f'policy_player{i}' in model_params:
                agent.policy_network.load_state_dict(model_params[f'policy_player{i}'])
                print(f"已加载player{i}的策略网络参数")
                
            if f'value_player{i}' in model_params:
                agent.value_network.load_state_dict(model_params[f'value_player{i}'])
                print(f"已加载player{i}的价值网络参数")
                
            ppo_agents.append(agent)
    else:
        # 原有的模型加载逻辑
        model_path_0 = f"{args.model_path}_player0.pt"
        model_path_2 = f"{args.model_path}_player2.pt"
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path_0):
            print(f"错误: 模型文件不存在 - {model_path_0}")
            return
        
        ppo_agent_0 = PPOGuandanAgent(
            player_id=0,
            np_random=random_state,
            state_dim=256,
            action_dim=300,
            model_path=model_path_0,
            device=device
        )
        
        print(f"已加载模型: {model_path_0}")
        
        # 如果是团队模式，也加载player2的模型
        if args.team_mode:
            if os.path.exists(model_path_2):
                ppo_agent_2 = PPOGuandanAgent(
                    player_id=2,
                    np_random=random_state,
                    state_dim=256,
                    action_dim=300,
                    model_path=model_path_2,
                    device=device
                )
                print(f"已加载模型: {model_path_2}")
            else:
                print(f"警告: 模型文件不存在 - {model_path_2}")
                print("将使用player0的模型参数复制给player2")
                
                ppo_agent_2 = PPOGuandanAgent(
                    player_id=2,
                    np_random=random_state,
                    state_dim=256,
                    action_dim=300,
                    device=device
                )
                ppo_agent_2.network.load_state_dict(ppo_agent_0.network.state_dict())
            
            ppo_agents = [ppo_agent_0, ppo_agent_2]
        else:
            ppo_agents = [ppo_agent_0]
    
    # 设置环境智能体
    env.set_agents(ppo_agents)
    
    # 评估智能体
    if args.opponent == 'all':
        print(f"开始对抗所有规则模型的评估，每种对手 {args.games} 局游戏")
        results = evaluate_against_all_rule_models(
            env=env,
            ppo_agent=ppo_agents,
            num_games=args.games,
            team_mode=args.team_mode
        )
    else:
        print(f"开始对抗 {args.opponent} 的评估，共 {args.games} 局游戏")
        results = evaluate_agent(
            env=env,
            ppo_agent=ppo_agents,
            opponent_name=args.opponent,
            num_games=args.games,
            team_mode=args.team_mode
        )
        results = {args.opponent: results}  # 包装成字典格式
    
    if args.use_wandb and 'overall' in results:
        wandb.log(results['overall'])
        for model_name, model_results in results.items():
            if model_name != 'overall':
                for k, v in model_results.items():
                    if k != 'rewards' and k != 'game_steps':  # 排除大型列表
                        wandb.log({f"{model_name}_{k}": v})

    # 打印评估结果
    print("\n===== 评估结果摘要 =====")
    if 'overall' in results:
        overall = results['overall']
        print(f"总体胜率 (WR): {overall['win_rate']:.4f} ({overall['total_wins']}/{overall['total_games']})")
        print(f"期望得分 (ES): {overall['avg_reward']:.4f}")
        print(f"平均出牌步数 (AS): {overall['avg_steps']:.2f}")
        print(f"推理速度: {overall['inference_speed']:.2f} 推理/秒")
        print("\n各规则模型评估结果:")
        for model_name, model_results in results.items():
            if model_name != 'overall':
                print(f"  {model_name}:")
                print(f"    胜率 (WR): {model_results['win_rate']:.4f} ({model_results['total_wins']}/{model_results['total_games']})")
                print(f"    期望得分 (ES): {model_results['avg_reward']:.4f}")
                print(f"    平均出牌步数 (AS): {model_results['avg_steps']:.2f}")
                print(f"    推理速度: {model_results['inference_speed']:.2f} 推理/秒")
    else:
        model_name = list(results.keys())[0]
        model_results = results[model_name]
        print(f"对抗 {model_name} 的评估结果:")
        print(f"  胜率 (WR): {model_results['win_rate']:.4f} ({model_results['total_wins']}/{model_results['total_games']})")
        print(f"  期望得分 (ES): {model_results['avg_reward']:.4f}")
        print(f"  平均出牌步数 (AS): {model_results['avg_steps']:.2f}")
        print(f"  推理速度: {model_results['inference_speed']:.2f} 推理/秒")
    print("=========================\n")
    
    # 保存评估结果
    if args.save_results:
        results_dir = 'evaluation_results'
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(args.model_path)
        mode = "team" if args.team_mode else "solo"
        results_path = os.path.join(results_dir, f"{model_name}_{args.opponent}_{mode}_{timestamp}.json")
        
        # 转换rewards列表为普通列表，以便JSON序列化
        results_copy = {}
        for key, value in results.items():
            value_copy = value.copy()
            value_copy['rewards'] = [float(r) for r in value_copy['rewards']]
            results_copy[key] = value_copy
        
        with open(results_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"评估结果已保存至: {results_path}")
        
        # 绘制评估结果图表
        if 'overall' in results:
            plot_rewards = results['overall']['rewards']
            win_rate = results['overall']['win_rate']
            title = f"对抗所有规则模型 ({mode}模式)"
            
            # 绘制柱状图比较不同规则模型的胜率
            comparison_path = os.path.join(results_dir, f"{model_name}_{args.opponent}_comparison_{mode}_{timestamp}.png")
            plot_rule_models_comparison(results, save_path=comparison_path, show=False)
            print(f"规则模型对比图表已保存至: {comparison_path}")
        else:
            model_name = list(results.keys())[0]
            plot_rewards = results[model_name]['rewards']
            win_rate = results[model_name]['win_rate']
            title = f"对抗 {model_name} ({mode}模式)"
        
        plot_path = os.path.join(results_dir, f"{model_name}_{args.opponent}_{mode}_{timestamp}.png")
        plot_evaluation_results(
            rewards=plot_rewards,
            win_rate=win_rate,
            title=title,
            save_path=plot_path,
            show=False
        )
        print(f"评估结果图表已保存至: {plot_path}")
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
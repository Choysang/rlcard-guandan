"""
训练可视化和评估工具
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import multiprocessing as mp
from datetime import datetime
import csv


def plot_training_progress(episode_rewards, win_rates, eval_win_rates=None, eval_timestamps=None, save_path=None, show=True):
    """
    绘制训练进度图表，包括评估结果
    
    参数:
    episode_rewards: 每局游戏的奖励列表
    win_rates: 每局游戏的胜率列表
    eval_win_rates: 评估阶段的胜率列表（可选）
    eval_timestamps: 评估时的回合数列表（可选）
    save_path: 图表保存路径，如果为None则不保存
    show: 是否显示图表
    """
    try:
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 获取数据
        episodes = list(range(1, len(win_rates) + 1))
        
        # 绘制奖励曲线
        ax1.plot(episodes, episode_rewards, 'b-', alpha=0.3)  # 原始数据，透明
        
        # 添加滑动平均线
        window_size = min(50, len(episode_rewards) // 10) if len(episode_rewards) > 50 else 1
        if window_size > 1:
            rewards_smoothed = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
            ax1.plot(episodes[window_size-1:], rewards_smoothed, 'b-', linewidth=2)
        
        ax1.set_ylabel('回合奖励')
        ax1.set_title('训练过程中的回合奖励')
        ax1.grid(True, alpha=0.3)
        
        # 绘制胜率曲线
        ax2.plot(episodes, win_rates, 'g-', linewidth=2, label='训练胜率')
        
        # 如果有评估数据，添加评估胜率点
        if eval_win_rates and eval_timestamps and len(eval_win_rates) > 0:
            ax2.scatter(eval_timestamps, eval_win_rates, color='r', s=50, marker='o', label='评估胜率')
            # 用虚线连接评估点
            if len(eval_win_rates) > 1:
                ax2.plot(eval_timestamps, eval_win_rates, 'r--', alpha=0.7)
        
        ax2.set_xlabel('回合数')
        ax2.set_ylabel('胜率')
        ax2.set_title('胜率变化')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 添加最终胜率标记
        final_win_rate = win_rates[-1]
        ax2.axhline(y=final_win_rate, color='g', linestyle='--', alpha=0.7)
        ax2.text(len(episodes) * 0.02, final_win_rate + 0.02, f'最终训练胜率: {final_win_rate:.2f}', color='g')
        
        if eval_win_rates and len(eval_win_rates) > 0:
            final_eval_win_rate = eval_win_rates[-1]
            ax2.text(len(episodes) * 0.02, final_eval_win_rate - 0.04, f'最终评估胜率: {final_eval_win_rate:.2f}', color='r')
        
        # 设置x轴为整数
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            print(f"训练进度图表已保存至 {save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"绘制训练进度图表时出错: {str(e)}")
        
    finally:
        plt.close('all')

def plot_evaluation_results(rewards, win_rate, title=None, save_path=None, show=True):
    """
    绘制评估结果图表
    
    参数:
    rewards: 评估局的奖励列表
    win_rate: 胜率
    title: 图表标题
    save_path: 图表保存路径，如果为None则不保存
    show: 是否显示图表
    """
    try:
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制奖励分布直方图
        plt.hist(rewards, bins=20, alpha=0.7, color='blue')
        
        # 添加平均线
        avg_reward = np.mean(rewards)
        plt.axvline(x=avg_reward, color='r', linestyle='--', linewidth=2)
        
        # 添加标题和标签
        chart_title = title if title else f'评估结果 - 胜率: {win_rate:.2f}'
        plt.title(chart_title)
        plt.xlabel('奖励值')
        plt.ylabel('局数')
        
        # 添加统计信息文本
        stats_text = (
            f'平均奖励: {avg_reward:.2f}\n'
            f'中位数: {np.median(rewards):.2f}\n'
            f'最小值: {np.min(rewards):.2f}\n'
            f'最大值: {np.max(rewards):.2f}\n'
            f'标准差: {np.std(rewards):.2f}\n'
            f'胜率: {win_rate:.2f}'
        )
        
        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            print(f"评估结果图表已保存至: {save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"绘制评估结果图表时出错: {str(e)}")
        
    finally:
        plt.close('all')

def plot_rule_models_comparison(results, save_path=None, show=True):
    """
    绘制对抗不同规则模型的胜率对比图
    
    参数:
    results: 评估结果字典，格式为 {'model_name': {'win_rate': float, ...}, ...}
    save_path: 图表保存路径，如果为None则不保存
    show: 是否显示图表
    """
    try:
        # 提取模型名称和胜率
        models = []
        win_rates = []
        
        # 排除overall
        for model_name, model_results in results.items():
            if model_name != 'overall':
                models.append(model_name)
                win_rates.append(model_results['win_rate'])
        
        # 对模型按照胜率排序
        sorted_indices = np.argsort(win_rates)
        models = [models[i] for i in sorted_indices]
        win_rates = [win_rates[i] for i in sorted_indices]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 创建水平条形图
        bars = plt.barh(models, win_rates, color='skyblue', alpha=0.8)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{win_rates[i]:.2f}', va='center')
        
        # 添加参考线
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
        
        # 如果有overall，添加为标题信息
        if 'overall' in results:
            overall_win_rate = results['overall']['win_rate']
            plt.title(f'对抗不同规则模型的胜率对比 (总体胜率: {overall_win_rate:.2f})')
        else:
            plt.title('对抗不同规则模型的胜率对比')
        
        plt.xlabel('胜率')
        plt.ylabel('规则模型')
        plt.xlim(0, 1.1)  # 设置x轴范围，留出空间显示数值
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            print(f"规则模型对比图表已保存至: {save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"绘制规则模型对比图表时出错: {str(e)}")
        
    finally:
        plt.close('all')

def plot_performance_metrics(metrics_dict, save_path=None, show=True):
    """
    绘制四个关键性能指标的对比图
    
    参数:
    metrics_dict: 包含不同模型的四个关键指标的字典
    save_path: 图表保存路径
    show: 是否显示图表
    """
    try:
        # 提取模型名称和指标
        models = []
        win_rates = []
        expected_scores = []
        avg_steps = []
        inference_speeds = []
        
        # 排除overall
        for model_name, model_metrics in metrics_dict.items():
            if model_name != 'overall':
                models.append(model_name)
                win_rates.append(model_metrics.get('win_rate', 0))
                expected_scores.append(model_metrics.get('avg_reward', 0))
                avg_steps.append(model_metrics.get('avg_steps', 0))
                inference_speeds.append(model_metrics.get('inference_speed', 0))
        
        # 创建2x2子图布局
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 胜率子图
        axs[0, 0].bar(models, win_rates, color='blue', alpha=0.7)
        axs[0, 0].set_title('胜率 (WR)')
        axs[0, 0].set_ylabel('胜率')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].set_xticklabels(models, rotation=45, ha='right')
        
        # 期望得分子图
        axs[0, 1].bar(models, expected_scores, color='green', alpha=0.7)
        axs[0, 1].set_title('期望得分 (ES)')
        axs[0, 1].set_ylabel('得分')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].set_xticklabels(models, rotation=45, ha='right')
        
        # 平均出牌步数子图
        axs[1, 0].bar(models, avg_steps, color='red', alpha=0.7)
        axs[1, 0].set_title('平均出牌步数 (AS)')
        axs[1, 0].set_ylabel('步数')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].set_xticklabels(models, rotation=45, ha='right')
        
        # 推理速度子图
        axs[1, 1].bar(models, inference_speeds, color='purple', alpha=0.7)
        axs[1, 1].set_title('推理速度 (推理/秒)')
        axs[1, 1].set_ylabel('推理/秒')
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].set_xticklabels(models, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            print(f"性能指标图表已保存至: {save_path}")
        
        # 显示图表
        if show:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"绘制性能指标图表时出错: {str(e)}")
        
    finally:
        plt.close('all')

def print_training_summary(stats):
    """
    打印训练结果摘要
    
    参数:
    stats: 训练统计数据
    """
    print("\n===== 训练结果摘要 =====")
    
    # 获取统计数据
    episode_rewards = stats.get('episode_rewards', [])
    win_rates = stats.get('win_rates', [])
    avg_steps_list = stats.get('avg_steps', [])
    inference_speeds = stats.get('inference_speeds', [])
    
    if not episode_rewards and not win_rates:
        print("无可用的训练统计数据")
        return
    
    # 计算统计指标
    final_win_rate = win_rates[-1] if win_rates else 0
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    median_reward = np.median(episode_rewards) if episode_rewards else 0
    reward_std = np.std(episode_rewards) if episode_rewards else 0
    
    # 打印所有四个关键指标
    print(f"最终胜率 (WR): {final_win_rate:.4f}")
    print(f"平均奖励 (ES): {avg_reward:.4f}")
    print(f"奖励中位数: {median_reward:.4f}")
    print(f"奖励标准差: {reward_std:.4f}")
    
    # 打印平均出牌步数
    if avg_steps_list:
        avg_steps = np.mean(avg_steps_list)
        print(f"平均出牌步数 (AS): {avg_steps:.2f}")
    
    # 打印推理速度
    if inference_speeds:
        avg_inference_speed = np.mean(inference_speeds)
        print(f"平均推理速度: {avg_inference_speed:.2f} 推理/秒")
    
    # 计算性能提升
    if len(win_rates) >= 100:
        early_win_rate = np.mean(win_rates[:100])
        late_win_rate = np.mean(win_rates[-100:])
        win_rate_improvement = late_win_rate - early_win_rate
        print(f"胜率提升 (前100局 vs 后100局): {win_rate_improvement:.4f}")
    
    print("===== 训练完成 =====\n")

# 添加分布式训练配置函数
def get_distributed_config(config_name='single_machine'):
    """返回预设的分布式训练配置"""
    configs = {
        'single_machine': {
            'num_workers': max(1, mp.cpu_count() - 1),
            'batch_size': 1024,
            'buffer_capacity': 1000000,
            'optimize_cuda': True,
            'pin_memory': True
        },
        'multi_node': {
            'distributed': True,
            'num_workers': 6,
            'batch_size': 2048, 
            'buffer_capacity': 2000000,
            'optimize_cuda': True
        },
        'fine_tuning': {
            'num_workers': 8,
            'batch_size': 1024,
            'learning_rate': 1e-4,
            'entropy_coef': 0.005,
            'fine_tune': True
        }
    }
    return configs.get(config_name, configs['single_machine'])

# 添加训练监控和可视化函数
def monitor_training_metrics(metrics, output_dir='logs'):
    """监控并保存训练指标"""
    import os
    import csv
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加时间戳
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 记录到CSV
    metrics_file = os.path.join(output_dir, 'training_metrics.csv')
    
    # 检查文件是否存在
    file_exists = os.path.isfile(metrics_file)
    
    # 写入CSV
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    
    # 每10条记录创建一次可视化
    try:
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                reader = csv.reader(f)
                line_count = sum(1 for _ in reader)
            
            if line_count > 0 and line_count % 10 == 0:
                # 创建可视化
                generate_training_visualizations(metrics_file, output_dir)
    except Exception as e:
        print(f"创建可视化时出错: {e}")
    
    return metrics_file

def generate_training_visualizations(metrics_file, output_dir):
    """根据指标文件生成可视化图表"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 加载指标
    df = pd.read_csv(metrics_file)
    
    # 绘制训练进度图
    if 'total_steps' in df.columns and 'win_rate' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['total_steps'], df['win_rate'], 'b-')
        plt.title('训练胜率变化')
        plt.xlabel('训练步数')
        plt.ylabel('胜率')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'win_rate_progress.png'))
        plt.close()
    
    # 绘制损失图
    if 'policy_loss' in df.columns and 'value_loss' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['policy_loss'], 'r-', label='策略损失')
        plt.plot(df.index, df['value_loss'], 'g-', label='价值损失')
        plt.title('训练损失变化')
        plt.xlabel('记录索引')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'loss_progress.png'))
        plt.close()
import os, sys
#获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
#获取父目录的父目录(上两级目录)
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

from guandan_rlcard.game.player import GuandanPlayer as Player
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from enum import Enum
import random
import collections

# 导入模型和状态编码器
from guandan_rlcard.baselines.perfectdan.models import GuandanLSTMPolicyNetwork, GuandanValueNetwork, OptimizedGuandanStateEncoder

# 掼蛋游戏卡牌等级定义
CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
            'A', 'BJ', 'RJ']

SUITS = ['S', 'H', 'C', 'D']
CANONICAL_CARDS = [f'{s}{r}' for r in CARD_RANK[:-2] for s in SUITS] + ['BJ', 'RJ']
CARD_TO_INDEX = {card: i for i, card in enumerate(CANONICAL_CARDS)}

class PPOGuandanAgent(Player):
    """
    掼蛋游戏的PPO智能体。
    该智能体使用解耦的策略/价值网络，并依赖一个强大的StateEncoder来处理状态和历史信息。
    """
    def __init__(self, player_id, np_random, 
                 state_dim=673,
                 action_dim=143,
                 abstract_action_dim=336,
                 perfect_info_dim=165,
                 model_path=None,
                 device='auto',
                 verbose=False,
                 lr=3e-4,
                 use_lr_scheduler=True,
                 preserve_rnn_state=True,
                 fine_tuning=False):
        """
        初始化PPO掼蛋智能体。
        ...
        """
        if isinstance(player_id, dict):
            self.player_id = player_id['player_id']
        else:
            self.player_id = player_id
        
        super().__init__(self.player_id, np_random)
        
        self.use_raw = True
        self.verbose = verbose
        self.preserve_rnn_state = preserve_rnn_state  
        self.train_mode = True

        # 初始化日志记录器
        self.logger = logging.getLogger(f'ppo_agent_{self.player_id}')
        if verbose and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - P{self.player_id} - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        elif not verbose and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - P{self.player_id} - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)
        
        # 设置设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 存储维度信息
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.abstract_action_dim = abstract_action_dim
        self.perfect_info_dim = perfect_info_dim
        
        # 初始化状态编码器
        self.state_encoder = OptimizedGuandanStateEncoder(state_dim=state_dim, device=self.device)
        
        # 初始化RNN隐藏状态
        self.policy_hidden_state = None
        self.value_hidden_state = None

        # 初始化网络
        self.policy_network = GuandanLSTMPolicyNetwork(
            state_dim=state_dim, action_feature_dim=action_dim, abstract_action_dim=abstract_action_dim, 
            lstm_hidden_dim=256, ff_hidden_dim=512, num_lstm_layers=2, device=self.device
        ).to(self.device)
        
        self.value_network = GuandanValueNetwork(
            state_dim=state_dim, perfect_info_dim=perfect_info_dim,
            lstm_hidden_dim=256, ff_hidden_dim=512, num_lstm_layers=2, device=self.device
        ).to(self.device)
        
        # 初始化优化器
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        # 初始化学习率调度器
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            self.policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.policy_optimizer, mode='min', factor=0.5, patience=5, threshold=0.01, min_lr=1e-5)
            self.value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.value_optimizer, mode='min', factor=0.5, patience=5, threshold=0.01, min_lr=1e-5)
        
        # 初始化奖励整形器
        self.reward_shaper = RewardShaper(
            config={'win_reward': 2.0, 'lose_reward': -2.0, 'quick_win_bonus': 1.0},
            fine_tuning=fine_tuning, verbose=verbose
        )
        
        # 关键修复：初始化 history_buffer
        self.history_buffer = collections.deque(maxlen=40)
        
        # 初始化经验回放池
        self.memory = PPOMemory(batch_size=1024, max_size=4000, gamma=0.99, gae_lambda=0.95, device=self.device)
        
        # PPO超参数
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.05
        self.max_grad_norm = 0.5
        self.n_epochs = 10
        self.steps_per_update = 2000
        
        # 统计和状态变量
        self.stats = {'training_steps': 0, 'updates': 0, 'policy_loss': [], 'value_loss': [], 'entropy': []}
        self.game_steps = []
        self.current_state = None
        self.current_action = None
        
        # 加载预训练模型
        if model_path:
            self.load_model(model_path)
            if self.verbose:
                self.logger.info(f"已加载模型: {model_path}")

    def set_training_phase(self, current_step, total_steps, threshold=0.8):
        """
        根据训练进度调整训练参数（例如熵系数），在训练后期进入微调阶段。
        """
        should_fine_tune = current_step > total_steps * threshold
        current_fine_tuning = self.reward_shaper.fine_tuning
        
        if current_fine_tuning != should_fine_tune:
            self.logger.info(f"步数: {current_step}/{total_steps} - 训练阶段切换: {'标准训练' if current_fine_tuning else '微调'} -> {'微调' if should_fine_tune else '标准训练'}")
            self.reward_shaper.set_training_phase(should_fine_tune)
            if should_fine_tune:
                self.entropy_coef = 0.01
                self.logger.info(f"微调阶段参数调整: 熵系数 {self.entropy_coef}")
            else: # Back to standard
                self.entropy_coef = 0.05 # Reset to default or original value
                self.logger.info(f"标准训练阶段参数调整: 熵系数 {self.entropy_coef}")

    def step(self, state):
        """
        根据当前游戏状态选择一个动作。
        此方法是与环境交互的主入口，负责处理所有情况。
        """
        # 关键入口检查：处理所有不需要决策的场景
        # 场景1: 玩家已出完牌 (current_hand 为空)
        # 场景2: 轮到玩家，但没有任何合法动作 (actions 为空)
        if not state.get('current_hand') or not state.get('actions'):
            # 记录玩家已完成状态，以备后用
            if not state.get('current_hand'):
                self.has_finished = True
            
            # 在这些情况下，智能体无需决策，只能出 'PASS'
            # 我们模拟一个“空”决策结果，以保持数据结构一致性
            empty_state = torch.zeros(1, self.state_dim, device=self.device)
            empty_history = self.get_history_tensor()
            
            # 创建一个完整的、但代表“无操作”的 info 字典
            info = {
                'state': empty_state,
                'history_states': empty_history,
                'perfect_info': None,
                'action_idx': -1,
                'abstract_action_idx': -1,
                'log_prob': 0.0,
                'value': 0.0,
                'action_mask': torch.zeros(1, self.abstract_action_dim, device=self.device),
                'action_features': [],
                'next_policy_hidden': self.policy_hidden_state,
                'next_value_hidden': self.value_hidden_state,
                'raw_state': state
            }
            
            # 将这个“无操作”步骤记录下来
            self.current_state = self.parse(state)
            self.current_action = ['PASS', 'PASS', 'PASS']
            
            if not hasattr(self, 'step_history'):
                self.step_history = []
            
            step_info = {
                'raw_state': state, 'state_tensor': info['state'], 'history_tensor': info['history_states'],
                'action': self.current_action, 'action_idx': info['action_idx'], 'abstract_action_idx': info['abstract_action_idx'],
                'log_prob': info['log_prob'], 'value': info['value'], 'action_mask': info['action_mask'],
                'action_features': info['action_features'], 'next_policy_hidden': info['next_policy_hidden'],
                'next_value_hidden': info['next_value_hidden'], 'reward': 0.0, 'player_id': self.player_id,
                'step_index': len(self.step_history), 'has_finished': hasattr(self, 'has_finished') and self.has_finished
            }
            self.step_history.append(step_info)
            
            return ['PASS', 'PASS', 'PASS']

        # --- 如果通过了入口检查，说明这是一个需要做决策的正常步骤 ---
        
        # 1. 解析：一次性完成所有信息转换
        parsed_state = self.parse(state)
        
        # 2. 决策：将解析后的完整信息传入
        action, info = self.select_action(parsed_state, evaluation=not self.train_mode)
        
        # 3. 记录：使用决策过程返回的 info 字典填充历史记录
        self.current_state = parsed_state
        self.current_action = action
        
        if not hasattr(self, 'step_history'):
            self.step_history = []
        
        step_info = {
            'raw_state': state, 'state_tensor': info['state'], 'history_tensor': info['history_states'],
            'action': action, 'action_idx': info['action_idx'], 'abstract_action_idx': info['abstract_action_idx'],
            'log_prob': info['log_prob'], 'value': info['value'], 'action_mask': info['action_mask'],
            'action_features': info['action_features'], 'next_policy_hidden': info['next_policy_hidden'],
            'next_value_hidden': info['next_value_hidden'], 'reward': 0.0, 'player_id': self.player_id,
            'step_index': len(self.step_history), 'has_finished': hasattr(self, 'has_finished') and self.has_finished
        }
        self.step_history.append(step_info)
        return action
    
    def observe(self, next_state):
        """
        观察执行动作后的下一个状态。如果一轮（trick）结束，则计算并分配该轮的奖励。
        """
        if not hasattr(self, 'current_state') or self.current_state is None or \
           not hasattr(self, 'current_action') or self.current_action is None:
            return
        
        if self.verbose:
            self.logger.info(f"执行observe，当前动作: {self.current_action}")
        
        # 不再需要解析状态为 `next_parsed_state`，因为奖励计算直接使用原始字典
        trace = next_state.get('trace', [])
        turn_completed = next_state.get('round_completed', False)
        global_turn_count = next_state.get('global_turn_count', 0)
        turn_winner = next_state.get('turn_winner', -1)
        
        current_round_actions = []
        if turn_completed and len(trace) >= 3:
            lastindex = len(trace) - 1
            for i in range(len(trace) - 3, -1, -1):
                if (trace[i][1][0] == 'PASS' and trace[i+1][1][0] == 'PASS' and trace[i+2][1][0] == 'PASS'):
                    lastindex = i - 1; break
            firstindex = 0
            for i in range(lastindex - 2, -1, -1):
                if (trace[i][1][0] == 'PASS' and trace[i+1][1][0] == 'PASS' and trace[i+2][1][0] == 'PASS'):
                    firstindex = i + 3; break
            current_round_actions = trace[firstindex:lastindex+4]
            if self.verbose:
                self.logger.info(f"本轮动作轨迹 (len={len(current_round_actions)}): {current_round_actions}")
        
        if global_turn_count == 0 and self.reward_shaper.turn_count > 0:
            if self.verbose: self.logger.info(f"新小局开始，重置奖励累计. RewardShaper TC before: {self.reward_shaper.turn_count}")
            self.reward_shaper.reset() # 完全重置
        elif global_turn_count > 0 : # Sync if not first turn
            self.reward_shaper.turn_count = global_turn_count
        
        if turn_completed and current_round_actions:
            if self.verbose:
                self.logger.info(f"全局轮次 {global_turn_count} 完成，牌权获得者: {turn_winner}，计算奖励...")
            
            last_non_pass_player = None
            for p_id, act in reversed(current_round_actions):
                if act[0] != 'PASS': last_non_pass_player = p_id; break
            
            self.reward_shaper.action_history = [{'player_id': p_id, 'action': act, 'is_pass': act[0] == 'PASS'} for p_id, act in current_round_actions]
            
            if last_non_pass_player is not None:
                self.reward_shaper.current_turn_states.update({
                    'last_non_pass_player': last_non_pass_player,
                    'turn_actions': self.reward_shaper.action_history[:],
                    'all_players_hands': next_state.get('all_players_hands', {}) 
                })
            
            turn_rewards_for_all_players = self.calculate_turn_rewards() 
            
            if self.player_id in turn_rewards_for_all_players:
                my_turn_rewards_info = turn_rewards_for_all_players[self.player_id]
                if self.verbose:
                    self.logger.info(f"我的 ({self.player_id}) 轮次奖励详情: {my_turn_rewards_info}")
                
                if isinstance(my_turn_rewards_info, dict) and 'reward' in my_turn_rewards_info:
                    current_round_steps = [step for step in self.step_history if step['reward'] == 0.0 and step['player_id'] == self.player_id]
                    
                    if current_round_steps:
                        reward_to_assign = my_turn_rewards_info['reward']
                        allocation_policy = my_turn_rewards_info.get('allocation', 'last_non_pass_action')
                        
                        if allocation_policy == 'first_action' and current_round_steps[0]['player_id'] == self.player_id :
                            current_round_steps[0]['reward'] = reward_to_assign
                            if self.verbose: self.logger.info(f"分配奖励 {reward_to_assign:.4f} 给我的第一个步骤: {current_round_steps[0]['action']}")
                        elif allocation_policy == 'last_non_pass_action':
                            for step in reversed(current_round_steps):
                                if step['player_id'] == self.player_id and isinstance(step['action'], list) and step['action'][0] != 'PASS':
                                    step['reward'] = reward_to_assign
                                    if self.verbose: self.logger.info(f"分配奖励 {reward_to_assign:.4f} 给我的最后一个非PASS步骤: {step['action']}")
                                    break
        
        self.current_state = None
        self.current_action = None
    
    def parse(self, state):
        """
        将原始的游戏状态字典解析为智能体内部使用的、唯一的、完整的信息源。
        所有后续方法都应只依赖此方法返回的字典。
        """
        assert isinstance(state, dict), "State must be a dictionary"
        msg = {}
        
        # --- 基础信息和不完美信息 ---
        msg['curPos'] = self.player_id
        actions = state.get('actions', [])
        msg['actionList'] = actions
        msg['indexRange'] = len(actions) - 1 if actions else -1
        msg['greaterPos'] = state.get('greaterPos', -1)
        greater_action = state.get('greaterAction', [])
        msg['greaterAction'] = greater_action
        msg['curAction'] = greater_action
        rank_list = state.get('rank_list', [])
        play_team = state.get('play_team', 0)
        msg['curRank'] = CARD_RANK[rank_list[play_team]] if rank_list and len(rank_list) > play_team else '2'
        msg['handCards'] = state.get('current_hand', [])
        num_cards_left = state.get('num_cards_left', [27, 27, 27, 27])
        msg['remain_cards'] = num_cards_left
        pass_num = state.get('pass_num', [0, 0, 0, 0])
        my_pass_num = state.get('my_pass_num', [0, 0, 0, 0])
        msg['pass_num'] = pass_num[self.player_id] if len(pass_num) > self.player_id else 0
        msg['my_pass_num'] = my_pass_num[self.player_id] if len(my_pass_num) > self.player_id else 0
        played_cards = state.get('played_cards', {i: [] for i in range(4)})
        msg['history'] = {str(i): {'send': played_cards.get(i, []), 'remain': num_cards_left[i] if len(num_cards_left) > i else 0} for i in range(4)}
        msg['publicInfo'] = {i: {'rest': num_cards_left[i] if len(num_cards_left) > i else 0} for i in range(4)}

        # --- 完美信息 (如果存在) ---
        # 将完美信息也复制到 msg 字典中，供后续使用
        if 'all_players_hands' in state:
            msg['all_players_hands'] = state['all_players_hands']
        if 'min_steps_estimation' in state:
            msg['min_steps_estimation'] = state['min_steps_estimation']
            
        # --- 原始 state (用于调试或特殊情况) ---
        msg['raw_state'] = state

        return msg
    
    def reset(self):
        """重置智能体在一局游戏开始前的状态。"""
        super().reset()
        # self.history_buffer.clear() # 移除
        self.reward_shaper.reset()
        self.current_state = None
        self.current_action = None
        if hasattr(self, 'step_history'): self.step_history = []

    def update_history(self, state_tensor):
        """将当前状态张量添加到历史状态缓冲区中。"""
        # state_tensor 形状通常是 [1, state_dim]，我们需要移除批次维度
        self.history_buffer.append(state_tensor.squeeze(0).clone().detach())

    def get_history_tensor(self):
        """从历史缓冲区获取历史状态序列张量，用于LSTM输入。"""
        if not self.history_buffer:
            # 如果历史为空，返回一个形状正确的全零张量
            # 形状: [batch_size, seq_len, feature_dim] -> [1, 1, state_dim]
            # seq_len=1 是为了避免在LSTM中因序列长度为0而出错
            return torch.zeros(1, 1, self.state_dim, device=self.device)
        
        # 将deque中的张量堆叠起来，并增加批次维度
        # torch.stack 会创建一个新的维度
        # list(self.history_buffer) -> [tensor1, tensor2, ...] (每个tensor形状 [state_dim])
        # torch.stack(...) -> [seq_len, state_dim]
        # .unsqueeze(0) -> [1, seq_len, state_dim] (增加 batch_size 维度)
        return torch.stack(list(self.history_buffer), dim=0).unsqueeze(0)

    def select_action(self, state_dict, evaluation=False):
        """
        根据当前状态字典选择动作。
        此方法假定输入的 state_dict (来自parse) 是一个需要进行决策的有效状态。
        """
        self.state_dict = state_dict
        
        # --- 直接开始决策流程 ---
        action_list = state_dict["actionList"]
        
        current_state_tensor = self.state_encoder.encode_state(state_dict)
        history_states_tensor = self.get_history_tensor()
        action_features, action_indices = self.state_encoder.create_action_features(state_dict, action_list)
        _, action_mapping = self.abstract_action_space(action_list)
        
        with torch.no_grad():
            action_scores, new_policy_hidden = self.policy_network(
                current_state_tensor, history_states_tensor, 
                self.policy_hidden_state if self.preserve_rnn_state else None)

        legal_action_mask = torch.zeros(1, self.abstract_action_dim, device=self.device)
        action_to_abstract_mapping = {}
        abstract_id_to_action_indices = {}
        
        for i, _ in enumerate(action_features):
            if i < len(action_indices):
                orig_idx = action_indices[i]
                abstract_id = next((abs_id for abs_id, orig_ids in action_mapping.items() if orig_idx in orig_ids), None)
                if abstract_id is None: raise ValueError(f"Cannot find abstract_id for orig_idx {orig_idx}")
                
                action_to_abstract_mapping[orig_idx] = abstract_id
                legal_action_mask[0, abstract_id] = 1.0
                abstract_id_to_action_indices.setdefault(abstract_id, []).append(orig_idx)
        
        trump_rank = state_dict.get('curRank')
        masked_logits = torch.full((1, self.abstract_action_dim), -1e9, device=self.device)
        
        for abstract_id, orig_indices_list in abstract_id_to_action_indices.items():
            normal_actions_scores, special_actions_scores = [], []
            for idx in orig_indices_list:
                action = action_list[idx]
                is_straight_flush = action[0] == 'StraightFlush'
                has_trump = any(card[0] == 'H' and card[1:] == trump_rank for card in action[2] if isinstance(card, str) and len(card) >=2) if len(action) > 2 and isinstance(action[2], list) and trump_rank else False
                
                action_idx_pos = action_indices.index(idx) if idx in action_indices else -1
                if action_idx_pos == -1 or action_idx_pos >= len(action_scores[0]): continue
                    
                score = action_scores[0][action_idx_pos].item()
                (special_actions_scores if is_straight_flush or has_trump else normal_actions_scores).append(score)
            
            if normal_actions_scores: masked_logits[0, abstract_id] = np.mean(normal_actions_scores)
            elif special_actions_scores: masked_logits[0, abstract_id] = np.mean(special_actions_scores)
        
        probs = F.softmax(masked_logits, dim=1)
        abstract_action_idx = torch.argmax(probs, dim=1).item() if evaluation else torch.distributions.Categorical(probs).sample().item()
        
        perfect_info_tensor = self._extract_perfect_info_features(state_dict) if self.train_mode else None
        value_pred, new_value_hidden = self.value_network(current_state_tensor, history_states_tensor, perfect_info_tensor, self.value_hidden_state)
        
        selected_indices_for_abstract = abstract_id_to_action_indices.get(abstract_action_idx)
        if not selected_indices_for_abstract:
             if not action_list:
                  return ['PASS','PASS','PASS'], {} 
             original_action_idx = random.choice(action_indices) if action_indices else 0
             chosen_action = action_list[original_action_idx]
        else:
            chosen_action, original_action_idx = self._decode_action_internal(selected_indices_for_abstract, action_list, trump_rank)

        log_prob_val = torch.log(probs[0, abstract_action_idx] + 1e-10).item()
        
        self.policy_hidden_state, self.value_hidden_state = new_policy_hidden, new_value_hidden
        self.update_history(current_state_tensor)
        
        return chosen_action, {
            'state': current_state_tensor, 
            'history_states': history_states_tensor, 
            'perfect_info': perfect_info_tensor,
            'action_idx': original_action_idx, 
            'abstract_action_idx': abstract_action_idx, 
            'log_prob': log_prob_val,
            'value': value_pred.item(), 
            'action_mask': legal_action_mask, 
            'action_features': action_features,
            'next_policy_hidden': new_policy_hidden, 
            'next_value_hidden': new_value_hidden, 
            'raw_state': state_dict.get('raw_state')
        }
        
    def _decode_action_internal(self, original_indices, action_list, trump_rank=None):
        """
        从属于同一个抽象动作的多个原始动作中，根据启发式规则选择一个。
        优先选择不包含级牌或同花顺的普通牌型中的一个（随机）。
        """
        if not original_indices: return ['PASS','PASS','PASS'], -1 # Should not happen if called correctly
        
        # Simplified: prefer non-trump, non-SF. If multiple, pick first.
        # This can be made more sophisticated if needed.
        best_action = None
        best_idx = -1
        
        other_actions = []
        trump_actions = []
        sf_actions = []
        
        for idx in original_indices:
            action = action_list[idx]
            is_sf = action[0] == 'StraightFlush'
            has_trump = any(c[0]=='H' and c[1:]==trump_rank for c in action[2] if isinstance(c,str) and len(c)>=2) if len(action) > 2 and isinstance(action[2],list) and trump_rank else False

            if is_sf: sf_actions.append((idx, action))
            elif has_trump: trump_actions.append((idx, action))
            else: other_actions.append((idx, action))

        if other_actions: best_idx, best_action = random.choice(other_actions)
        elif trump_actions: best_idx, best_action = random.choice(trump_actions)
        elif sf_actions: best_idx, best_action = random.choice(sf_actions)
        else: # Should be covered by above, but as a fallback
            best_idx = original_indices[0]
            best_action = action_list[best_idx]
            
        return best_action, best_idx

    def _extract_perfect_info_features(self, state_dict):
        """
        提取完美信息特征，主要包括其他玩家的当前手牌和预估的最小出完步数。
        此方法依赖于 parse 方法打包好的 state_dict。
        """
        perfect_info = torch.zeros(1, self.perfect_info_dim, device=self.device)
        
        # 从 parse 后的 state_dict 中获取完美信息
        all_hands = state_dict.get('all_players_hands', {})
        min_steps = state_dict.get('min_steps_estimation', {})
        
        hand_idx_offset = 0
        for p_id in range(4):
            if p_id == self.player_id: continue
            player_cards = all_hands.get(p_id, [])
            for card_str in player_cards:
                if isinstance(card_str, str) and len(card_str) >= 2:
                    try:
                        # 此处需要一个从 card_str 到一个固定索引 (0-53) 的映射
                        # 这是一个示例，具体实现取决于您的状态编码细节
                        # pass # Placeholder: Implement robust card to index mapping for perfect info
                        pass
                    except ValueError:
                        pass
            hand_idx_offset += 54

        step_idx_offset = 3 * 54
        other_player_idx = 0
        for p_id in range(4):
            if p_id == self.player_id: continue
            steps = min_steps.get(p_id, 999)
            perfect_info[0, step_idx_offset + other_player_idx] = float(steps)
            other_player_idx += 1
        
        return perfect_info

    def abstract_action_space(self, action_list):
        """
        将原始的合法动作列表抽象化，减少动作空间的复杂度。
        相同牌型和点数的动作（如不同花色的单张A）归为同一个抽象动作。
        """
        action_groups, action_mapping, abstract_actions = {}, {}, []
        abstract_idx = 0
        for idx, action in enumerate(action_list):
            if not isinstance(action, list) or not action: continue
            key = 'PASS' if action[0] == 'PASS' else f"{action[0]}_{action[1]}" if len(action) >=2 else action[0]
            action_groups.setdefault(key, []).append((idx, action))
        
        for _, actions_in_group in action_groups.items():
            if actions_in_group:
                first_orig_idx, first_action = actions_in_group[0]
                abstract_actions.append(first_action)
                action_mapping[abstract_idx] = [orig_idx for orig_idx, _ in actions_in_group]
                abstract_idx += 1
        return abstract_actions, action_mapping

    def on_episode_end(self, final_state, winner_team):
        """
        在一小局（episode）结束时调用。
        计算最终奖励，并直接将 step_history 中的数据存入 PPOMemory。
        """
        if self.verbose:
            self.logger.info(f"小局结束处理. 赢家队伍: {winner_team}, 我的队伍: {self.player_id % 2}")
        
        if not hasattr(self, 'step_history') or not self.step_history:
            self.reward_shaper.reset()
            return

        # 1. 计算并分配最终奖励
        final_reward_val = self.reward_shaper.calculate_final_reward(final_state, winner_team, self.player_id)
        
        last_non_pass_step = next((step for step in reversed(self.step_history) if step['player_id'] == self.player_id and step['action'][0] != 'PASS'), None)
        target_step_for_final_reward = last_non_pass_step
        if not target_step_for_final_reward:
             # 如果玩家全程PASS，将奖励给最后一个属于他的步骤
             my_last_step = next((step for step in reversed(self.step_history) if step['player_id'] == self.player_id), None)
             target_step_for_final_reward = my_last_step

        if target_step_for_final_reward:
            target_step_for_final_reward['reward'] += final_reward_val
            if self.verbose: self.logger.info(f"最终奖励 {final_reward_val:.4f} 分配给步骤: {target_step_for_final_reward['action']}")

        # 2. 将本局所有属于该智能体的步骤存入 PPOMemory
        # 我们直接遍历 step_history，不再需要 process_experiences
        num_my_steps = sum(1 for step in self.step_history if step['player_id'] == self.player_id)
        my_step_count = 0
        
        for step in self.step_history:
            if step['player_id'] == self.player_id:
                my_step_count += 1
                # 判断是否为当前 agent 的最后一步
                is_done = (my_step_count == num_my_steps)
                
                # 显式地将 step 字典中的值映射到 store 方法的参数
                self.memory.store(
                    state=step['state_tensor'],
                    history_state=step['history_tensor'],
                    action=step['action_idx'],
                    log_prob=step['log_prob'],
                    val=step['value'],
                    reward=step.get('reward', 0.0),
                    done=is_done,
                    action_mask=step.get('action_mask'),
                    perfect_info=step.get('perfect_info'),
                    action_features=step.get('action_features'),
                    initial_policy_hidden=step.get('next_policy_hidden'),
                    initial_value_hidden=step.get('next_value_hidden'),
                    raw_state=step.get('raw_state')
                )
        
        if self.verbose: self.logger.info(f"为玩家 {self.player_id} 存储了 {my_step_count} 条经验")

        # 3. 重置状态
        self.reward_shaper.reset()
        self.step_history = []
        if self.verbose: self.logger.info(f"小局结束处理完成, RewardShaper TC after reset: {self.reward_shaper.turn_count}")

    def process_experiences(self, step_history, winner_team):
        # 这个实现是完整的，但当前流程已不再调用它
        experiences = []
        for step in step_history:
            if step['player_id'] == self.player_id:
                # 创建一个新的字典，明确键名
                exp_data = {
                    'state_tensor': step['state_tensor'],
                    'history_tensor': step['history_tensor'],
                    'action_idx': step['action_idx'],
                    'abstract_action_idx': step['abstract_action_idx'],
                    'log_prob': step['log_prob'],
                    'value': step['value'],
                    'reward': step.get('reward', 0.0),
                    'action_mask': step.get('action_mask'),
                    'action_features': step.get('action_features'),
                    'next_policy_hidden': step.get('next_policy_hidden'),
                    'next_value_hidden': step.get('next_value_hidden'),
                    'step_index': step['step_index'],
                    'has_finished': step.get('has_finished', False),
                    'perfect_info': step.get('perfect_info'),
                    'raw_state': step.get('raw_state')
                }
                experiences.append(exp_data)
        return experiences
    
    def tribute_act(self, actionLists, rank):
        """选择并执行进贡动作"""
        if not isinstance(actionLists, list) or len(actionLists) == 0:
            self.logger.error("进贡动作列表为空或格式不正确")
            return ['PASS', 'PASS', 'PASS']  # 返回默认值而不是None
        
        # 选择点数最大的牌
        best_index = 0
        best_rank_value = -1
        
        for i, action in enumerate(actionLists):
            if isinstance(action, list) and len(action) >= 3 and isinstance(action[2], list) and len(action[2]) > 0:
                card = action[2][0]
                
                if len(card) >= 2:
                    rank_str = card[1:]
                    rank_value = self._get_rank_value(rank_str)
                    
                    if rank_value > best_rank_value:
                        best_rank_value = rank_value
                        best_index = i
        
        # 选择并执行最佳动作
        act = actionLists[best_index]
        try:
            self.execute_tribute(act)
        except Exception as e:
            self.logger.error(f"执行进贡动作失败: {e}")
        
        # 返回前进行最终检查
        if act is None or not isinstance(act, list):
            self.logger.error(f"进贡动作无效: {act}, 使用备选动作")
            return actionLists[0] if actionLists else ['tribute', 'PASS', ['PASS']]
                
        return act

    def back_act(self, actionLists, rank, tribute_result):
        """
        选择并执行还贡动作
        
        参数:
        actionLists: 可用的还贡动作列表
        rank: 当前级牌
        tribute_result: 进贡结果信息
        
        返回:
        act: 选择的还贡动作(完整动作，不修改格式)
        """
        if not isinstance(actionLists, list) or len(actionLists) == 0:
            return None
        
        # 获取当前完整手牌
        current_hand = self.current_hand_str
        
        # 识别级牌和通配牌
        trump_rank = rank
        trump_card = None
        
        for card in current_hand:
            if len(card) >= 2 and card[0] == 'H' and card[1:] == trump_rank:
                trump_card = card
                break
        
        # 分析手牌和计算每个动作的分数
        current_analysis = self._analyze_hand_cards(current_hand, trump_rank, trump_card)
        current_value = self._calculate_total_hand_value(current_analysis)
        
        best_score = float('-inf')
        best_index = 0
        
        for i, action in enumerate(actionLists):
            try:
                if isinstance(action, list) and len(action) >= 3 and isinstance(action[2], list) and len(action[2]) > 0:
                    card = action[2][0]
                    score = self._evaluate_card_impact(card, current_hand, current_analysis, current_value, trump_rank)
                    
                    if score > best_score:
                        best_score = score
                        best_index = i
            except Exception as e:
                print(f"评估动作 {action} 时出错: {e}")
        
        # 选择最佳动作并执行
        act = actionLists[best_index]
        self.execute_back(act)
        
        # 直接返回完整的动作对象，不修改格式
        return act

    def _analyze_hand_cards(self, cards, trump_rank, trump_card):
        """
        分析手牌构成，识别各种牌型
        """
        result = {
            'rank_counts': {},         # 点数统计
            'suit_counts': {},         # 花色统计
            'straights': [],           # 顺子
            'potential_straights': [], # 潜在顺子(通配牌辅助)
            'sequence_pairs': [],      # 三连对 
            'sequence_trips': [],      # 钢板
            'three_with_twos': [],     # 三带二
            'bombs': [],               # 炸弹
            'straight_flushes': [],    # 同花顺
            'trump_rank': trump_rank,  # 级牌点数
            'trump_card': trump_card,  # 通配牌
            'has_trump': trump_card is not None  # 是否有通配牌
        }
        
        # 统计点数和花色
        for card in cards:
            if len(card) < 2:
                continue
                
            suit, rank_str = card[0], card[1:]
            
            # 点数统计
            result['rank_counts'][rank_str] = result['rank_counts'].get(rank_str, 0) + 1
            
            # 花色统计
            result['suit_counts'][suit] = result['suit_counts'].get(suit, 0) + 1
        
        # 识别炸弹
        for r, count in result['rank_counts'].items():
            if count >= 4:
                result['bombs'].append(r)
        
        # 识别顺子和潜在顺子
        self._identify_straights(cards, result, trump_rank, trump_card)
        
        # 识别三连对、钢板等复合牌型
        self._identify_compound_patterns(result)
        
        return result

    def _identify_straights(self, cards, result, trump_rank, trump_card):
        """
        识别所有可能的顺子，包括通配牌辅助形成的顺子
        修复inhomogeneous shape错误
        """
        # 转换为点数值进行顺子判断
        rank_values = {}
        for card in cards:
            if len(card) < 2:
                continue
            rank_str = card[1:]
            value = self._get_rank_value(rank_str)
            if value > 0:
                if value not in rank_values:
                    rank_values[value] = []
                rank_values[value].append(card)
        
        # 常规顺子识别(5张连续牌)
        for start in range(2, 11):  # 从2开始到10结束(10-J-Q-K-A)
            straight_cards = []
            missing_positions = []
            
            for i in range(5):  # 5张连续牌
                check_value = start + i
                if check_value in rank_values:
                    # 只存储一张该点数的牌，避免不规则数组
                    straight_cards.append(rank_values[check_value][0])
                else:
                    missing_positions.append(i)
            
            # 完整顺子
            if not missing_positions:
                result['straights'].append((start, straight_cards))
            # 缺一张可用通配牌替代的潜在顺子
            elif len(missing_positions) == 1 and result['has_trump']:
                result['potential_straights'].append((start, missing_positions[0], straight_cards))
        
        # 特殊处理A-2-3-4-5顺子
        if all(v in rank_values for v in [14, 2, 3, 4, 5]):  # A,2,3,4,5
            straight_cards = []
            for v in [14, 2, 3, 4, 5]:
                # 只存储一张该点数的牌
                straight_cards.append(rank_values[v][0])
            result['straights'].append((14, straight_cards))  # 特殊标记A开头顺子
        
        # 缺一张的A-2-3-4-5顺子
        missing_values = [v for v in [14, 2, 3, 4, 5] if v not in rank_values]
        if len(missing_values) == 1 and result['has_trump']:
            straight_cards = []
            missing_position = [14, 2, 3, 4, 5].index(missing_values[0])
            for v in [14, 2, 3, 4, 5]:
                if v != missing_values[0]:
                    # 只存储一张该点数的牌
                    straight_cards.append(rank_values[v][0])
            result['potential_straights'].append((14, missing_position, straight_cards))

    def _identify_compound_patterns(self, result):
        """
        识别复合牌型：三连对、钢板、三带二等
        """
        rank_counts = result['rank_counts']
        
        # 找出所有对子和三张
        pairs = [r for r, count in rank_counts.items() if count >= 2]
        trips = [r for r, count in rank_counts.items() if count >= 3]
        
        # 转换为数值并排序
        pair_values = sorted([self._get_rank_value(r) for r in pairs])
        trip_values = sorted([self._get_rank_value(r) for r in trips])
        
        # 识别三连对(三对连续的对子)
        for i in range(len(pair_values) - 2):
            if pair_values[i+1] == pair_values[i] + 1 and pair_values[i+2] == pair_values[i] + 2:
                result['sequence_pairs'].append((pair_values[i], pair_values[i+1], pair_values[i+2]))
        
        # 识别钢板(两组连续的三张)
        for i in range(len(trip_values) - 1):
            if trip_values[i+1] == trip_values[i] + 1:
                result['sequence_trips'].append((trip_values[i], trip_values[i+1]))
        
        # 识别三带二
        for trip in trips:
            for pair in pairs:
                if trip != pair:  # 确保不是同一个点数
                    result['three_with_twos'].append((trip, pair))

    def _evaluate_card_impact(self, card, hand_cards, analysis, current_value, trump_rank):
        """
        评估一张牌作为还贡牌的影响，分数越高越适合还贡
        使用模拟移除+计算价值变化的评估逻辑，保留优化后的牌型价值体系
        """
        # 如果是通配牌，绝对不还
        if card == analysis['trump_card']:
            return -1000
        
        # 基础分数
        score = 50
        
        # 提取牌的点数和花色
        if len(card) < 2:
            return -1000
        
        suit, rank_str = card[0], card[1:]
        rank_value = self._get_rank_value(rank_str)
        
        # 1. 模拟移除该牌，分析牌型变化
        remaining_cards = [c for c in hand_cards if c != card]
        analysis_after_removal = self._analyze_hand_cards(remaining_cards, trump_rank, analysis['trump_card'])
        
        # 2. 计算手牌总价值变化
        value_before = self._calculate_total_hand_value(analysis)
        value_after = self._calculate_total_hand_value(analysis_after_removal)
        
        # 3. 价值变化: 正数表示价值降低(不好)，负数表示价值提高(好)
        value_impact = value_before - value_after
        
        # 4. 根据价值变化调整分数
        score -= value_impact
        
        # 5. 考虑点数 - 优先选择点数小的牌
        if rank_value <= 10:
            # 点数越小越好
            score += (11 - rank_value) * 2  # 2得10分，10得2分
        
        # 6. 花色考虑 - 黑桃和梅花优先
        if suit in ['S', 'C']:
            score += 2
        
        return score

    def _calculate_total_hand_value(self, analysis):
        """
        计算手牌的总体价值
        使用优化后的牌型价值体系
        """
        total_value = 0
        
        # 1. 炸弹价值 (最高价值)
        bomb_value = sum(200 for _ in analysis['bombs'])
        
        # 2. 顺子价值 - 高价值
        straight_value = 0
        for start, cards in analysis['straights']:
            if start == 14:  # A开头顺子
                straight_value += 100
            else:
                straight_value += 80 + start * 2  # 顺子价值与起始点数强相关
        
        # 3. 潜在顺子价值 - 基于顺子打折但仍然高价值
        potential_straight_value = 0
        for start, missing_pos, cards in analysis['potential_straights']:
            if start == 14:
                potential_straight_value += 75
            else:
                potential_straight_value += 60 + start * 1.5
        
        # 4. 三连对价值 - 略低于顺子
        sequence_pair_value = 0
        for p1, p2, p3 in analysis['sequence_pairs']:
            sequence_pair_value += 120 + (p1 - 2) * 3
        
        # 5. 钢板价值 - 与三连对相当
        sequence_trip_value = 0
        for t1, t2 in analysis['sequence_trips']:
            sequence_trip_value += 120 + (t1 - 2) * 3
        
        # 6. 三带二价值 - 中等价值
        three_with_two_value = 0
        for trip, pair in analysis['three_with_twos']:
            three_with_two_value += 60 + self._get_rank_value(trip) * 0.5
        
        # 7. 单独的三张和对子 - 低价值
        trips_and_pairs_value = 0
        for rank, count in analysis['rank_counts'].items():
            rank_val = self._get_rank_value(rank)
            if count == 3:
                # 检查是否是单独的三张(不在钢板或三带二中)
                is_in_other_pattern = False
                for t1, t2 in analysis['sequence_trips']:
                    if rank_val in [t1, t2]:
                        is_in_other_pattern = True
                        break
                
                for trip, pair in analysis['three_with_twos']:
                    if rank == trip:
                        is_in_other_pattern = True
                        break
                
                if not is_in_other_pattern:
                    trips_and_pairs_value += 25 + (rank_val - 2) * 0.5
            elif count == 2:
                # 检查是否是单独的对子(不在连对或三带二中)
                is_in_other_pattern = False
                for p1, p2, p3 in analysis['sequence_pairs']:
                    if rank_val in [p1, p2, p3]:
                        is_in_other_pattern = True
                        break
                
                for trip, pair in analysis['three_with_twos']:
                    if rank == pair:
                        is_in_other_pattern = True
                        break
                
                if not is_in_other_pattern:
                    trips_and_pairs_value += 15 + (rank_val - 2) * 0.3
        
        # 8. 通配牌额外价值
        trump_value = 50 if analysis['has_trump'] else 0  # 大幅提高通配牌价值
        
        # 总价值计算
        total_value = bomb_value + straight_value + potential_straight_value + \
                        sequence_pair_value + sequence_trip_value + \
                        three_with_two_value + trips_and_pairs_value + trump_value
        
        return total_value

    def _get_rank_value(self, rank_str):
        return {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, 'B': 15, 'R': 16}.get(rank_str, 0)
    
    def update_policy(self):

        if len(self.memory) < self.memory.batch_size:
            return None
        
        # 1. 计算优势和回报
        self.memory.compute_advantages_and_returns()
        
        policy_losses, value_losses, entropy_values = [], [], []
        
        # 2. PPO的多次迭代更新
        for _ in range(self.n_epochs):
            for batch in self.memory.get_minibatch_generator(as_tensor_dict=True):
                states_batch, acts_batch, old_log_p_batch, returns_batch, advs_batch, p_info_batch, \
                raw_states_batch, action_features_batch = \
                    batch['states'], batch['actions'], batch['log_probs'], \
                    batch['returns'], batch['advantages'], batch.get('perfect_info'), \
                    batch['raw_states'], batch['action_features']

                # 标准化优势 (在整个minibatch上进行)
                if advs_batch.shape[0] > 1:
                    advs_batch = (advs_batch - advs_batch.mean()) / (advs_batch.std() + 1e-8)
                
                # --- 价值网络部分 (可以保持批处理，效率更高) ---
                vals, _ = self.value_network(states_batch, perfect_info=p_info_batch)
                val_loss = F.mse_loss(vals.squeeze(), returns_batch)

                # --- 核心修正：对minibatch中的每个样本单独处理策略部分 ---
                new_log_probs_list = []
                entropies_list = []

                for i in range(states_batch.size(0)):
                    # 提取单个样本的数据
                    state_tensor_single = states_batch[i:i+1] # 保持 [1, dim] 的形状
                    raw_state = raw_states_batch[i]
                    action_features = action_features_batch[i]
                    
                    # 重新计算该状态下所有具体动作的得分
                    if not action_features:
                        # 如果没有动作，跳过这个样本的策略部分计算
                        # 理论上不应发生，因为有动作才会有记录
                        continue
                    
                    specific_action_scores, _ = self.policy_network(state_tensor_single, action_features)
                    
                    # --- 复现聚合逻辑 ---
                    action_list = raw_state.get('actions')
                    _, action_indices = self.state_encoder.create_action_features(raw_state, action_list)
                    _, action_mapping = self.abstract_action_space(action_list)
                    
                    abstract_id_to_action_indices = {}
                    for j, _ in enumerate(action_features):
                        orig_idx = action_indices[j]
                        abstract_id = next((abs_id for abs_id, orig_ids in action_mapping.items() if orig_idx in orig_ids), None)
                        if abstract_id is not None:
                            abstract_id_to_action_indices.setdefault(abstract_id, []).append(j)

                    # 计算抽象动作的logits (平均值)
                    abstract_logits = torch.full((1, self.abstract_action_dim), -1e9, device=self.device)
                    for abstract_id, specific_indices in abstract_id_to_action_indices.items():
                        if specific_indices:
                            scores_for_this_abstract = specific_action_scores[0, specific_indices]
                            abstract_logits[0, abstract_id] = scores_for_this_abstract.mean()

                    # 创建新的概率分布
                    dist = torch.distributions.Categorical(logits=abstract_logits)
                    
                    # acts_batch[i] 是之前采样的抽象动作ID
                    new_log_prob = dist.log_prob(acts_batch[i])
                    new_log_probs_list.append(new_log_prob)
                    entropies_list.append(dist.entropy())

                if not new_log_probs_list: # 如果整个batch都没有可处理的策略样本
                    continue

                # --- 组合batch的结果 ---
                new_log_p = torch.stack(new_log_probs_list)
                entropy = torch.stack(entropies_list).mean()
                
                # PPO 比例和裁剪
                # 注意：如果batch中某些样本被跳过，需要对齐advs_batch
                ratio = torch.exp(new_log_p - old_log_p_batch) # 假设长度一致
                surr1 = ratio * advs_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advs_batch
                
                # --- 损失计算和反向传播 ---
                pol_loss = -torch.min(surr1, surr2).mean()
                
                # 价值网络梯度
                self.value_optimizer.zero_grad()
                val_loss.backward(retain_graph=True) # retain_graph以防共享编码器
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                # 策略网络梯度
                pol_total_loss = pol_loss - self.entropy_coef * entropy
                self.policy_optimizer.zero_grad()
                pol_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                policy_losses.append(pol_loss.item())
                value_losses.append(val_loss.item())
                entropy_values.append(entropy.item())

        if self.use_lr_scheduler:
            self.policy_scheduler.step(np.mean(policy_losses))
            self.value_scheduler.step(np.mean(value_losses))
            
        self.memory.clear()
        return {'policy_loss': np.mean(policy_losses), 'value_loss': np.mean(value_losses), 'entropy': np.mean(entropy_values)}

    def save_model(self, path):
        torch.save({
                'policy_network_state_dict': self.policy_network.state_dict(),
                'value_network_state_dict': self.value_network.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            # 'stats': self.stats, # Stats can be large and non-essential for inference
        }, path)
        if self.verbose: self.logger.info(f"模型已保存到 {path}")

    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            # if 'stats' in checkpoint: self.stats = checkpoint['stats']
            if self.verbose: self.logger.info(f"模型已从 {path} 加载")
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}", exc_info=True)
            raise

    def calculate_first_select_reward(self, first_player_in_turn, last_non_pass_player_in_turn, turn_actions):
        """
        计算首出奖励。
        规则：如果当前轮次第一个出牌的玩家，其队伍最终掌握了牌权，则获得正奖励；否则受惩罚。
        例外：如果牌权是通过同花顺或炸弹获得的，则不计算此奖励。
        此奖励仅给予当前轮次的第一个出牌玩家。
        """
        if first_player_in_turn == -1 or last_non_pass_player_in_turn == -1: return 0.0

        last_action_detail = None
        for act_info in reversed(turn_actions):
            if act_info['player_id'] == last_non_pass_player_in_turn and not act_info['is_pass']:
                last_action_detail = act_info['action']; break
        
        if last_action_detail and last_action_detail[0] in ['StraightFlush', 'Bomb']:
            if self.verbose: self.logger.info(f"  (P{first_player_in_turn}首出奖励判断: 牌权由P{last_non_pass_player_in_turn}通过{last_action_detail[0]}获得,首出奖励=0)")
            return 0.0

        first_player_team = first_player_in_turn % 2
        turn_winner_team = last_non_pass_player_in_turn % 2
        reward = 0.3 if first_player_team == turn_winner_team else -0.3
        
        if self.verbose:
            outcome = "保持/夺得牌权" if first_player_team == turn_winner_team else "失去牌权给对手"
            self.logger.info(f"  (P{first_player_in_turn}首出奖励判断: 首出P{first_player_in_turn}(T{first_player_team}) {outcome} (牌权P{last_non_pass_player_in_turn},T{turn_winner_team}), 首出奖励={reward:.1f})")
        return reward

    def _calculate_player_perfect_reward_component(self, player_id, oracle_val_before_turn, oracle_val_after_turn, is_current_agent_verbose_flag):
        """
        计算单个玩家的完美奖励（Oracle奖励）部分。
        基于本轮行动前后，全局Oracle值的变化。
        """
        final_perfect_reward = 0.0
        if oracle_val_before_turn is not None:
            oracle_change = oracle_val_after_turn - oracle_val_before_turn
            player_team = player_id % 2
            final_perfect_reward = -oracle_change if player_team == 0 else oracle_change
            if self.verbose and is_current_agent_verbose_flag:
                self.logger.info(f"    (P{player_id}(T{player_team})完美组件: OracleAfter={oracle_val_after_turn}, OracleBefore={oracle_val_before_turn}, Change={oracle_change} => Reward: {final_perfect_reward:.2f})")
        elif self.verbose and is_current_agent_verbose_flag:
            self.logger.info(f"    (P{player_id}完美组件: 首轮/无OracleBefore, 完美奖励=0.0)")
        return final_perfect_reward

    def calculate_turn_rewards(self):
        """
        计算一轮（trick）结束后，所有玩家应得的各项奖励。
        包括：首出奖励、出牌差奖励、完美奖励。
        """
        rewards = {}
        rs_states = self.reward_shaper.current_turn_states # shortcut
        last_non_pass_player = rs_states.get('last_non_pass_player')
        turn_actions = rs_states.get('turn_actions', [])
        all_hands_after_turn = rs_states.get('all_players_hands', {})

        if last_non_pass_player is None or not turn_actions:
            if self.verbose: self.logger.info("calculate_turn_rewards: 无效轮次信息，跳过。")
            return rewards

        first_player_of_this_turn = turn_actions[0]['player_id']
        current_turn_oracle_value = 0
        if all_hands_after_turn: # Calculate oracle based on hands *after* this turn
            h0,h1,h2,h3 = all_hands_after_turn.get(0,[]), all_hands_after_turn.get(1,[]), all_hands_after_turn.get(2,[]), all_hands_after_turn.get(3,[])
            current_turn_oracle_value = min(len(h0), len(h2)) - min(len(h1), len(h3))
        
        oracle_before_this_turn = rs_states.get('previous_oracle') # Oracle from *before* this turn
        
        if self.verbose:
            self.logger.info(f"=== Agent {self.player_id} 开始计算轮次奖励 (TC: {self.reward_shaper.turn_count}) ===")
            self.logger.info(f"本轮首出: P{first_player_of_this_turn}, 本轮牌权: P{last_non_pass_player}")
            self.logger.info(f"Oracle值: Previous={oracle_before_this_turn}, Current(AfterTurn)={current_turn_oracle_value}")

        team_cards_count = {0: 0, 1: 0}
        for act_info in turn_actions:
            if not act_info['is_pass'] and len(act_info['action']) > 2 and isinstance(act_info['action'][2], list):
                team_cards_count[act_info['player_id'] % 2] += len(act_info['action'][2])
        
        card_diff_scale = 0.1
        if self.verbose: self.logger.info(f"队伍出牌数: T0={team_cards_count[0]}, T1={team_cards_count[1]}")

        for p_id_iter in range(4):
            is_my_turn_to_log_details = (p_id_iter == self.player_id) # Log details only for the current agent's ID

            first_play_rew = 0.0
            if p_id_iter == first_player_of_this_turn:
                first_play_rew = self.calculate_first_select_reward(first_player_of_this_turn, last_non_pass_player, turn_actions)
            
            card_diff_rew = card_diff_scale * (team_cards_count[p_id_iter % 2] - team_cards_count[(p_id_iter % 2 + 1) % 2])
            
            perfect_rew = self._calculate_player_perfect_reward_component(p_id_iter, oracle_before_this_turn, current_turn_oracle_value, is_my_turn_to_log_details)
            
            total_rew = first_play_rew + card_diff_rew + perfect_rew
            
            if self.verbose and is_my_turn_to_log_details:
                self.logger.info(f"  -- P{p_id_iter} (Agent {self.player_id}) 奖励明细 --")
                self.logger.info(f"    1. 首出奖励: {first_play_rew:.2f}")
                self.logger.info(f"    2. 出牌差奖励(T{p_id_iter % 2}): {card_diff_rew:.2f}")
                self.logger.info(f"    3. 完美奖励: {perfect_rew:.2f} (内部已有打印)")
                self.logger.info(f"    >>> P{p_id_iter} 总轮次奖励: {total_rew:.2f}")

            rewards[p_id_iter] = {
                'reward': total_rew,
                'details': {'first_play_reward': first_play_rew, 'card_difference_reward': card_diff_rew, 'perfect_reward': perfect_rew},
                'allocation': 'first_action' if p_id_iter == first_player_of_this_turn else 'last_non_pass_action'
            }
            # Accumulating total rewards for the episode in the agent's own reward_shaper instance
            self.reward_shaper.total_rewards[p_id_iter] = self.reward_shaper.total_rewards.get(p_id_iter, 0) + total_rew

        # Update 'previous_oracle' in this agent's shaper for the next turn's calculation
        self.reward_shaper.current_turn_states['previous_oracle'] = current_turn_oracle_value
        if self.verbose:
            self.logger.info(f"更新 Agent {self.player_id} 的下一轮 'previous_oracle' 为: {current_turn_oracle_value}")
            self.logger.info(f"=== Agent {self.player_id} 轮次奖励计算结束 ===")
        return rewards

class PPOMemory:
    """
    PPO轨迹缓冲区。
    """
    def __init__(self, batch_size=1024, max_size=4000, gamma=0.99, gae_lambda=0.95, 
                 device='cpu'):
        self.batch_size = batch_size
        self.max_size = max_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # 存储空间
        self.states = None
        self.history_states_data = {}
        self.actions = np.zeros(max_size, dtype=np.int64)
        self.log_probs = np.zeros(max_size, dtype=np.float32)
        self.vals = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.int8)
        self.action_masks = None
        self.perfect_info = None
        self.action_features_data = {}
        
        # FIX: 为 raw_state 添加存储空间
        # raw_state 是字典，不能直接存入numpy数组，所以我们用一个列表来存储
        self.raw_states = [None] * max_size
        
        # 计算的回报和优势
        self.returns = np.zeros(max_size, dtype=np.float32)
        self.advantages = np.zeros(max_size, dtype=np.float32)
        
        # RNN 隐藏状态
        self.initial_policy_hidden_data = {}
        self.initial_value_hidden_data = {}
        
        # 位置和大小指针
        self.position = 0
        self.size = 0
        self.state_dim = None

    def __len__(self):
        """返回当前存储的经验数量"""
        return self.size
    
    def _initialize_storage(self, state, action_mask, perfect_info):
        if self.states is None and state is not None:
            self.state_dim = state.shape[1:]
            self.states = np.zeros((self.max_size, *self.state_dim), dtype=np.float32)
        if self.action_masks is None and action_mask is not None:
            self.action_masks = np.zeros((self.max_size, *action_mask.shape[1:]), dtype=np.float32)
        if self.perfect_info is None and perfect_info is not None:
            self.perfect_info = np.zeros((self.max_size, *perfect_info.shape[1:]), dtype=np.float32)
    
    def store(self, state, history_state, action, log_prob, val, reward, done, 
              action_mask=None, perfect_info=None, action_features=None,
              initial_policy_hidden=None, initial_value_hidden=None, raw_state=None):
        """
        存储一条经验到轨迹中。
        此版本接受 raw_state 参数。
        """
        if self.size == 0:
            self._initialize_storage(state, action_mask, perfect_info)

        idx = self.position
        
        if self.states is not None and state is not None:
            self.states[idx] = state.detach().cpu().numpy()
        
        if history_state is not None:
            self.history_states_data[idx] = history_state.detach().cpu().numpy()
            
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.vals[idx] = val
        self.rewards[idx] = reward
        self.dones[idx] = done
        
        if self.action_masks is not None and action_mask is not None:
            self.action_masks[idx] = action_mask.detach().cpu().numpy()
            
        if self.perfect_info is not None and perfect_info is not None:
            self.perfect_info[idx] = perfect_info.detach().cpu().numpy()
            
        if action_features is not None:
            self.action_features_data[idx] = [feat.detach().cpu().numpy() for feat in action_features]
            
        if initial_policy_hidden is not None:
            h, c = initial_policy_hidden
            self.initial_policy_hidden_data[idx] = (h.detach().cpu().numpy(), c.detach().cpu().numpy())
            
        if initial_value_hidden is not None:
            h, c = initial_value_hidden
            self.initial_value_hidden_data[idx] = (h.detach().cpu().numpy(), c.detach().cpu().numpy())
            
        # 存储 raw_state
        if raw_state is not None:
            self.raw_states[idx] = raw_state

        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def compute_advantages_and_returns(self, last_value=0):
        """高效计算整个轨迹的折扣回报和GAE优势估计"""
        if self.size == 0: return
        
        path_slice = slice(0, self.size)
        rewards = self.rewards[path_slice]
        values = self.vals[path_slice]
        dones = self.dones[path_slice]
        
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_values = values[t+1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        self.advantages = advantages
        self.returns = advantages + values
    
    def get_minibatch_generator(self, as_tensor_dict=False, preserve_sequence=False):
        """生成训练用的小批量数据"""
        indices = np.arange(self.size)
        if not preserve_sequence:
            np.random.shuffle(indices)

        for start in range(0, self.size, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]
            
            # 准备字典
            batch = {
                'states': torch.tensor(self.states[batch_indices], device=self.device),
                'actions': torch.tensor(self.actions[batch_indices], device=self.device),
                'log_probs': torch.tensor(self.log_probs[batch_indices], device=self.device),
                'returns': torch.tensor(self.returns[batch_indices], device=self.device),
                'advantages': torch.tensor(self.advantages[batch_indices], device=self.device),
            }
            
            if self.action_masks is not None:
                batch['action_masks'] = torch.tensor(self.action_masks[batch_indices], device=self.device)
            if self.perfect_info is not None:
                batch['perfect_info'] = torch.tensor(self.perfect_info[batch_indices], device=self.device)

            if as_tensor_dict:
                yield batch
            else:
                yield tuple(batch.values())
    
    def clear(self):
        """高效清空缓冲区"""
        self.position = 0
        self.size = 0
        self.history_states_data.clear()
        self.action_features_data.clear()
        self.initial_policy_hidden_data.clear()
        self.initial_value_hidden_data.clear()
        self.raw_states = [None] * self.max_size

class RewardShaper:
    """
    辅助 PPOAgent 管理奖励计算相关的状态和逻辑。
    主要负责存储轮次信息（如动作历史、手牌）、局末奖励计算的框架，以及训练阶段的标记。
    具体的轮次奖励组件（首出、出牌差、完美）的计算逻辑在Agent类中。
    """
    def __init__(self, config=None, **kwargs):
        """
        初始化RewardShaper。
        参数:
            config (dict, optional): 包含基础奖励值（如胜利/失败奖励）的配置字典。
            verbose (bool): 是否输出RewardShaper自身的详细日志。
            fine_tuning (bool): 是否处于微调模式。
        """
        self.config = config or {}
        for key, value in kwargs.items(): self.config[key] = value
        
        # 从配置中读取基础奖励值
        self.win_reward = self.config.get('win_reward', 2.0)
        self.lose_reward = self.config.get('lose_reward', -2.0)

        self.team_rank_rewards = {
            # key是0索引的排名元组, value是奖励值
            (0, 1): self.config.get('team_rank_12_reward', 3.0),  # 团队包揽冠亚军
            (0, 2): self.config.get('team_rank_13_reward', 2.0),  # 团队获得第1、3名
            (0, 3): self.config.get('team_rank_14_reward', 1.0),  # 团队获得第1、4名
            (1, 2): self.config.get('team_rank_23_reward', -1.0), # 团队获得第2、3名
            (1, 3): self.config.get('team_rank_24_reward', -2.0), # 团队获得第2、4名
            (2, 3): self.config.get('team_rank_34_reward', -3.0)  # 团队包揽三四名
        }

        #self.quick_win_bonus = self.config.get('quick_win_bonus', 1.0) # 快速胜利的额外奖励
        
        self.verbose = self.config.get('verbose', False) 
        self.fine_tuning = self.config.get('fine_tuning', False)
        
        self.logger = logging.getLogger('reward_shaper_global') 
        if self.verbose and not self.logger.handlers: 
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - RewardShaper - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.action_history = [] # 由Agent的observe方法填充，用于追踪一轮内的动作
        self.turn_count = 0      # 全局轮次计数，由Agent同步
        self.total_rewards = {0:0.0, 1:0.0, 2:0.0, 3:0.0} # 每个Agent实例追踪自己视角的各玩家累计奖励

        # 存储当前轮次的状态信息，由Agent的observe方法填充，Agent的calculate_turn_rewards方法读取
        self.current_turn_states = {
            'last_non_pass_player': -1,   # 本轮最后一个出非PASS牌的玩家
            'turn_actions': [],           # 本轮的动作序列 (元素是字典)
            'all_players_hands': {},      # 本轮结束后各玩家的手牌
            'previous_oracle': None       # 上一轮结束时的Oracle值 (即本轮开始前的Oracle值)
        }

    def set_training_phase(self, fine_tuning):
        """设置当前是否为微调阶段。"""
        self.fine_tuning = fine_tuning
        if self.verbose: self.logger.info(f"RewardShaper 训练阶段设置为: {'微调' if fine_tuning else '标准训练'}")

    def reset(self):
        """在一小局游戏结束时，由Agent调用，重置RewardShaper的状态。"""
        self.turn_count = 0
        self.total_rewards = {0:0.0, 1:0.0, 2:0.0, 3:0.0} # 重置累计奖励
        self.action_history = []
        self.current_turn_states = { # 重置轮次特定状态
            'last_non_pass_player': -1, 'turn_actions': [],
            'all_players_hands': {}, 'previous_oracle': None
        }
        if self.verbose: self.logger.info("RewardShaper 已为新小局重置。")

    def calculate_final_reward(self, state, winner_team, player_id):
        """
        计算一小局结束时的最终奖励。
        基于玩家的个人完成名次和其队伍的整体表现（如是否包揽前两名）。
        """
        final_total_reward_for_player = 0.0
        personal_rank_reward = 0.0
        team_combo_reward = 0.0
        player_team_id = player_id % 2
        
        # 获取玩家完成顺序
        finish_order = state.get('finished_players', [])
        if not finish_order and 'num_cards_left' in state: # 备用逻辑：根据剩余牌数判断
            cards_left = state.get('num_cards_left', {})
            if isinstance(cards_left, dict): finish_order = sorted(range(4), key=lambda p: cards_left.get(p, 999))
        if not finish_order: # 最终备用：简单按胜负队伍排列
            for p_iter in range(4): 
                if p_iter % 2 == winner_team: finish_order.append(p_iter)
            for p_iter in range(4):
                if p_iter % 2 != winner_team and p_iter not in finish_order: finish_order.append(p_iter)

        # 获取当前Agent（调用此方法的那个Agent）的日志器
        agent_logger = logging.getLogger(f'ppo_agent_{player_id}') # Get the specific agent's logger

        if self.verbose or agent_logger.isEnabledFor(logging.INFO): # Check agent's logger level too
            agent_logger.info(f"\n=== P{player_id} 计算小局结束奖励 ===")
            agent_logger.info(f"获胜队伍: {winner_team}, P{player_id} 队伍: {player_team_id}")
            agent_logger.info(f"玩家出牌完成顺序: {finish_order}")

        # 1. 计算个人排名奖励
        if player_id in finish_order:
            rank = finish_order.index(player_id) # 0-indexed rank
            if rank == 0: personal_rank_reward = self.config.get('win_reward', 2.0) 
            elif rank == 1: personal_rank_reward = self.config.get('win_reward', 2.0) * 0.5 
            elif rank == 2: personal_rank_reward = self.config.get('lose_reward', -2.0) * 0.5
            elif rank == 3: personal_rank_reward = self.config.get('lose_reward', -2.0)
        final_total_reward_for_player += personal_rank_reward
        
        if self.verbose or agent_logger.isEnabledFor(logging.INFO): 
            agent_logger.info(f"  1. 个人排名奖励 for P{player_id}: {personal_rank_reward:.1f} (排名 {finish_order.index(player_id)+1 if player_id in finish_order else 'N/A'})")

        # 2. 计算团队组合奖励
        teammate_id = (player_id + 2) % 4
        # 获取本队两名玩家的排名 (0-indexed)
        team_member_ranks = sorted([finish_order.index(p) for p in [player_id, teammate_id] if p in finish_order])

        if self.verbose or agent_logger.isEnabledFor(logging.INFO):
            self_rank_str = str(finish_order.index(player_id)+1) if player_id in finish_order else '未完成'
            mate_rank_str = str(finish_order.index(teammate_id)+1) if teammate_id in finish_order else '未完成'
            agent_logger.info(f"  P{player_id} 队伍成员排名: 自己(P{player_id})第{self_rank_str}名, 队友(P{teammate_id})第{mate_rank_str}名. 排序后0-idx ranks: {team_member_ranks}")
        
        # 只有当本队两名成员都完成时，才计算团队组合奖励
        if len(team_member_ranks) == 2:
            # 将列表转换为元组，以便作为字典的键
            rank_tuple = tuple(team_member_ranks)
            # 从初始化时定义的奖励字典中查找对应的团队奖励
            team_combo_reward = self.team_rank_rewards.get(rank_tuple, 0.0)
            
            if self.verbose or agent_logger.isEnabledFor(logging.INFO): 
                agent_logger.info(f"  2. 团队组合奖励 for P{player_id}'s team (ranks {team_member_ranks}): {team_combo_reward:.1f}")
        elif self.verbose or agent_logger.isEnabledFor(logging.INFO):
            agent_logger.info(f"  2. 团队组合奖励: 团队成员未全部完成，团队奖励为 0.0")

        # 3. 汇总总奖励
        final_total_reward_for_player = personal_rank_reward + team_combo_reward

        if self.verbose or agent_logger.isEnabledFor(logging.INFO):
            agent_logger.info(f"  >>> P{player_id} 总计结束奖励 (个人+团队): {personal_rank_reward:.1f} + {team_combo_reward:.1f} = {final_total_reward_for_player:.2f}")
            agent_logger.info(f"=== P{player_id} 小局结束奖励计算完毕 ===")
            
        return final_total_reward_for_player
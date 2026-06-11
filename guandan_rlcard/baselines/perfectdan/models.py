# models.py - 掼蛋游戏神经网络模型定义
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GuandanLSTMPolicyNetwork(nn.Module):
    """专为掼蛋设计的策略网络，仅负责动作选择"""
    def __init__(self, state_dim, action_feature_dim, abstract_action_dim, 
                 lstm_hidden_dim, ff_hidden_dim, num_lstm_layers, device):
        super(GuandanLSTMPolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim
        self.abstract_action_dim = abstract_action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.ff_hidden_dim = ff_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(state_dim + lstm_hidden_dim, ff_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_hidden_dim, ff_hidden_dim)
        self.output_layer = nn.Linear(ff_hidden_dim, abstract_action_dim)

    def forward(self, state, history, hidden, action_features=None):
        lstm_output, new_hidden = self.lstm(history, hidden)
        last_lstm_output = lstm_output[:, -1, :]
        combined_features = torch.cat([state, last_lstm_output], dim=1)
        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        logits = self.output_layer(x)
        return logits, new_hidden

    
    def _init_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)

class GuandanValueNetwork(nn.Module):
    """专为掼蛋设计的价值网络，使用完美和不完美信息"""
    def __init__(self, state_dim, perfect_info_dim, lstm_hidden_dim, 
                 ff_hidden_dim, num_lstm_layers, device):
        super(GuandanValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.perfect_info_dim = perfect_info_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.ff_hidden_dim = ff_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # 输入维度 = 当前状态维度 + LSTM隐藏层维度 + 完美信息维度
        self.fc1 = nn.Linear(state_dim + lstm_hidden_dim + perfect_info_dim, ff_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_hidden_dim, ff_hidden_dim // 2)
        self.output_layer = nn.Linear(ff_hidden_dim // 2, 1)

    def forward(self, state, history, perfect_info, hidden):
        """
        前向传播。
        
        参数:
            state (torch.Tensor): 当前状态张量，形状 [batch_size, state_dim]
            history (torch.Tensor): 历史状态序列，形状 [batch_size, seq_len, state_dim]
            perfect_info (torch.Tensor): 完美信息张量，形状 [batch_size, perfect_info_dim]
            hidden (tuple): LSTM的隐藏状态 (h_0, c_0)

        返回:
            value (torch.Tensor): 状态价值估计，形状 [batch_size, 1]
            new_hidden (tuple): 更新后的LSTM隐藏状态
        """
        lstm_output, new_hidden = self.lstm(history, hidden)
        last_lstm_output = lstm_output[:, -1, :]
        
        # 拼接所有特征
        if perfect_info is None:
            # 如果没有提供完美信息（例如在评估时），则用零向量代替
            perfect_info = torch.zeros(state.size(0), self.perfect_info_dim, device=self.device)
            
        combined_features = torch.cat([state, last_lstm_output, perfect_info], dim=1)
        
        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        value = self.output_layer(x)
        
        return value, new_hidden

class OptimizedGuandanStateEncoder:
    """
    优化的掼蛋游戏状态编码器
    使用矩阵表示方式编码游戏状态和动作，捕捉游戏的关键特征
    """
    def __init__(self, state_dim=673, device='cpu'):
        """初始化状态编码器"""
        self.state_dim = state_dim
        self.device = device
        
        # 基础矩阵维度
        self.card_matrix_rows = 4  # 四种花色
        self.card_matrix_cols = 15  # 2-A加大小王
        
        # 优化后的特征维度
        self.hand_cards_dim = 54  # 手牌压缩表示 (4×13普通牌 + 2大小王 = 54)
        self.unknown_cards_dim = 54  # 未知牌压缩表示 (同上)
        self.played_cards_dim = 4 * 54  # 四个玩家已出的牌 (每位玩家54维)
        self.recent_plays_dim = 4 * 55  # 四个玩家最近一次出牌 (每位玩家55维，包括红心级牌信息)
        self.lstm_history_out_dim = 128  # LSTM历史特征输出 (128维)
        
        # 初始化LSTM组件用于处理变长历史序列
        self.history_lstm = nn.LSTM(
            input_size=self.card_matrix_rows * self.card_matrix_cols,
            hidden_size=self.lstm_history_out_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        ).to(device)
        
        # 牌面映射
        self.suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        self.rank_map = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 
            'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14
        }
    
    def encode_state(self, state_dict):
        """将游戏状态按照指定方案编码为特征向量"""
        if not isinstance(state_dict, dict):
            print(f"警告: state_dict不是字典类型: {type(state_dict)}")
            return torch.zeros(1, self.state_dim, device=self.device)
        
        features = []
        player_id = state_dict.get("curPos", 0)
        
        # 1. 手牌组合矩阵（54维）
        hand_cards = state_dict.get("handCards", [])
        hand_feature = self._encode_hand_cards(hand_cards)
        features.append(hand_feature)  # 54维
        
        # 2. 未被打出的牌（54维）
        unknown_feature = self._encode_unknown_cards(state_dict)
        features.append(unknown_feature)  # 54维
        
        # 3. 四个玩家已出的牌（4×54=216维）
        player_played_features = self._encode_players_played_cards(state_dict)
        for player_feature in player_played_features:
            features.append(player_feature)  # 每位玩家54维，共216维
        
        # 4-7. 各玩家最近一次出牌（4×55=220维）
        recent_plays = self._encode_last_plays_by_position(state_dict, player_id)
        for player_play in recent_plays:
            features.append(player_play)  # 每个55维，共220维
        
        # 8. 当前自己手牌最少出完牌的步数（1维）
        min_steps = state_dict.get("min_steps_estimation", {}).get(player_id, 1)
        min_steps_feature = torch.tensor([float(min_steps)], dtype=torch.float32, device=self.device)
        features.append(min_steps_feature)  # 1维
        
        # 9. 最近五回合出牌（输入LSTM，输出128维）
        last_trick_plays = state_dict.get("last_trick_plays", [])
        lstm_history_feature = self._process_history_with_lstm(last_trick_plays)
        features.append(lstm_history_feature)  # 128维
        
        # 合并所有特征
        state_tensor = torch.cat(features, dim=0)
        
        # 调整维度
        if state_tensor.size(0) < self.state_dim:
            padding = torch.zeros(self.state_dim - state_tensor.size(0), dtype=torch.float32, device=self.device)
            state_tensor = torch.cat([state_tensor, padding], dim=0)
        elif state_tensor.size(0) > self.state_dim:
            state_tensor = state_tensor[:self.state_dim]
        
        return state_tensor.unsqueeze(0)  # 添加batch维度

    def create_action_features(self, state_dict, action_list):
        """
        为每个动作创建独立的特征向量，优化相同牌型牌力的表示
        
        参数:
        state_dict: 游戏状态字典
        action_list: 合法动作列表
        
        返回:
        action_features: 每个动作的特征向量列表
        action_indices: 保留下来的动作在原始列表中的索引
        """
        player_id = state_dict.get("curPos", 0)
        
        # 获取上下家和队友的剩余牌数
        num_cards_left = state_dict.get("num_cards_left", {})
        if not isinstance(num_cards_left, dict):
            num_cards_left = {}
            
        prev_player = (player_id - 1) % 4  # 上家
        next_player = (player_id + 1) % 4  # 下家
        teammate = (player_id + 2) % 4     # 队友
        
        prev_cards = num_cards_left.get(prev_player, 0)
        next_cards = num_cards_left.get(next_player, 0)
        teammate_cards = num_cards_left.get(teammate, 0)
        
        # 牌型和牌值映射
        type_to_row = {
            'Single': 0, 'Pair': 1, 'Trips': 2, 'ThreeWithTwo': 3,
            'ThreePair': 4, 'TwoTrips': 5, 'Straight': 6, 
            'StraightFlush': 7, 'Bomb': 8
        }
        
        rank_to_col = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 
            'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14
        }
        
        # 从后往前查找炸弹和同花顺
        bomb_card_sets = []  # 储存所有炸弹的牌组
        straight_flush_card_sets = []  # 储存所有同花顺的牌组
        has_found_bombs_straights = False
        
        for action in reversed(action_list):
            if isinstance(action, list) and len(action) >= 1:
                action_type = action[0]
                if action_type == 'Bomb':
                    has_found_bombs_straights = True
                    if len(action) >= 3 and isinstance(action[2], list):
                        bomb_card_sets.append(action[2])
                elif action_type == 'StraightFlush':
                    has_found_bombs_straights = True
                    if len(action) >= 3 and isinstance(action[2], list):
                        straight_flush_card_sets.append(action[2])
                elif action_type == 'Straight' and not has_found_bombs_straights:
                    # 如果已经遍历到Straight，且没有遇到炸弹和同花顺，则可以停止查找
                    break
        
        # 为每个动作创建特征，同时进行优化处理
        action_features = []
        action_indices = []  # 保存特征对应的原始动作索引
        
        # 按照牌型和牌力分组
        action_groups = {}
        
        for i, action in enumerate(action_list):
            if not isinstance(action, list) or len(action) < 2:
                # PASS 或者无效动作
                group_key = 'PASS'
            else:
                action_type = action[0]
                rank = action[1] if len(action) >= 2 else ''
                group_key = f"{action_type}_{rank}"
            
            if group_key not in action_groups:
                action_groups[group_key] = []
            action_groups[group_key].append((i, action))
        
        # 为每个组创建一个代表性特征
        for group_key, group_actions in action_groups.items():
            seen_features = {}  # 用于检测最后两维特征的组合是否重复
            
            for idx, (orig_idx, action) in enumerate(group_actions):
                # 1. 创建空的9×15矩阵（默认全0）
                action_matrix = torch.zeros((9, 15), dtype=torch.float32, device=self.device)
                
                # 2. 设置当前动作的对应位置
                if isinstance(action, list) and len(action) >= 2 and action[0] != 'PASS':
                    action_type = action[0]
                    rank = action[1]
                    
                    if action_type in type_to_row and rank in rank_to_col:
                        row = type_to_row[action_type]
                        col = rank_to_col[rank]
                        action_matrix[row, col] = 1.0
                
                # 3. 展平矩阵为135维向量
                flat_matrix = action_matrix.reshape(-1)
                
                # 4. 添加是否为PASS的标志（1维）
                is_pass = torch.tensor([1.0 if isinstance(action, list) and action[0] == 'PASS' else 0.0], 
                                    dtype=torch.float32, device=self.device)
                
                # 5. 添加动作牌力值（1维）- 根据位置，值越靠后牌力值越大
                power_value = float(orig_idx) / (len(action_list) - 1) if len(action_list) > 1 else 0.0
                feature_power = torch.tensor([power_value], dtype=torch.float32, device=self.device)
                
                # 6. 添加动作与玩家手牌数相等关系（3维）
                cards_equals = torch.zeros(3, dtype=torch.float32, device=self.device)
                
                # 7-9. 添加特殊牌型特征（炸弹、同花顺、红心通配牌）
                in_bomb = torch.tensor([0.0], dtype=torch.float32, device=self.device)
                in_straight_flush = torch.tensor([0.0], dtype=torch.float32, device=self.device)
                use_heart_wildcard = torch.tensor([0.0], dtype=torch.float32, device=self.device)
                
                # 统计动作中的牌数并填充特征
                if isinstance(action, list) and len(action) >= 3 and isinstance(action[2], list):
                    cards = action[2]
                    cards_count = len(cards)
                    
                    # 判断牌数是否等于各玩家剩余牌数
                    cards_equals[0] = 1.0 if cards_count == prev_cards else 0.0    # 上家
                    cards_equals[1] = 1.0 if cards_count == next_cards else 0.0    # 下家
                    cards_equals[2] = 1.0 if cards_count == teammate_cards else 0.0 # 队友
                    
                    # 检测是否包含炸弹中的牌
                    if action[0] == 'Bomb':
                        # 如果动作本身是炸弹，则标记为1
                        in_bomb[0] = 1.0
                    else:
                        # 检查动作中的牌是否存在于任何一个炸弹组牌中
                        for bomb_cards in bomb_card_sets:
                            if self._has_overlapping_cards(cards, bomb_cards):
                                in_bomb[0] = 1.0
                                break
                    
                    # 检测是否包含同花顺中的牌
                    if action[0] == 'StraightFlush':
                        # 如果动作本身是同花顺，则标记为1
                        in_straight_flush[0] = 1.0
                    else:
                        # 检查动作中的牌是否存在于任何一个同花顺组牌中
                        for sf_cards in straight_flush_card_sets:
                            if self._has_overlapping_cards(cards, sf_cards):
                                in_straight_flush[0] = 1.0
                                break
                    
                    # 检测是否使用红心通配牌
                    if action[0] in ['Straight', 'ThreePair', 'TwoTrips']:
                        for card in cards:
                            if isinstance(card, str) and len(card) >= 2 and card[0] == 'H':
                                use_heart_wildcard[0] = 1.0
                                break
                
                # 合并所有特征 (135 + 1 + 1 + 3 + 1 + 1 + 1 = 143)
                complete_feature = torch.cat([
                    flat_matrix, is_pass, feature_power, cards_equals,
                    in_bomb, in_straight_flush, use_heart_wildcard
                ])
                
                # 检查最后两维特征是否重复（同花顺和红心通配牌）
                last_two_dims = (in_straight_flush.item(), use_heart_wildcard.item())
                
                if last_two_dims not in seen_features:
                    seen_features[last_two_dims] = True
                    action_features.append(complete_feature)
                    action_indices.append(orig_idx)
        
        return action_features, action_indices

    def _parse_card(self, card_str):
        """解析卡牌字符串返回花色和点数索引"""
        if len(card_str) < 2:
            return 0, 0  # 默认值
        
        # 解析花色
        suit_char = card_str[0]
        suit = self.suit_map.get(suit_char, 0)
        
        # 解析点数
        rank_str = card_str[1:]
        if rank_str == 'BJ':
            rank = 13  # 小王
        elif rank_str == 'RJ':
            rank = 14  # 大王
        else:
            rank = self.rank_map.get(rank_str, 0)
        
        return suit, rank
    
    def _encode_hand_cards(self, hand_cards):
        """编码手牌为压缩的54维向量"""
        # 先创建4×15矩阵表示
        hand_matrix = torch.zeros((self.card_matrix_rows, self.card_matrix_cols), dtype=torch.float32, device=self.device)
        
        for card in hand_cards:
            if isinstance(card, str) and len(card) >= 2:
                suit, rank = self._parse_card(card)
                if 0 <= suit < self.card_matrix_rows and 0 <= rank < self.card_matrix_cols:
                    hand_matrix[suit, rank] += 1
        
        # 压缩为54维向量
        compressed_hand = torch.zeros(54, dtype=torch.float32, device=self.device)
        
        # 1. 复制普通牌 (4×13 = 52维)
        idx = 0
        for suit in range(4):
            for rank in range(13):  # 只包括2-A (前13列)
                compressed_hand[idx] = hand_matrix[suit, rank]
                idx += 1
        
        # 2. 大小王 (2维)
        compressed_hand[52] = hand_matrix[0, 13]  # 小王 (位于第0行第13列)
        compressed_hand[53] = hand_matrix[0, 14]  # 大王 (位于第0行第14列)
        
        return compressed_hand
    
    def _encode_unknown_cards(self, state_dict):
        """编码未知牌为压缩的54维向量"""
        # 开始于全2矩阵（两副牌）
        unknown_matrix = torch.full((self.card_matrix_rows, self.card_matrix_cols), 2, dtype=torch.float32, device=self.device)
        
        # 减去手牌
        hand_cards = state_dict.get("handCards", [])
        for card in hand_cards:
            if isinstance(card, str) and len(card) >= 2:
                suit, rank = self._parse_card(card)
                if 0 <= suit < self.card_matrix_rows and 0 <= rank < self.card_matrix_cols:
                    unknown_matrix[suit, rank] -= 1
        
        # 减去已出牌
        played_cards = state_dict.get("played_cards", {})
        if isinstance(played_cards, dict):
            for player_id, cards in played_cards.items():
                if isinstance(cards, list):
                    for card in cards:
                        if isinstance(card, str) and len(card) >= 2:
                            suit, rank = self._parse_card(card)
                            if 0 <= suit < self.card_matrix_rows and 0 <= rank < self.card_matrix_cols:
                                unknown_matrix[suit, rank] -= 1
        
        # 确保无负数
        unknown_matrix = torch.clamp(unknown_matrix, min=0)
        
        # 压缩为54维向量
        compressed_unknown = torch.zeros(54, dtype=torch.float32, device=self.device)
        
        # 1. 复制普通牌 (4×13 = 52维)
        idx = 0
        for suit in range(4):
            for rank in range(13):  # 只包括2-A (前13列)
                compressed_unknown[idx] = unknown_matrix[suit, rank]
                idx += 1
        
        # 2. 大小王 (2维)
        compressed_unknown[52] = unknown_matrix[0, 13]  # 小王
        compressed_unknown[53] = unknown_matrix[0, 14]  # 大王
        
        return compressed_unknown
    
    def _encode_players_played_cards(self, state_dict):
        """
        编码四个玩家已出的牌，每位玩家一个54维向量
        
        返回:
        player_features: 包含四个玩家已出牌特征的列表，每个元素是54维向量
        """
        player_features = []
        
        # 获取所有玩家已出的牌
        played_cards = state_dict.get("played_cards", {})
        if not isinstance(played_cards, dict):
            played_cards = {}
        
        # 为每个玩家生成已出牌特征
        for player_id in range(4):
            # 创建4×15矩阵
            player_matrix = torch.zeros((self.card_matrix_rows, self.card_matrix_cols), 
                                     dtype=torch.float32, device=self.device)
            
            # 填充该玩家已出的牌
            player_cards = played_cards.get(player_id, [])
            if isinstance(player_cards, list):
                for card in player_cards:
                    if isinstance(card, str) and len(card) >= 2:
                        suit, rank = self._parse_card(card)
                        if 0 <= suit < self.card_matrix_rows and 0 <= rank < self.card_matrix_cols:
                            player_matrix[suit, rank] += 1
            
            # 压缩为54维向量
            compressed_player = torch.zeros(54, dtype=torch.float32, device=self.device)
            
            # 1. 复制普通牌 (4×13 = 52维)
            idx = 0
            for suit in range(4):
                for rank in range(13):  # 只包括2-A (前13列)
                    compressed_player[idx] = player_matrix[suit, rank]
                    idx += 1
            
            # 2. 大小王 (2维)
            compressed_player[52] = player_matrix[0, 13]  # 小王
            compressed_player[53] = player_matrix[0, 14]  # 大王
            
            player_features.append(compressed_player)
        
        return player_features

    def _encode_last_plays_by_position(self, state_dict, player_id):
        """
        编码各个位置玩家的最近一次出牌（自己、上家、下家、队友）
        
        返回:
        [self_play, prev_play, next_play, teammate_play]: 列表，每个元素是55维向量
        """
        self_id = player_id
        prev_id = (player_id - 1) % 4  # 上家
        next_id = (player_id + 1) % 4  # 下家
        teammate_id = (player_id + 2) % 4  # 队友
        
        # 获取所有玩家的最近出牌记录
        history = state_dict.get("history", {})
        if not isinstance(history, dict):
            history = {}
        
        # 创建四个55维向量，分别表示四个位置玩家的最近出牌
        self_play = torch.zeros(55, dtype=torch.float32, device=self.device)
        prev_play = torch.zeros(55, dtype=torch.float32, device=self.device)
        next_play = torch.zeros(55, dtype=torch.float32, device=self.device)
        teammate_play = torch.zeros(55, dtype=torch.float32, device=self.device)
        
        # 获取自己最近出牌
        self_history = history.get(str(self_id), {})
        if isinstance(self_history, dict):
            send_history = self_history.get('send', [])
            if isinstance(send_history, list) and send_history:
                self_last_action = send_history[-1]
                if isinstance(self_last_action, list) and len(self_last_action) >= 3:
                    self_play = self._encode_action_to_feature(self_last_action)
        
        # 获取上家最近出牌
        prev_history = history.get(str(prev_id), {})
        if isinstance(prev_history, dict):
            send_history = prev_history.get('send', [])
            if isinstance(send_history, list) and send_history:
                prev_last_action = send_history[-1]
                if isinstance(prev_last_action, list) and len(prev_last_action) >= 3:
                    prev_play = self._encode_action_to_feature(prev_last_action)
        
        # 获取下家最近出牌
        next_history = history.get(str(next_id), {})
        if isinstance(next_history, dict):
            send_history = next_history.get('send', [])
            if isinstance(send_history, list) and send_history:
                next_last_action = send_history[-1]
                if isinstance(next_last_action, list) and len(next_last_action) >= 3:
                    next_play = self._encode_action_to_feature(next_last_action)
        
        # 获取队友最近出牌
        teammate_history = history.get(str(teammate_id), {})
        if isinstance(teammate_history, dict):
            send_history = teammate_history.get('send', [])
            if isinstance(send_history, list) and send_history:
                teammate_last_action = send_history[-1]
                if isinstance(teammate_last_action, list) and len(teammate_last_action) >= 3:
                    teammate_play = self._encode_action_to_feature(teammate_last_action)
        
        return [self_play, prev_play, next_play, teammate_play]
    
    def _encode_action_to_feature(self, action):
        """将动作编码为55维特征向量"""
        # 创建55维向量
        action_feature = torch.zeros(55, dtype=torch.float32, device=self.device)
        
        # 如果是PASS动作
        if action[0] == 'PASS':
            return action_feature  # 返回全零向量
        
        # 获取动作中的牌列表
        cards = action[2] if len(action) >= 3 and isinstance(action[2], list) else []
        
        # 创建临时矩阵
        card_matrix = torch.zeros((self.card_matrix_rows, self.card_matrix_cols), 
                                dtype=torch.float32, device=self.device)
        
        # 红心级牌标志
        heart_rank_used = False
        
        # 填充牌面
        for card in cards:
            if isinstance(card, str) and len(card) >= 2:
                suit, rank = self._parse_card(card)
                if 0 <= suit < self.card_matrix_rows and 0 <= rank < self.card_matrix_cols:
                    card_matrix[suit, rank] += 1
                    
                    # 检查是否使用了红心级牌
                    if suit == 1:  # 红心
                        heart_rank_used = True
        
        # 压缩为54维
        idx = 0
        for suit in range(4):
            for rank in range(13):  # 只包括2-A (前13列)
                action_feature[idx] = card_matrix[suit, rank]
                idx += 1
        
        # 大小王
        action_feature[52] = card_matrix[0, 13]  # 小王
        action_feature[53] = card_matrix[0, 14]  # 大王
        
        # 第55维表示是否使用了红心级牌
        if action[0] in ['Straight', 'ThreePair', 'TwoTrips'] or heart_rank_used:
            action_feature[54] = 1.0
        
        return action_feature
    
    def _process_history_with_lstm(self, last_trick_plays):
        """使用LSTM处理变长历史序列，处理最近五轮出牌"""
        if not last_trick_plays:
            return torch.zeros(self.lstm_history_out_dim, dtype=torch.float32, device=self.device)
        
        # 对历史轮次进行限制，最多保留最近5轮
        history_tricks = last_trick_plays[-5:] if len(last_trick_plays) > 5 else last_trick_plays
        
        # 将每轮出牌编码为5×220维的表示
        all_tricks_feature = torch.zeros(5, 220, dtype=torch.float32, device=self.device)
        
        for i, trick in enumerate(history_tricks):
            if i >= 5:  # 最多处理5轮
                break
                
            # 对于每轮，获取四个玩家的出牌并编码为55×4=220维
            if isinstance(trick, dict):
                # 如果是字典形式，按玩家ID获取
                for pid in range(4):
                    player_cards = trick.get(str(pid), [])
                    if isinstance(player_cards, list):
                        # 编码成55维
                        player_feature = self._encode_player_trick(player_cards)
                        # 存入对应位置
                        all_tricks_feature[i, pid*55:(pid+1)*55] = player_feature
            elif isinstance(trick, list):
                # 如果是列表形式，假设每个元素是一张牌
                # 将所有牌编码为一个55维向量放在第一个玩家位置
                all_cards_feature = self._encode_player_trick(trick)
                all_tricks_feature[i, 0:55] = all_cards_feature
        
        # 展平为一维向量用于LSTM处理
        flat_tricks = all_tricks_feature.view(1, -1)  # [1, 5*220]
        
        # 使用LSTM处理
        # 需要调整LSTM的输入维度以匹配新的特征大小
        # 这里简化处理，直接返回处理后的特征
        # 实际应用中，可能需要重新设计LSTM网络
        
        # 简化处理：使用线性投影将扁平特征投影到LSTM输出维度
        projection = nn.Linear(flat_tricks.size(1), self.lstm_history_out_dim).to(self.device)
        with torch.no_grad():
            lstm_feature = projection(flat_tricks).squeeze(0)
            
        return lstm_feature
    
    def _encode_player_trick(self, cards):
        """将一位玩家的出牌编码为55维向量"""
        # 创建一个矩阵来记录牌面
        card_matrix = torch.zeros((self.card_matrix_rows, self.card_matrix_cols), 
                                dtype=torch.float32, device=self.device)
        
        # 红心级牌标志
        heart_rank_used = False
        
        # 填充牌面
        for card in cards:
            if isinstance(card, str) and len(card) >= 2:
                suit, rank = self._parse_card(card)
                if 0 <= suit < self.card_matrix_rows and 0 <= rank < self.card_matrix_cols:
                    card_matrix[suit, rank] += 1
                    
                    # 检查是否使用了红心级牌
                    if suit == 1:  # 红心
                        heart_rank_used = True
        
        # 压缩为55维向量
        compressed_cards = torch.zeros(55, dtype=torch.float32, device=self.device)
        
        # 1. 普通牌 (4×13 = 52维)
        idx = 0
        for suit in range(4):
            for rank in range(13):  # 只包括2-A (前13列)
                compressed_cards[idx] = card_matrix[suit, rank]
                idx += 1
        
        # 2. 大小王 (2维)
        compressed_cards[52] = card_matrix[0, 13]  # 小王
        compressed_cards[53] = card_matrix[0, 14]  # 大王
        
        # 3. 红心级牌 (1维)
        compressed_cards[54] = 1.0 if heart_rank_used else 0.0
        
        return compressed_cards
        
    def _has_overlapping_cards(self, cards1, cards2):
        """
        判断两个牌组是否有重叠的牌
        
        参数:
        cards1, cards2: 两个牌组列表
        
        返回:
        bool: 如果有重叠的牌则返回True
        """
        # 创建两个牌组的集合
        set1 = set(cards1)
        set2 = set(cards2)
        
        # 检查两个集合是否有交集
        return len(set1.intersection(set2)) > 0
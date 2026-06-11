
from random import randint

import copy
import torch
import numpy as np
from guandan_rlcard.game.player import GuandanPlayer as Player
from .models import PlayerModel
from .env_utils import card2array, card2num, action_vector, combine_handcards
# from danzero.actor import Action
from ..danzero.danzero_agent import DanzeroAgent

RANK = {
    '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8,
    'T':9, 'J':10, 'Q':11, 'K':12, 'A':13
}

CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
    'A', 'BJ', 'RJ']


RANK2 = {
    '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
    'T':10, 'J':11, 'Q':12, 'K':13, 'A':1
}

actions_dict = {
    'Single': 0,
    'Pair': 1,
    'Trips': 2,
    'ThreePair': 3,
    'ThreeWithTwo': 4,
    'TwoTrips': 5,
    'Straight': 6,
    'StraightFlush': 7,
    'Bomb': 8,
    'tribute': 9,
    # 'back': 10,
    'PASS': 11
}

actions_dict_1 = {
    'Single': 0,
    'Pair': 1,
    'Trips': 2,
    'ThreePair': 3,
    'ThreeWithTwo': 4,
    'TwoTrips': 5,
    'Straight': 6,
    'StraightFlush': 7,
    'Bomb': 8,
    'tribute': 9,
    'back': 10,
    'PASS': 11
}


actions_index = {
    '0': 1,
    '1': 2,
    '2': 3,
    '3': 6,
    '4': 5,
    '5': 6,
    '6': 5,
    '7': 5,
    '8': 8,  # 炸弹最高有八张
    # 'tribute': 9,
    # 'back': 10,
    # 'PASS': 11
}


card_value_idx = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 
    'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14,'JOKER':13
}

card_value_idx_1 = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 
    'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14
}


flower_idx = {
    0: 'H' , 1: 'S' , 2:'C' , 3:'D' , 4:'A' , 5:'N'

}

CardToNum_hand = {
    'H': {'2': (0, 0), '3': (1, 0), '4': (2, 0), '5': (3, 0), '6': (4, 0), '7': (5, 0), '8': (6, 0), '9': (7, 0), 'T': (8, 0), 'J': (9, 0), 'Q': (10, 0), 'K': (11, 0), 'A': (12, 0), 'B': (13, 0), 'R': (14, 0)},
    'S': {'2': (0, 1), '3': (1, 1), '4': (2, 1), '5': (3, 1), '6': (4, 1), '7': (5, 1), '8': (6, 1), '9': (7, 1), 'T': (8, 1), 'J': (9, 1), 'Q': (10, 1), 'K': (11, 1), 'A': (12, 1), 'B': (13, 1), 'R': (14, 1)},
    'C': {'2': (0, 2), '3': (1, 2), '4': (2, 2), '5': (3, 2), '6': (4, 2), '7': (5, 2), '8': (6, 2), '9': (7, 2), 'T': (8, 2), 'J': (9, 2), 'Q': (10, 2), 'K': (11, 2), 'A': (12, 2), 'B': (13, 2), 'R': (14, 2)},
    'D': {'2': (0, 3), '3': (1, 3), '4': (2, 3), '5': (3, 3), '6': (4, 3), '7': (5, 3), '8': (6, 3), '9': (7, 3), 'T': (8, 3), 'J': (9, 3), 'Q': (10, 3), 'K': (11, 3), 'A': (12, 3), 'B': (13, 3), 'R': (14, 3)},
}


ActionToRank = {
    0: {0: 0, 1: -99, 2: -99, 3: -99, 4: -99, 5: -99, 6: -99, 7: -99},  # Single
    1: {0: 0, 1: 0, 2: -99, 3: -99, 4: -99, 5: -99, 6: -99, 7: -99},  # Pair
    2: {0: 0, 1: 0, 2: 0, 3: -99, 4: -99, 5: -99, 6: -99, 7: -99},  # Trips
    3: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: -99, 7: -99},  # ThreePair
    4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: -99, 6: -99, 7: -99},  # ThreeWithTwo 后两部分为0的原因时因为要读取的是act2部分
    5: {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: -99, 7: -99},  # TwoTrips
    6: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: -99, 6: -99, 7: -99},  # Straight
    7: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: -99, 6: -99, 7: -99},  # StraightFlush
    8: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},  # Bomb 炸弹这里要做特殊处理当前一部分使用了pass之后，就不允许继续了
}



# 反向映射
reverse_actions_dict = {v: k for k, v in actions_dict_1.items()}
reverse_card_value_idx = {v: k for k, v in card_value_idx_1.items()}
reverse_rank2_value_idx = {v: k for k, v in RANK2.items()}

def _action_seq_list2array(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = card2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 216)
    return action_seq_array


def _process_action_seq(sequence, length=20):
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


def _get_one_hot_array(num_left_cards, max_num_cards, flag):
    if flag == 0:     # 级数的情况
        one_hot = np.zeros(max_num_cards)
        one_hot[num_left_cards - 1] = 1
    else:
        one_hot = np.zeros(max_num_cards+1)    # 剩余的牌（0-1阵格式）
        one_hot[num_left_cards] = 1
    return one_hot

class PPOAgent(Player):
    def __init__(self, player_id, np_random, model=None, device='cpu'):
        super().__init__(player_id, np_random)
        self.name = 'ppo_agent'
        self.use_raw = True
        self.epsilon = 0.2
        if not device == "cpu":
            self.device = 'cuda:' + str(device)
        else:
            self.device = device
        if model:
            self.model = model
            self.device = next(model.parameters()).device


        # self.mode_q = DanzeroAgent(player_id, np_random)

    def step(self, state, model=None, training=False):
        if not state['actions']:
            return []
        msg = self.parse(state)
        obs = self.state_to_obs(msg, self.player_id)
        obs_input = torch.from_numpy(obs['last_index_batch']).float().to(self.device)
        lstm_input = torch.from_numpy(obs['lstm_input']).float().to(self.device)
        # print(obs_input.shape)
        # print(lstm_input.shape)
        act_index = 0
        with torch.no_grad():
            if training:
                output = model.forward(str(self.player_id) , lstm_input, obs_input)
                li_values = output['values']
                if np.random.rand() < self.epsilon:
                    index = torch.randint(li_values.shape[0], (1,))[0]
                else:
                    index = torch.argmax(li_values).item()
            else:
                output = self.model.forward(lstm_input , obs_input)
                li_values = output['values']
                index = torch.argmax(li_values).item()
        act_index = index
        final_action = state['actions'][act_index]
        # print(final_action)
        return final_action

    def parse(self, state):
        assert type(state) == dict
        msg = {}
        rank_list = state['rank_list']
        play_team = state['play_team']
        msg['curPos'] = self.player_id
        msg['indexRange'] = len(state["actions"]) - 1
        msg['curAction'] = state['greaterAction']
        msg['curRank'] = CARD_RANK[rank_list[play_team]] # str类型
        msg['remain_cards'] = state['remain_cards']
        msg['pass_num'] = state['pass_num'][self.player_id]
        msg['my_pass_num'] = state['my_pass_num'][self.player_id]


        check_list = []
        for t in state["trace"]:
            check_list.append(t[1])        
        msg['actions_seq'] = []
        for action in check_list:
            num_action = card2num(action[2])
            if(num_action == [-1]):
                msg['actions_seq'].append([])
            else:
                msg['actions_seq'].append(num_action)



        msg['others_hand'] = state['others_hand']
        # print(state['others_hand'])
        # exit()


        msg['history'] =  {
            '0': {
                'send': state['played_cards'][0],
                'remain': state['num_cards_left'][0],
            },
            '1': {
                'send': state['played_cards'][1],
                'remain': state['num_cards_left'][1],
            },
            '2': {
                'send': state['played_cards'][2],
                'remain': state['num_cards_left'][2],
            },
            '3': {
                'send': state['played_cards'][3],
                'remain': state['num_cards_left'][3],
            },
        }
        msg["greaterPos"] = state['greaterPos']
        msg["greaterAction"] = state['greaterAction']
        msg["actionList"] = state['actions']
        msg["handCards"] = state['current_hand']
        msg["publicInfo"] = {0:{}, 1:{}, 2:{}, 3:{}}
        msg["num_cards_left"] = state['num_cards_left']
        msg["trace"] = state["trace"]
        msg["rank_list"] = rank_list
        msg["play_team"] = play_team

        for i in range(4):
            msg['publicInfo'][i]['rest'] = state['num_cards_left'][i]



        return msg

    def reset(self, result=None):
        super().reset()
        self.begin = True
        # self.mode_q.reset()
    def get_reward(self, message):
        team = [self.player_id, (self.player_id + 2) % 4]
        order = message['order']
        rewards = {"1100": 3, "1010": 2, "1001": 1, "0110": -1, "0101": -2, "0011": -3}
        res = ""
        for i in order:
            if i in team:
                res += '1'
            else:
                res += '0'
        return rewards[res]

    # def state_to_legalaction(self , message , player_id):
    #     legal_action_vector = [0] * 12
    #     # 第一层合法动作


    def state_to_obs(self, message, player_id):

        # 第一层：合法动作
        legal_action_vector = [0] * 12
        # 第二层：合法主牌
        action_rank_matrix = np.zeros((12, 16), dtype=int)
        # 第三层：合法花牌
        action_rank_matrix_second = np.zeros((15, 16), dtype=int)

        for action in message['actionList']:
            action_type = action[0]
            rank_type = action[1]
            cards = action[2]        
            if action_type == 'PASS':
                legal_action_vector[11] = 1 
                action_rank_matrix[11][15] = 1
                continue            
            if action_type in actions_dict:
                legal_action_vector[actions_dict[action_type]] = 1    
                action_rank_matrix[actions_dict[action_type]][card_value_idx[rank_type]] = 1
        
        num_legal_actions = message['indexRange'] + 1
        # legal_actions = [card2num(i[2]) for i in message['actionList']]
        my_handcards = card2array(card2num(message['handCards']))   # 自己的手牌, (1，54)

        my_handcards_batch = np.repeat(my_handcards[np.newaxis, :], # (num_legal_actions, 54)
                                    num_legal_actions, axis=0)

        universal_card_flag = np.zeros(13, dtype=np.int8)           # (1, 13) 标记通配牌及其数量
        universal_card_flag[RANK[message['curRank']]-1] += my_handcards[(RANK[message['curRank']]-1)*4]
        
        universal_card_flag_batch = np.repeat(universal_card_flag[np.newaxis, :],  # (num_legal_actions, 13)
                                    num_legal_actions, axis=0)

        other_handcards = np.zeros(54, dtype=np.int8)   # 剩余牌库，（1，54）
        for suit in ['H', 'S', 'C', 'D']:
            index, s_id = 0, 0
            for x in message['remain_cards'][suit]:
                if x != 0:
                    if suit == 'H' and index == 13:
                        other_handcards[-1] = x # 大王
                    elif suit == 'S' and index == 13:
                        other_handcards[-2] = x # 小王
                    else:
                        other_handcards[s_id + index*4] = x
                index += 1
            s_id += 1
        
        other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],  # (num_legal_actions, 54)
                                        num_legal_actions, axis=0)

        # greater_action (1, 83)
        # 分四层，0-3位记录玩家，4-13位记录牌型，14-28记录牌力，29-82记录组牌
        # 若为空，则全为-1 (理论上不可能是PASS)
        last_action = -1*np.ones(83, dtype=np.int8)        
        if message['greaterAction'] != []:
            action = message['greaterAction']
            if action[0] == 'PASS':
                raise ValueError('greaterAction is PASS!') 
            if message['greaterPos'] != -1 or message['greaterPos'] != player_id:
                last_action = action_vector(message['greaterPos'], action)
            
        
        last_action_batch = np.repeat(last_action[np.newaxis, :],
                                    num_legal_actions, axis=0)
        
        # print(message['greaterAction'])
        # print('last_action_vec: ', last_action, end='\n\n')
        
        # 同理，动作向量形状为 (1, 83)
        last_teammate_action = -1*np.ones(83, dtype=np.int8)                  # 队友最后的动作
        if message['publicInfo'][(player_id+2)%4]['rest']!=0 and len(message['trace']):
            # 队友没走，且出过牌
            action = []
            for record in message['trace']:
                if record[0] == (player_id+2) % 4:
                    action = record[1]
            if action != []:
                # 队友动作存在（出过牌）
                last_teammate_action = action_vector((player_id+2) % 4, action)
        
        last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)
        
        # print('last_teammate_action_vec: ', last_teammate_action, end='\n\n')

        my_action_batch = np.zeros((num_legal_actions, 83), dtype=np.int8)     # 合法动作，(num_legal_actions, 83)
        for j, action in enumerate(message['actionList']):
            my_action_batch[j, :] = action_vector(player_id, action)

        # 剩余牌数
        down_num_cards_left = _get_one_hot_array(message['publicInfo'][(player_id + 1) % 4]['rest'], 27, 1)   # 下家剩余的牌数， 27维
        
        down_num_cards_left_batch = np.repeat(down_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        teammate_num_cards_left = _get_one_hot_array(message['publicInfo'][(player_id + 2) % 4]['rest'], 27, 1)   # 对家剩余的牌数

        teammate_num_cards_left_batch = np.repeat(teammate_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        up_num_cards_left = _get_one_hot_array(message['publicInfo'][(player_id + 3) % 4]['rest'], 27, 1)   # 上家剩余的牌数
        
        up_num_cards_left_batch = np.repeat(up_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        # print('up_num_cards_left_vec: ', up_num_cards_left)
        # print('teammate_num_cards_left_vec: ', teammate_num_cards_left)

        # 已出牌的记录
        if len(message['history'][str((player_id + 1) % 4)]['send']) > 0:
            down_played_cards = card2array(card2num(message['history'][str((player_id + 1) % 4)]['send']))    # 下家打过的牌， 54维
        else:
            down_played_cards = card2array([])
        
        down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(message['history'][str((player_id + 2) % 4)]['send']) > 0:
            teammate_played_cards = card2array(card2num(message['history'][str((player_id + 2) % 4)]['send']))    # 对家打过的牌
        else:
            teammate_played_cards = card2array([])

        teammate_played_cards_batch = np.repeat(teammate_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(message['history'][str((player_id + 3) % 4)]['send']) > 0:
            up_played_cards = card2array(card2num(message['history'][str((player_id + 3) % 4)]['send']))    # 上家打过的牌
        else:
            up_played_cards = card2array([])

        up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        # print('up_num_cards_played_vec: ', up_played_cards)
        # print('teammate_num_cards_played_vec: ', teammate_played_cards)

        # 等级
        ori_self_rank = message['rank_list'][player_id % 2]
        self_rank = _get_one_hot_array(ori_self_rank+1, 13, 0)         # 己方当前的级牌，13维
        self.rank = ori_self_rank+1

        self_rank_batch = np.repeat(self_rank[np.newaxis, :], num_legal_actions, axis=0)

        ori_oppo_rank = message['rank_list'][(player_id+1)%2]
        oppo_rank = _get_one_hot_array(ori_oppo_rank+1, 13, 0)         # 敌方当前的级牌
        self.oppo_rank = ori_oppo_rank+1

        oppo_rank_batch = np.repeat(oppo_rank[np.newaxis, :], num_legal_actions, axis=0)

        cur_rank = _get_one_hot_array(RANK[message['curRank']], 13, 0)         # 当前的级牌

        cur_rank_batch = np.repeat(cur_rank[np.newaxis, :], num_legal_actions, axis=0)

        # 历史动作，取最近8个动作
        history_act = np.empty((8, 83), dtype=np.int8)
        if len(message['trace']) < 8:
            for index in range(8):
                if index < (8-len(message['trace'])):
                    history_act[index] = -1*np.ones(83, dtype=np.int8)
                else:
                    history_act[index] = action_vector(message['trace'][7-index][0], message['trace'][7-index][1])
        else:
            for id, act in enumerate(message['trace'][-8:]) :
                history_act[id] = action_vector(act[0], act[1])
        
        z_batch = np.repeat(history_act[np.newaxis, :], num_legal_actions, axis=0)        
        # print('history_act: ', history_act)

        x_batch = np.hstack((my_handcards_batch,
                        universal_card_flag_batch,
                        other_handcards_batch,
                        last_action_batch,
                        last_teammate_action_batch,
                        down_played_cards_batch,
                        teammate_played_cards_batch,
                        up_played_cards_batch,
                        down_num_cards_left_batch,
                        teammate_num_cards_left_batch,
                        up_num_cards_left_batch,
                        self_rank_batch,
                        oppo_rank_batch,
                        cur_rank_batch,
                        my_action_batch))
        x_no_action = np.hstack((my_handcards_batch,
                        universal_card_flag_batch,
                        other_handcards_batch,
                        last_action_batch,
                        last_teammate_action_batch,
                        down_played_cards_batch,
                        teammate_played_cards_batch,
                        up_played_cards_batch,
                        down_num_cards_left_batch,
                        teammate_num_cards_left_batch,
                        up_num_cards_left_batch,
                        self_rank_batch,
                        oppo_rank_batch,
                        cur_rank_batch,
                    ))
        processed_seq = _process_action_seq(message['actions_seq'])
        lstm_input = _action_seq_list2array(processed_seq)  # 再转换为(5, 216)
        lstm_input = np.repeat(lstm_input[np.newaxis, :], num_legal_actions, axis=0) # 添加重复创建batchsize变量（batchsize ， 5 ，216）

        out_legal_actions = np.zeros(9*15+1, dtype=int)


        # 添加完美信息

        player_up_cards = card2array(card2num(message['others_hand']['player_up']))   # 自己的手牌, (1，54)

        player_up_cards_batch = np.repeat(player_up_cards[np.newaxis, :], # (num_legal_actions, 54)
                                    num_legal_actions, axis=0)

        player_down_cards = card2array(card2num(message['others_hand']['player_down']))   # 自己的手牌, (1，54)

        player_down_cards_batch = np.repeat(player_down_cards[np.newaxis, :], # (num_legal_actions, 54)
                                    num_legal_actions, axis=0)

        player_opp_cards = card2array(card2num(message['others_hand']['player_opp']))   # 自己的手牌, (1，54)

        player_opp_cards_batch = np.repeat(player_opp_cards[np.newaxis, :], # (num_legal_actions, 54)
                                    num_legal_actions, axis=0)


        
        # print("player_up:" , player_up_cards.shape)
        # print("player_up:" , player_up_cards_batch.shape)
        # print("player_down:" , player_down_cards.shape)
        # print("player_down:" , player_down_cards_batch.shape)
        # print("player_opp_cards:" , player_opp_cards.shape)
        # print("player_opp_cards:" , player_opp_cards_batch.shape)
        # exit()

        for i in range(9):
            if(legal_action_vector[i] == 0):
                continue
            for j in range(15):
                out_legal_actions[i*15 + j] = action_rank_matrix[i][j]

        if legal_action_vector[11] == 1:
            out_legal_actions[-1] = 1
        # 这里直接去掉完美信息部分
        last_index_batch = np.hstack((my_handcards_batch,
                        universal_card_flag_batch,
                        other_handcards_batch,
                        last_action_batch,
                        last_teammate_action_batch,
                        down_played_cards_batch,
                        teammate_played_cards_batch,
                        up_played_cards_batch,
                        down_num_cards_left_batch,
                        teammate_num_cards_left_batch,
                        up_num_cards_left_batch,
                        self_rank_batch,
                        oppo_rank_batch,
                        cur_rank_batch,
                        # player_down_cards_batch,
                        # player_opp_cards_batch,
                        # player_up_cards_batch,
                        my_action_batch,
                        ))
        last_index_no_action_batch = np.concatenate((my_handcards,   #54
                    universal_card_flag,        
                    other_handcards,
                    last_action,
                    last_teammate_action,
                    down_played_cards,
                    teammate_played_cards,
                    up_played_cards,
                    down_num_cards_left,
                    teammate_num_cards_left,
                    up_num_cards_left,
                    self_rank,
                    oppo_rank,
                    cur_rank,
                    player_down_cards,
                    player_opp_cards,
                    player_up_cards,
                    ), axis=0)



        # ppo_state = np.expand_dims(ppo_state, axis=0)

        # out_legal_actions = np.expand_dims(out_legal_actions, axis=0)
        # lstm_input = np.expand_dims(lstm_input, axis=0)
        # my_handcards_matrix = np.zeros((16, 6), dtype=int) # A 2 - K ，B R 共15张，四种花色
        # print(out_legal_actions.shape)
        # print(ppo_state.shape)
        # print(last_index_batch.shape)
        # print(my_action_batch.shape)
        # exit()

        # my_handcards_matrix[15][-1] = 1
        # for now_card in message['handCards']:
        #     suit = now_card[0] # 花色
        #     value = now_card[1] # 牌
        #     row, col = CardToNum_hand[suit][value]
        #     my_handcards_matrix[row, col] += 1

        # randcardsuit = 'H'
        # randcardvalue = message['curRank']
        # row1, col1 = CardToNum_hand[randcardsuit][randcardvalue]
        # # 更新通配牌
        # for i in range(13):
        #     my_handcards_matrix[i , 4] = my_handcards_matrix[row1, col1] 
        obs = {
            'position': player_id,
            'x_batch': x_batch.astype(np.int8),
            'x_no_action': x_no_action.astype(np.float32),
            'z': history_act,
            'z_batch': z_batch.astype(np.int8),
            'legal_actions': out_legal_actions.astype(np.int8),
            # 'ppo_state': ppo_state.astype(np.int8),
            'lstm_input':lstm_input.astype(np.int8),
            # 'myhand_matrix': my_handcards_matrix,
            # 'rankcard': (row1, col1),
            'last_index_batch': last_index_batch,
            'last_index_no_action_batch':last_index_no_action_batch,
        }
        return obs

    # 进贡  
    def tribute_act(self, actionList, rank):
        """根据合法动作集选择动作执行"""
        rank_card = 'H'+rank
        first_action = actionList[0]
        if rank_card in first_action[2]:
            act = actionList[1]
        else:
            act = actionList[0]
        self.execute_tribute(act)
        return act
    
    # 还贡
    def back_act(self, actionLists, rank, tribute_result):
        """根据合法动作集选择动作执行"""
        self.actionList = actionLists
        handCards = self.current_hand_str
        mypos = self.player_id
        card_val = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                    "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
        card_val[rank] = 15
        def flag_TJQ(handCards_X) -> tuple:
            flag_T = False
            flag_J = False
            flag_Q = False
            for i in range(len(handCards_X)):
                if handCards_X[i][0][-1] == "T":
                    flag_T = True
                if handCards_X[i][0][-1] == "J":
                    flag_J = True
                if handCards_X[i][0][-1] == "Q":
                    flag_Q = True
            return flag_T, flag_J, flag_Q

        def get_card_index(target: str) -> int:
            for i in range(len(self.actionList)):
                if self.actionList[i][2][0] == target:
                    return i

        def choose_in_single(single_list) -> str:
            for my_pos in tribute_result:
                if my_pos[1] == mypos:
                    tribute_pos = my_pos[0]

            n = len(single_list)
            if (int(tribute_pos) + int(mypos)) % 2 != 0:  
                for card in single_list:
                    if card in ['H5', 'HT']:  
                        return card
                    elif card in ['S5', 'C5', 'D5', 'ST', 'CT', 'DT']:
                        return card  
                
                return single_list[randint(0, n - 1)]
            else:  
                back_list = []
                for card in single_list:
                    if card[-1] != 'T':
                        if int(card[-1]) < 5:
                            back_list.append(card)  
                if back_list:
                    return back_list[randint(0, len(back_list) - 1)]
                return single_list[randint(0, n - 1)]

        def choose_in_pair(pair_list, pair_list_from_handcards) -> str:
            val_dict = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10}
            if len(pair_list) < 3:
                return pair_list[0][0]
            for i in range(len(pair_list)):
                flag = False
                if i >= 2:
                    pair_first_val, pair_second_val, pair_third_val = pair_list[i - 2][0][-1], pair_list[i - 1][0][-1], \
                                                                      pair_list[i][0][-1]
                    if val_dict[pair_first_val] == val_dict[pair_second_val] - 1 and val_dict[pair_second_val] == \
                            val_dict[pair_third_val] - 1:
                        flag = True
                if 1 <= i <= len(pair_list) - 2:
                    pair_first_val, pair_second_val, pair_third_val = pair_list[i - 1][0][-1], pair_list[i][0][-1], \
                                                                      pair_list[i + 1][0][-1]
                    if val_dict[pair_first_val] == val_dict[pair_second_val] - 1 and val_dict[pair_second_val] == \
                            val_dict[pair_third_val] - 1:
                        flag = True
                if i <= len(pair_list) - 3:
                    pair_first_val, pair_second_val, pair_third_val = pair_list[i][0][-1], pair_list[i + 1][0][-1], \
                                                                      pair_list[i + 2][0][-1]
                    if val_dict[pair_first_val] == val_dict[pair_second_val] - 1 and val_dict[pair_second_val] == \
                            val_dict[pair_third_val] - 1:
                        flag = True
                if pair_list[i][0][-1] == '9':
                    flag_T, flag_J, flag_Q = flag_TJQ(pair_list_from_handcards)
                    if flag_T and flag_J:
                        flag = True
                if pair_list[i][0][-1] == 'T':
                    flag_T, flag_J, flag_Q = flag_TJQ(pair_list_from_handcards)
                    if flag_J and flag_Q:
                        flag = True
                if flag:
                    continue
                else:
                    return pair_list[i][0]
            return pair_list[0][0]

        def choose_in_trips(trips_list, trips_list_from_handcards) -> str:
            val_dict = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10}
            if len(trips_list) < 2:
                return trips_list[0][0]
            for i in range(len(trips_list)):
                flag = False
                if i >= 1:
                    pair_first_val, pair_second_val = trips_list[i - 1][0][-1], trips_list[i][0][-1]
                    if val_dict[pair_first_val] == val_dict[pair_second_val] - 1:
                        flag = True
                if i <= len(trips_list) - 2:
                    pair_first_val, pair_second_val = trips_list[i][0][-1], trips_list[i + 1][0][-1]
                    if val_dict[pair_first_val] == val_dict[pair_second_val] - 1:
                        flag = True
                if trips_list[i][0][-1] == 'T':
                    flag_T, flag_J, flag_Q = flag_TJQ(trips_list_from_handcards)
                    if flag_J:
                        flag = True
                if flag:
                    continue
                else:
                    return trips_list[i][0]
            return trips_list[0][0]

        def choose_in_bomb(bomb_list, bomb_info) -> str:
            def get_card_from_bomb(bomb_list, key):
                for bomb in bomb_list:
                    for card in bomb:
                        if card[-1] == key:
                            return card

            for key, value in bomb_info:
                if value > 4:
                    return get_card_from_bomb(bomb_list, key)
            return bomb_list[0][0]

        combined_handcards, handCards_bomb_info = combine_handcards(handCards, rank, card_val)  

        combined_temp = {"Single": [], "Trips": [], "Pair": [], "Bomb": []}
        temp_bomb_info = {}
        for card in combined_handcards["Single"]:
            if card_val[card[-1]] <= 10:
                combined_temp["Single"].append(card)
        for trips_card in combined_handcards["Trips"]:
            if card_val[trips_card[0][-1]] <= 10:
                combined_temp["Trips"].append(trips_card)
        for pair_card in combined_handcards["Pair"]:
            if card_val[pair_card[0][-1]] <= 10:
                combined_temp["Pair"].append(pair_card)
        for bomb_card in combined_handcards["Bomb"]:
            if card_val[bomb_card[0][-1]] <= 10:
                combined_temp["Bomb"].append(bomb_card)
        for key, values in handCards_bomb_info.items():
            if card_val[key] <= 10:
                temp_bomb_info[key] = values

        card = None
        if combined_temp["Single"]:
            
            card = choose_in_single(combined_temp["Single"])
        elif combined_temp["Trips"]:
            
            card = choose_in_trips(combined_temp["Trips"], combined_handcards["Trips"])
        elif combined_temp["Pair"]:
            
            card = choose_in_pair(combined_temp["Pair"], combined_handcards["Pair"])
        # elif combined_temp["Bomb"]:
            
        #     card = choose_in_bomb(combined_temp["Bomb"], temp_bomb_info)
        else:
            
            temp = []  
            for handCard in handCards:
                if card_val[handCard[-1]] <= 10:
                    temp.append(handCard)
            card = temp[randint(0, len(temp) - 1)]
            
        act_index = get_card_index(card)
        act = actionLists[act_index]
        self.execute_back(act)
        return act


    def get_action_index_list(self , acts , state):

        if acts[0].item() == 135:
            return [(0 , 0)]
        
        action1 = reverse_actions_dict[acts[0].item() // 15]
        action2 = reverse_card_value_idx[acts[0].item() % 15]
        # print(action1 , action2)
        legal_actions_all = []
        for idx, action in enumerate(state['actionList']):
            # 对比当前action[0] 与 action1
            # action = ['Straight', '', ['H2', 'C2', 'H2', 'S4', 'S5']]
            # print(action1)
            if action[0] != action1:
                continue
            if action[1] != action2:
                continue
            legal_actions_all.append((idx, action))
        return legal_actions_all





    def get_action_index(self , message ,  acts, state):
        # 获取对应的动作
        if acts == 135:
            return 0
        # acts = 111
        action1 = reverse_actions_dict[acts // 15]
        action2 = reverse_card_value_idx[acts % 15]
        # print(acts)
        # print(action1 , action2)
        # print(action2)
        # 当前手牌
        # my_handcards = card2array(card2num(message['handCards'])) 
        my_handcards = state['myhand_matrix']
        sf_matrix = self.scan_sf(my_handcards , state['rankcard'][0] , state['rankcard'][1])
        # print(sf_matrix)
        # print(state['myhand_matrix'])
        # print(sf_matrix*state['myhand_matrix'])

        # 获得符合上述action1，action2的卡牌
        # 输出符合下述条件的卡牌
        randcard = 'H'+ message['curRank']
        legal_actions_all = []
        legal_actions_no_rank_sf = []
        legal_actions_no_rank_card = []
        legal_actions_no_sf = []

        for idx, action in enumerate(message['actionList']):
            # 对比当前action[0] 与 action1
            # action = ['Straight', '', ['H2', 'C2', 'H2', 'S4', 'S5']]
            # print(action1)
            if action[0] != action1:
                continue
            if action[1] == 'JOKER' and acts % 15 == 13:
                # 王炸的特殊处理
                legal_actions_all.append((idx, action))
            if action[1] == '':
                for cidx, card in enumerate(action[2]):
                    if card != randcard:
                        nowid = RANK2[card[-1]]
                        rankid = nowid - cidx
                        rank_type = reverse_rank2_value_idx[rankid]
                        break
                tempaction = [action[0] , rank_type , action[2]]
                # print(tempaction)
                # exit()
                if rank_type == action2:
                    legal_actions_all.append((idx, action))
                continue

            if action[1] != action2:
                continue
            # 对比当前action[1] 与 action2



            # 将符合要求的存入 
            legal_actions_all.append((idx, action))
            check_rank = False
            check_sf = False
            for cidx, card in enumerate(action[2]):
                suit = card[0]
                value = card[1] # 牌
                row, col = CardToNum_hand[suit][value]
                if card == randcard:
                    check_rank = True
                if sf_matrix[row , col] > 0:
                    check_sf = True


            # 先检查包含级牌吗：
            if not check_rank and not check_sf:
                # print(action)
                return idx
                # 符合抽象规则的所有要求，直接返回
                legal_actions_no_rank_sf.append((idx, action))
                legal_actions_no_rank_card.append((idx, action))
                legal_actions_no_sf.append((idx, action))

                continue
            if not check_rank:
                legal_actions_no_rank_card.append((idx, action))

            if not check_sf:
                legal_actions_no_sf.append((idx, action))

        min_idx = -1
        min_action_sum = 10
        min_action = None
        for idx, action in legal_actions_no_rank_card:
            # 获取动作的卡牌集合
            # action = ['Straight', '', ['H2', 'C2', 'H2', 'S4', 'S5']]
            cards = action[2]
            card_out =  np.zeros((16, 6), dtype=int)
            for card in cards:
                suit = card[0]
                value = card[1] # 牌
                row, col = CardToNum_hand[suit][value]
                card_out[row, col] = 1
            action_sum = np.sum(sf_matrix*card_out) 
            # print(action_sum)
            if action_sum < min_action_sum:
                min_action_sum = action_sum
                min_idx = idx
                min_action = action
            if action_sum == 0:
                # print(action)
                return idx
        # 使用 card2num(cards) 转为对应位置list[...]
        # 检查list中的牌组，选出第一个全部牌在sf_matrix中不存在的
        # 如果没有找到，则输出legal_actions的第一个动作

            # 如果没有找到符合条件的合法动作，则返回legal_actions的第一个动作
        if min_idx != -1:
            # print(action)
            return min_idx


        #  如果必须用到级牌了，则考虑在使用级牌的情况下，破坏同花顺最少的牌
        min_idx = -1
        min_action_sum = 10
        min_action = None
        for idx, action in legal_actions_no_sf:
            # 获取动作的卡牌集合
            # action = ['Straight', '', ['H2', 'C2', 'H2', 'S4', 'S5']]
            cards = action[2]
            card_out =  np.zeros((16, 6), dtype=int)
            for card in cards:
                suit = card[0]
                value = card[1] # 牌
                row, col = CardToNum_hand[suit][value]
                card_out[row, col] = 1
            action_sum = np.sum(sf_matrix*card_out) 
            # print(action_sum)
            if action_sum < min_action_sum:
                min_action_sum = action_sum
                min_idx = idx
                min_action = action
            if action_sum == 0:
                # print(action)
                return idx
        if min_idx != -1:
            # print(action)
            return min_idx          

        #  如果必须用到级牌了，则考虑在使用级牌的情况下，破坏同花顺最少的牌
        min_idx = -1
        min_action_sum = 10000
        min_action = None
        # 又要使用级牌，又要破坏同花顺，则在所有动作中选一个损失最小的
        sf_matrix[state['rankcard'][0] , state['rankcard'][1]] = 1
        # 把级牌加进去一起考虑
        for idx, action in legal_actions_all:
            # 获取动作的卡牌集合
            # action = ['Straight', '', ['H2', 'C2', 'H2', 'S4', 'S5']]
            cards = action[2]
            card_out =  np.zeros((16, 6), dtype=int)
            for card in cards:
                suit = card[0]
                value = card[1] # 牌
                row, col = CardToNum_hand[suit][value]
                card_out[row, col] = 1
            action_sum = np.sum(sf_matrix*card_out) 
            # print(action_sum)
            if action_sum < min_action_sum:
                min_action_sum = action_sum
                min_idx = idx
                min_action = action
            if action_sum == 0:
                # print(action)
                return idx
        if min_idx != -1:
            # print(action)
            return min_idx   
        # print(legal_actions)
        # print(message['actionList'])
        # print("pass")
        if(len(legal_actions_all) > 0):
            idx , action = legal_actions_all[0]
            return idx
        print('error')
        print(legal_actions_all)
        print(message['actionList'])
        print(acts)
        print(message['handCards'])
        exit()

        return 0
    def scan_sf(self, myhand_cards , row , col):
        # 首先去除级牌 
        rank_num = copy.deepcopy(myhand_cards[row][col])
        # myhand_cards 的形式为行（size = 4 ， 代表花色）列（size=13, 代表卡牌级数 0-2 ， 9-J ， 10-Q, 11-K , 12 -A）
        myhand_cards[row][col] = 0
        sf_member =  np.zeros((16, 6), dtype=int)
        # 检查是否有同花顺 注意这里需要检查A-5 即 对13取余，但是没有K-4
        # 可以检查连续5位同花色卡牌是否能组成同花顺，此外可以使用级牌作为万能拍
        # 先检查非A情况
        for start in range(9):
            for j in range(4):
                required_cards = [myhand_cards[(start + i) % 13][j] > 0 for i in range(5)]
                missing_count = required_cards.count(False)
                if missing_count <= rank_num:
                    for i in range(5):
                       sf_member[(start + i) % 13][j] =  myhand_cards[(start + i) % 13][j]
        # 检查A的情况
        start = 12
        for j in range(4):
            required_cards = [myhand_cards[(start + i) % 13][j] > 0 for i in range(5)]
            missing_count = required_cards.count(False)
            if missing_count <= rank_num:
                for i in range(5):
                    sf_member[(start + i) % 13][j] =  myhand_cards[(start + i) % 13][j]      


        
        # 返回所有能组成同花顺的成员
        return sf_member

    def dan_prepare(self, message):
        # print(self.rank)
        num_legal_actions = message['indexRange'] + 1
        # print('num_legals: ', num_legal_actions)
        legal_actions = [card2num(i[2]) for i in message['actionList']]

        my_handcards = card2array(card2num(message['handCards']))   # 自己的手牌,54维
        # self.logger.info(f"handcards: {message['handCards']}")

        my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

        universal_card_flag = self.proc_universal(my_handcards, RANK[message['curRank']])     # 万能牌的标志位, 12维

        universal_card_flag_batch = np.repeat(universal_card_flag[np.newaxis, :],
                                   num_legal_actions, axis=0)

        count_a = np.array([self.count_A])

        count_a_self = np.array([self.count_A_self])

        count_a_oppo = np.array([self.count_A_oppo])

        other_hands = []       # 其余所有玩家手上剩余的牌，54维
        for i in range(54): 
            if self.other_left_hands[i] == 1:
                other_hands.append(i)
            elif self.other_left_hands[i] == 2:
                other_hands.append(i)
                other_hands.append(i)
        
        other_handcards = card2array(other_hands)      
        
        other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

        last_action = []         # 最新的动作，54维
        if len(self.action_seq) > 0: 
            if(self.action_seq[-1] == [-1]):
                last_action = card2array([])
            else:
                last_action = card2array(self.action_seq[-1])
        else:
            # 刚开局的情况
            last_action = card2array([-1])
        
        last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)
        
        # print(last_action_batch)
        
        last_teammate_action = []               # 队友最后的动作， 54维
        if len(self.history_action[(self.mypos + 2) % 4]) > 0 and (self.mypos + 2) % 4 not in self.over:
            last_teammate_action = card2array(self.history_action[(self.mypos + 2) % 4][-1])
        else:
            last_teammate_action = card2array([-1])
        
        last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)

        my_action_batch = np.zeros(my_handcards_batch.shape)     # 合法动作，54维
        for j, action in enumerate(legal_actions):
            my_action_batch[j, :] = card2array(action)

        down_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 1) % 4], 27, 1)   # 下家剩余的牌数， 28维
        
        down_num_cards_left_batch = np.repeat(down_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        teammate_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 2) % 4], 27, 1)   # 对家剩余的牌数

        teammate_num_cards_left_batch = np.repeat(teammate_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        up_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 3) % 4], 27, 1)   # 上家剩余的牌数
        
        up_num_cards_left_batch = np.repeat(up_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 1) % 4]) > 0:
            down_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 1) % 4]))    # 下家打过的牌， 54维
        else:
            down_played_cards = card2array([])
        
        down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 2) % 4]) > 0:
            teammate_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 2) % 4]))    # 对家打过的牌
        else:
            teammate_played_cards = card2array([])

        teammate_played_cards_batch = np.repeat(teammate_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 3) % 4]) > 0:
            up_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 3) % 4]))    # 上家打过的牌
        else:
            up_played_cards = card2array([])

        up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], num_legal_actions, axis=0)
 
        
        ori_self_rank = message['rank_list'][self.mypos % 2]
        # print(ori_self_rank)
        # self_rank = _get_one_hot_array(RANK[ori_self_rank], 13, 0)         # 己方当前的级牌，13维
        self_rank = _get_one_hot_array(ori_self_rank+1, 13, 0)         # 己方当前的级牌，13维
        self.rank = ori_self_rank+1

        self_rank_batch = np.repeat(self_rank[np.newaxis, :], num_legal_actions, axis=0)

        ori_oppo_rank = message['rank_list'][(self.mypos+1)%2]
        # oppo_rank = _get_one_hot_array(RANK[ori_oppo_rank], 13, 0)         # 敌方当前的级牌
        oppo_rank = _get_one_hot_array(ori_oppo_rank+1, 13, 0)         # 敌方当前的级牌
        self.oppo_rank = ori_oppo_rank+1

        oppo_rank_batch = np.repeat(oppo_rank[np.newaxis, :], num_legal_actions, axis=0)

        cur_rank = _get_one_hot_array(RANK[message['curRank']], 13, 0)         # 当前的级牌

        cur_rank_batch = np.repeat(cur_rank[np.newaxis, :], num_legal_actions, axis=0)

        x_batch = np.hstack((my_handcards_batch,
                        universal_card_flag_batch,
                        other_handcards_batch,
                        last_action_batch,
                        last_teammate_action_batch,
                        down_played_cards_batch,
                        teammate_played_cards_batch,
                        up_played_cards_batch,
                        down_num_cards_left_batch,
                        teammate_num_cards_left_batch,
                        up_num_cards_left_batch,
                        self_rank_batch,
                        oppo_rank_batch,
                        cur_rank_batch,
                        my_action_batch))
        x_no_action = np.hstack((my_handcards,
                            universal_card_flag,
                            other_handcards,
                            last_action,
                            last_teammate_action,
                            down_played_cards,
                            teammate_played_cards,
                            up_played_cards,
                            down_num_cards_left,
                            teammate_num_cards_left,
                            up_num_cards_left,
                            self_rank,
                            oppo_rank,
                            cur_rank,
                            count_a,
                            count_a_self,
                            count_a_oppo
                            ))

        obs = {
            'x_batch': x_batch.astype(np.int8),
            'x_no_action': x_no_action.astype(np.float32),
        }

        return obs

    def proc_universal(self, handCards, cur_rank):
        res = np.zeros(12, dtype=np.int8)

        if handCards[(cur_rank-1)*4] == 0: # 手牌中不包含通配牌，返回全零数组
            return res

        res[0] = 1 # 存在通配牌
        rock_flag = 0
        for i in range(4):
            # 遍历每个花色
            left, right = 0, 5
            temp = [handCards[i + j*4] if i+j*4 != (cur_rank-1)*4 else 0 for j in range(5)]
            while right <= 12:
                zero_num = temp.count(0)
                if zero_num <= 1:
                    rock_flag = 1
                    break
                else:
                    temp.append(handCards[i + right*4] if i+right*4 != (cur_rank-1)*4 else 0)
                    temp.pop(0)
                    left += 1
                    right += 1
            if rock_flag == 1:
                break
        res[1] = rock_flag

        num_count = [0] * 13
        for i in range(4):
            for j in range(13):
                if handCards[i + j*4] != 0 and i + j*4 != (cur_rank-1)*4:
                    num_count[j] += 1
        num_max = max(num_count)
        if num_max >= 6:
            res[2:8] = 1
        elif num_max == 5:
            res[3:8] = 1
        elif num_max == 4:
            res[4:8] = 1
        elif num_max == 3:
            res[5:8] = 1
        elif num_max == 2:
            res[6:8] = 1
        else:
            res[7] = 1
        temp = 0
        for i in range(13):
            if num_count[i] != 0:
                temp += 1
                if i >= 1:
                    if num_count[i] == 2 and num_count[i-1] >= 3 or num_count[i] >= 3 and num_count[i-1] == 2:
                        res[9] = 1
                    elif num_count[i] == 2 and num_count[i-1] == 2:
                        res[11] = 1
                if i >= 2:
                    if num_count[i-2] == 1 and num_count[i-1] >= 2 and num_count[i] >= 2 or \
                        num_count[i-2] >= 2 and num_count[i-1] == 1 and num_count[i] >= 2 or \
                        num_count[i-2] >= 2 and num_count[i-1] >= 2 and num_count[i] == 1:
                        res[10] = 1
            else:
                temp = 0
        if temp >= 4:
            res[8] = 1
        return res




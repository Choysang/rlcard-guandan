def _debug_print(*args, **kwargs):
    # Debug output from the original research code, silenced for
    # the open-source release.
    pass


from functools import reduce
from random import randint
from .danutil import card2array, card2num, combine_handcards
from guandan_rlcard.game.player import GuandanPlayer as Player
from .actor import Action

import numpy as np
import copy


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



CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
    'A', 'BJ', 'RJ']

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

class DanzeroAgent(Player):
    
    name= 'Danzero'
    
    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)
        self.use_raw = True
        self.begin = True
        self.mypos = player_id
        self.history_action = {0: [], 1: [], 2: [], 3:[]}
        self.new_histor_action = {0: [], 1: [], 2: [], 3:[]}
        self.action_order = []
        self.action_seq = []
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.other_left_hands = [2 for _ in range(54)]
        self.flag = 0
        self.over = []
        self.tongji = {3: 0, 2: 0, 1: 0, -1: 0, -2: 0, -3: 0, 'all': 0}
        self.max_acion = 5000
        self.rank = 0
        self.oppo_rank = 0
        self.playing_self = -1
        self.count_A = 0
        self.count_A_self = 0
        self.count_A_oppo = 0
        self.action = Action()

    def step(self, state):
        if not state['actions']:
            return []
        msg = self.parse(state)
        
        act_index = self.received_message(msg)
        
        final_action = state['actions'][act_index]
    
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

        # check_list = []
        # for t in state["trace"]:
        #     check_list.append(t[1])
        # if state['actions_seq'] != check_list:
        #     _debug_print(state['actions_seq'])
        #     _debug_print(check_list)
            # _debug_print("omg")
        # # 这里可以使用trace代替check_list
        # # 添加actions_seq
        # msg['actions_seq'] = []
        # for action in state['actions_seq']:
        #     num_action = card2num(action[2])
        #     if(num_action == [-1]):
        #         msg['actions_seq'].append([])
        #     else:
        #         msg['actions_seq'].append(num_action)
        # check_list = []
        # for t in state["trace"]:
        #     check_list.append(t[1])        
        # msg['actions_seq'] = []
        # for action in check_list:
        #     num_action = card2num(action[2])
        #     if(num_action == [-1]):
        #         msg['actions_seq'].append([])
        #     else:
        #         msg['actions_seq'].append(num_action)

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
        self.history_action = {0: [], 1: [], 2: [], 3:[]}
        self.new_histor_action = {0: [], 1: [], 2: [], 3:[]}
        self.action_order = []
        self.action_seq = []
        self.other_left_hands = [2 for _ in range(54)]
        self.action_order = []
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.over = []
        msg = {'order':result, 'curRank':CARD_RANK[self.rank-1]}
        # reward = self.get_reward(msg)
        # _debug_print(reward)
        # self.logger.info(f'reward:{reward}')
        if self.playing_self == -1:
            self.count_A = 0
            self.count_A_self = 0
            self.count_A_oppo = 0
        # self.tongji[reward] += 1
        # self.tongji['all'] += 1
        self.oppo_list = []



    def received_message(self, message):
        # 先序列化收到的消息，转为Python中的字典
        if self.begin:
            self.begin = False
            init_hand = card2num(message['handCards'])
            for ele in init_hand:
                self.other_left_hands[ele] -= 1
            self.oppo_list = []
            for i in range(1, 4):
                self.oppo_list.append((self.player_id + i) % 4)
            self.count_A += int(message['rank_list'][message['play_team']] == 12)
            self.count_A_self += int(message['rank_list'][self.mypos%2] == 12)
            self.count_A_oppo += int(message['rank_list'][(self.mypos+1)%2] == 12)
            
        left_players = 3 - len(self.over)
        

        # _debug_print("orgin trace : " , message["trace"])


        # 更新对对手的建模
        if len(message["trace"]) >= left_players:
            trace = message["trace"][-left_players:] # 其他玩家的最近动作记录
        else:
            trace = message["trace"]
        # 理论上trace中只包含最近其他玩家的各一个动作
        # _debug_print("trace:" , trace)
        # over 的统计在 trace 之后，因为完牌玩家的最后一个动作也需要记录
        for act_pair in trace:
            # _debug_print(act_pair)
            # 记录打完牌的玩家
            if message['history'][str(act_pair[0])]['remain'] == 0:
                self.over.append(act_pair[0])

        if len(self.over) == 2:
            if (self.over[0] + self.over[1]) % 2 == 0:
                # 对家双上，此时只需随机出牌
                return randint(0, message['indexRange'])
        
        # 检查是否存在接风，如果是的话，trace包括自己和一个其他玩家
        if len(self.over) == 1 and trace[0][0] == self.player_id:
            trace = [trace[-1]]
        elif len(self.over) == 2 and trace[0][0] == self.player_id:
            trace = [message["trace"][-2]]
        
        if trace: # 如果自己是第一个出牌则无需记录
            # _debug_print(message['trace'])
            # _debug_print('self: ', self.player_id, '  trace: ', trace)
            for act_pair in trace:
                # act_pair 示例 [2, ['PASS', 'PASS', 'PASS]]，玩家id+动作的列表
                
                # 按顺序确定每个其他玩家的动作
                just_action = act_pair[1]
                if just_action[0] == 'PASS':
                    just_action[2] = -1
                action = card2num(just_action[2]) # action为数字列表
                
                if action != [-1]:
                    # 减去此轮对手的动作出牌，得到牌库剩余牌
                    for ele in action:
                        self.other_left_hands[ele] -= 1
            
                # 本实现默认notify不会通报自己的动作(是否有问题？)       
                if len(self.over) == 0:    # 如果没人出完牌，默默记录玩家动作
                    self.action_order.append(act_pair[0])
                    self.action_seq.append(action)
                    self.history_action[act_pair[0]].append(action)
                elif len(self.over) == 1:
                    # _debug_print('over: ', self.over)
                    # _debug_print('oppo_list: ', self.oppo_list)
                    # _debug_print(act_pair)
                    out_player = self.over[0]
                    # 如果是完牌者是当前玩家的上家
                    if self.oppo_list.index(out_player) - self.oppo_list.index(act_pair[0]) == 1:
                        self.action_order.append(out_player)
                        self.action_seq.append([-1])
                        self.history_action[out_player].append([-1])
                        
                    self.action_order.append(act_pair[0])
                    self.action_seq.append(action)
                    self.history_action[act_pair[0]].append(action)
                elif len(self.over) == 2:
                    # _debug_print('over: ', self.over)
                    # 注意，由于之前处理了对家双上的情况，因此，此时只剩下自己和一个对手的对决，而出完牌的两家一定是连着的
                    # 这两家要么在当前记录玩家之前，要么在之后
                    if self.oppo_list.index(self.over[0]) > self.oppo_list.index(act_pair[0]):
                        # 两个出完牌的玩家顺序在当前记录玩家之后，先记录当前玩家，然后在记录中插入两个-1
                        self.action_order.append(act_pair[0])
                        self.action_seq.append([action])
                        self.history_action[act_pair[0]].append(action)
                        
                        self.action_order.append((act_pair[0]+1)%4)
                        self.action_seq.append([-1])
                        self.history_action[(act_pair[0]+1)%4].append([-1])
                        self.action_order.append((act_pair[0]+2)%4)
                        self.action_seq.append([-1])
                        self.history_action[(act_pair[0]+2)%4].append([-1])
                    else:
                        # 两个出完牌的玩家顺序在当前记录玩家之前，先插入两个-1，然后在记录当前玩家
                        self.action_order.append((act_pair[0]+2)%4)
                        self.action_seq.append([-1])
                        self.history_action[(act_pair[0]+2)%4].append([-1])
                        self.action_order.append((act_pair[0]+3)%4)
                        self.action_seq.append([-1])
                        self.history_action[(act_pair[0]+3)%4].append([-1])
                        
                        self.action_order.append(act_pair[0])
                        self.action_seq.append(action)
                        self.history_action[act_pair[0]].append(action)
                else:
                    _debug_print('self.over >= 3, random play!')
                    return randint(0, message['indexRange'])

        self.remaining = message['num_cards_left']
        # _debug_print('remaining: ', self.remaining)
        # _debug_print(self.action_order)
        # _debug_print(self.action_seq)
        # _debug_print(self.history_action, end="\n")

        # 准备状态数据
        if len(message['actionList']) == 1:
            return 0
        else :
            state = self.prepare(message)
            # 将state传给模型做决策，返回index
            index = self.action.step(state)
            # _debug_print(index)
            return index

    def get_reward(self, message):
        team = [self.mypos, (self.mypos + 2) % 4]
        order = message['order']
        rewards = {"1100": 3, "1010": 2, "1001": 1, "0110": -1, "0101": -2, "0011": -3}
        res = ""
        for i in order:
            if i in team:
                res += '1'
            else:
                res += '0'
        
        if res not in rewards.keys():
            return 0.0
        
        return rewards[res]

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

    def prepare(self, message):
        # _debug_print(self.rank)
        num_legal_actions = message['indexRange'] + 1
        # _debug_print('num_legals: ', num_legal_actions)
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
        
        # _debug_print(last_action_batch)
        
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
        # _debug_print(ori_self_rank)
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
        elif combined_temp["Bomb"]:
            
            card = choose_in_bomb(combined_temp["Bomb"], temp_bomb_info)
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


    def get_message(self, message):
        # 先序列化收到的消息，转为Python中的字典
        if self.begin:
            self.begin = False
            init_hand = card2num(message['handCards'])
            for ele in init_hand:
                self.other_left_hands[ele] -= 1
            self.oppo_list = []
            for i in range(1, 4):
                self.oppo_list.append((self.player_id + i) % 4)
            self.count_A += int(message['rank_list'][message['play_team']] == 12)
            self.count_A_self += int(message['rank_list'][self.mypos%2] == 12)
            self.count_A_oppo += int(message['rank_list'][(self.mypos+1)%2] == 12)
            
        left_players = 3 - len(self.over)
        

        # _debug_print("orgin trace : " , message["trace"])


        # 更新对对手的建模
        if len(message["trace"]) >= left_players:
            trace = message["trace"][-left_players:] # 其他玩家的最近动作记录
        else:
            trace = message["trace"]
        # 理论上trace中只包含最近其他玩家的各一个动作
        # _debug_print("trace:" , trace)
        # over 的统计在 trace 之后，因为完牌玩家的最后一个动作也需要记录
        for act_pair in trace:
            # _debug_print(act_pair)
            # 记录打完牌的玩家
            if message['history'][str(act_pair[0])]['remain'] == 0:
                self.over.append(act_pair[0])

        if len(self.over) == 2:
            if (self.over[0] + self.over[1]) % 2 == 0:
                # 对家双上，此时只需随机出牌
                return randint(0, message['indexRange'])
        
        # 检查是否存在接风，如果是的话，trace包括自己和一个其他玩家
        if len(self.over) == 1 and trace[0][0] == self.player_id:
            trace = [trace[-1]]
        elif len(self.over) == 2 and trace[0][0] == self.player_id:
            trace = [message["trace"][-2]]
        
        if trace: # 如果自己是第一个出牌则无需记录
            # _debug_print(message['trace'])
            # _debug_print('self: ', self.player_id, '  trace: ', trace)
            for act_pair in trace:
                # act_pair 示例 [2, ['PASS', 'PASS', 'PASS]]，玩家id+动作的列表
                
                # 按顺序确定每个其他玩家的动作
                just_action = act_pair[1]
                if just_action[0] == 'PASS':
                    just_action[2] = -1
                action = card2num(just_action[2]) # action为数字列表
                
                if action != [-1]:
                    # 减去此轮对手的动作出牌，得到牌库剩余牌
                    for ele in action:
                        self.other_left_hands[ele] -= 1
            
                # 本实现默认notify不会通报自己的动作(是否有问题？)       
                if len(self.over) == 0:    # 如果没人出完牌，默默记录玩家动作
                    self.action_order.append(act_pair[0])
                    self.action_seq.append(action)
                    self.history_action[act_pair[0]].append(action)
                elif len(self.over) == 1:
                    # _debug_print('over: ', self.over)
                    # _debug_print('oppo_list: ', self.oppo_list)
                    # _debug_print(act_pair)
                    out_player = self.over[0]
                    # 如果是完牌者是当前玩家的上家
                    if self.oppo_list.index(out_player) - self.oppo_list.index(act_pair[0]) == 1:
                        self.action_order.append(out_player)
                        self.action_seq.append([-1])
                        self.history_action[out_player].append([-1])
                        
                    self.action_order.append(act_pair[0])
                    self.action_seq.append(action)
                    self.history_action[act_pair[0]].append(action)
                elif len(self.over) == 2:
                    # _debug_print('over: ', self.over)
                    # 注意，由于之前处理了对家双上的情况，因此，此时只剩下自己和一个对手的对决，而出完牌的两家一定是连着的
                    # 这两家要么在当前记录玩家之前，要么在之后
                    if self.oppo_list.index(self.over[0]) > self.oppo_list.index(act_pair[0]):
                        # 两个出完牌的玩家顺序在当前记录玩家之后，先记录当前玩家，然后在记录中插入两个-1
                        self.action_order.append(act_pair[0])
                        self.action_seq.append([action])
                        self.history_action[act_pair[0]].append(action)
                        
                        self.action_order.append((act_pair[0]+1)%4)
                        self.action_seq.append([-1])
                        self.history_action[(act_pair[0]+1)%4].append([-1])
                        self.action_order.append((act_pair[0]+2)%4)
                        self.action_seq.append([-1])
                        self.history_action[(act_pair[0]+2)%4].append([-1])
                    else:
                        # 两个出完牌的玩家顺序在当前记录玩家之前，先插入两个-1，然后在记录当前玩家
                        self.action_order.append((act_pair[0]+2)%4)
                        self.action_seq.append([-1])
                        self.history_action[(act_pair[0]+2)%4].append([-1])
                        self.action_order.append((act_pair[0]+3)%4)
                        self.action_seq.append([-1])
                        self.history_action[(act_pair[0]+3)%4].append([-1])
                        
                        self.action_order.append(act_pair[0])
                        self.action_seq.append(action)
                        self.history_action[act_pair[0]].append(action)
                else:
                    _debug_print('self.over >= 3, random play!')
                    return randint(0, message['indexRange'])

        self.remaining = message['num_cards_left']
        # _debug_print('remaining: ', self.remaining)
        # _debug_print(self.action_order)
        # _debug_print(self.action_seq)
        # _debug_print(self.history_action, end="\n")
        return message
        # # 准备状态数据
        # if len(message['actionList']) == 1:
        #     return 0
        # else :
        #     state = self.prepare(message)
        #     # 将state传给模型做决策，返回index
        #     index = self.action.step(state)
        #     # _debug_print(index)
        #     return index
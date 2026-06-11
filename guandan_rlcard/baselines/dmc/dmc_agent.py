from random import randint

import torch
import numpy as np
from guandan_rlcard.game.player import GuandanPlayer as Player
from .models import PlayerModel
from .env_utils import card2array, card2num, action_vector, combine_handcards

RANK = {
    '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8,
    'T':9, 'J':10, 'Q':11, 'K':12, 'A':13
}

CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
    'A', 'BJ', 'RJ']


def _get_one_hot_array(num_left_cards, max_num_cards, flag):
    if flag == 0:     # 级数的情况
        one_hot = np.zeros(max_num_cards)
        one_hot[num_left_cards - 1] = 1
    else:
        one_hot = np.zeros(max_num_cards+1)    # 剩余的牌（0-1阵格式）
        one_hot[num_left_cards] = 1
    return one_hot

class DMCAgent(Player):
    def __init__(self, player_id, np_random, model=None, device='cpu'):
        super().__init__(player_id, np_random)
        self.name = 'dmc_agent'
        self.use_raw = True
        self.epsilon = 0.2
        if not device == "cpu":
            self.device = 'cuda:' + str(device)
        else:
            self.device = device
        if model:
            self.model = model
            self.device = next(model.parameters()).device
        else:
            # FIX: allow standalone construction; the agent then plays
            # with untrained (random) weights until a model is provided.
            self.model = PlayerModel().to(self.device)

    def step(self, state, model=None, training=False):
        if not state['actions']:
            return []
        msg = self.parse(state)
        obs = self.state_to_obs(msg, self.player_id)
        index = 0
        
        # 准备状态数据
        if len(msg['actionList']) != 1:
            # 将state传给模型做决策，返回index
            z_batch = torch.from_numpy(obs['z_batch']).float().to(self.device)
            x_batch = torch.from_numpy(obs['x_batch']).float().to(self.device)
            
            # 使用模型进行rollout，决策出一个动作
            with torch.no_grad():
                if training:
                    # 此 model 需要选择位置
                    output = model.forward(str(self.player_id), z_batch, x_batch, training=True)
                    values = output['values']
                    # 选择价值最高的动作 （greedy）
                    # index = torch.argmax(values).item()
                    # epsilon-greedy 策略
                    if np.random.rand() < self.epsilon:
                        index = torch.randint(values.shape[0], (1,))[0]
                    else:
                        index = torch.argmax(values).item()
                else:
                    output = self.model.forward(z_batch, x_batch, return_value=False)
                    index = output['action']
    
        final_action = state['actions'][index]
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

    def state_to_obs(self, message, player_id):
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
        
        # print(x_batch.shape)
        # print(x_no_action.shape)
        # print(z_batch.shape)
        
        obs = {
            'position': player_id,
            'x_batch': x_batch.astype(np.int8),
            'x_no_action': x_no_action.astype(np.float32),
            'z': history_act,
            'z_batch': z_batch.astype(np.int8),
            'legal_actions': my_action_batch.astype(np.int8) 
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

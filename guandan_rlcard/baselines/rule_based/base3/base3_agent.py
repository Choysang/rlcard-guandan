# -*- coding: utf-8 -*-
def _debug_print(*args, **kwargs):
    # Debug output from the original research code, silenced for
    # the open-source release.
    pass


# @Time       : 2020/10/1 21:32
# @Author     : Duofeng Wu
# @File       : action.py
# @Description: 动作类
# 版本号：INDEX OS2.0.0

import random
from random import randint
from .message_Reyn_CUR import check_message, get_num, get_point_val, get_remain_VAL
from guandan_rlcard.game.player import GuandanPlayer as Player

# 中英文对照表
ENG2CH = {
    "Single": "单张",
    "Pair": "对子",
    "Trips": "三张",
    "ThreePair": "三连对",
    "ThreeWithTwo": "三带二",
    "TwoTrips": "钢板",
    "Straight": "顺子",
    "StraightFlush": "同花顺",
    "Bomb": "炸弹",
    "PASS": "过"
}
CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
    'A', 'BJ', 'RJ']

class Action(object):

    def __init__(self):
        self.action = []
        self.act_range = -1
        self.AI_choice = -1

    #该为完全随机数的行动
    def parse(self, msg):
        self.action = msg["actionList"]
        self.act_range = msg["indexRange"]
        # _debug_print(self.action)
        # _debug_print("可选动作范围为：0至{}".format(self.act_range))
        return randint(0, self.act_range)

    #该为有AI加持的确定行动
    def parse_AI(self, msg, pos):
        self.action = msg["actionList"]
        self.act_range = msg["indexRange"]
        # _debug_print(self.action)
        #运行AI来确定需要出的牌
        self.AI_choice = check_message(msg,pos)

        #由于没有考虑进贡，故而随机，否则bug
        if self.AI_choice == None:
            return randint(0, self.act_range)
        # _debug_print("AI选择的出牌编号为:{}".format(self.AI_choice))
        return self.AI_choice
    
class Base3Agent(Player):
    ''' Baseline 3 agent.
    '''
    name = 'Base3'
    
    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)
        self.action = Action()
        self.use_raw = True
        self.begin = True
    
    def step(self, state):
        if not state['actions']:
            return []
        msg = self.parse(state)
        act_index = self.action.parse_AI(msg, self.player_id)

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
        msg['curRank'] = CARD_RANK[rank_list[play_team]]
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
        for i in range(4):
            msg['publicInfo'][i]['rest'] = state['num_cards_left'][i]
            
        return msg
    
    def tribute_act(self, actionLists, rank):
        """根据合法动作集选择动作执行"""
        act = random.choice(actionLists)
        self.execute_tribute(act)
        return act
    
    def back_act(self, actionLists, rank, tribute_result):
        """根据合法动作集选择动作执行"""
        curRank = rank
        #开始遍历Actionlist
        index = -1
        #创建空操作权重列表[权重值]
        value = []
        for check_card in actionLists:
            index += 1
            # _debug_print('Reyn_AI Tip 当前判断Action',index,':',check_card)
            #读取手牌部分开始
            # _debug_print('Reyn_AI Log 进入到读取手牌部分')
            #创建按花色和顺序归类好的牌库，数值为该牌拥有的数量
            #黑桃:S 红桃:H 梅花:C 方片:D
            #花色牌列表[数量] 序号-点数 分别为:[0-A,1-2,2-3,3-4,4-5,5-6,6-7,7-8,8-9,9-T,10-J,11-Q,12-K]
            handCards_S = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            handCards_H = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            handCards_C = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            handCards_D = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            #数值总列表[数量] 序号-点数 分别为:[0-A,1-2,2-3,3-4,4-5,5-6,6-7,7-8,8-9,9-T,10-J,11-Q,12-K]
            handCards_A = [0,0,0,0,0,0,0,0,0,0,0,0,0]
            #级牌列表[点数-级牌数量-红桃级牌数量]
            handCards_R = [get_num(curRank)+1,0,0]
            #特殊牌列表[数量] 序号-点数 分别为:[0-B,1-R]
            handCards_K = [0,0]
            #遍历服务器所给的'handCards'，为上述数据赋值
            for card in self.current_hand_str:
                if card[1] == curRank:
                    handCards_R[1] += 1
                    if card[0] == 'H':
                        handCards_R[2] += 1

                if card[1] == 'B' or card[1] == 'R':
                    if card[1] == 'B':
                        handCards_K[0] += 1
                    if card[1] == 'R':
                        handCards_K[1] += 1
                else:
                    if card[0] == 'S' and card[1] != 'B':
                        handCards_S[get_num(card[1])] += 1
                        handCards_A[get_num(card[1])] += 1
                    if card[0] == 'H' and card[1] != 'R':
                        handCards_H[get_num(card[1])] += 1
                        handCards_A[get_num(card[1])] += 1
                    if card[0] == 'C':
                        handCards_C[get_num(card[1])] += 1
                        handCards_A[get_num(card[1])] += 1
                    if card[0] == 'D':
                        handCards_D[get_num(card[1])] += 1
                        handCards_A[get_num(card[1])] += 1
            
            #读取手牌部分结束
            # _debug_print('Reyn_AI Tip 开始选择最优还贡牌')
            val = -get_point_val(check_card[2][0],curRank)
            # _debug_print('Reyn_AI Tip 由于该还贡行动,权值目前为',val)
            #开始删除刚刚打出的牌
            if check_card[2][0] == 'SB':
                handCards_K[0] -= 1
            if check_card[2][0] == 'HR':
                handCards_K[1] -= 1
            if check_card[2][0][0] == 'S' and check_card[2][0][1] != 'B':
                handCards_S[get_num(check_card[2][0][1])] -= 1
            if check_card[2][0][0] == 'H' and check_card[2][0][1] != 'R':
                handCards_H[get_num(check_card[2][0][1])] -= 1
            if check_card[2][0][0] == 'C':
                handCards_C[get_num(check_card[2][0][1])] -= 1
            if check_card[2][0][0] == 'D':
                handCards_D[get_num(check_card[2][0][1])] -= 1
            if check_card[2][0][1] != 'B' and check_card[2][0][1] != 'R':
                handCards_A[get_num(check_card[2][0][1])] -= 1
            if check_card[2][0][1] == curRank:
                handCards_R[1] -= 1
                if check_card[2][0][0] == 'H':
                    handCards_R[2] -= 1
            val += get_remain_VAL(handCards_S,handCards_H,handCards_C,handCards_D,handCards_A,handCards_R,handCards_K,curRank)
            # _debug_print('Reyn_AI Tip 已计算出该操作权重为',val)
            value.append(val)
        max = -50000
        AI_choice = 0
        for i in range(index + 1):
            if value[i] > max:
                AI_choice = i
                max = value[i]
        
        act = actionLists[AI_choice] 
        self.execute_back(act)
        
        return act
# -*- coding: utf-8 -*-
def _debug_print(*args, **kwargs):
    # Debug output from the original research code, silenced for
    # the open-source release.
    pass


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录

from random import randint
import random
from . import config
from guandan_rlcard.game.player import GuandanPlayer as Player

CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
    'A', 'BJ', 'RJ']

from .strategy import Strategy
from .PlayCard import PlayCard
from .CountValue import CountValue
from .CreateActionList import CreateActionList

class Action(object):

    def __init__(self):
        self.action = []
        self.act_range = -1

    def GetIndexFromBack(self, msg, retValue): #"actionList": [['back', 'back', ['S2']], ['back', 'back', ['H2']]
        retIndex = 0
        # _debug_print("retValue:", retValue)
        retAction = retValue['action']
        for action in msg["actionList"]:
            if (action[2] == retAction):
                retIndex = msg["actionList"].index(action)
        # _debug_print("选择动作：", retIndex, "动作为：", msg["actionList"][retIndex])
        return retIndex

    def GetIndexFromPlay(self, msg, retValue):
        #_debug_print("actionlist:",msg["actionList"])

        sortedAction = retValue["action"]
        if retValue["type"] != "PASS":
            sortedAction.sort()
        # _debug_print("retValue:",retValue)
        retIndex = 0
        for action in msg["actionList"]:
            if (action[2]!="PASS"): action[2].sort()
            #_debug_print("retvalue:",retValue["type"], retValue["rank"], sortedAction)
            #_debug_print("actionfromlist:",action[0], action[1], action[2])
            if (action[0]==retValue["type"] and action[1]==retValue["rank"] and action[2]==sortedAction):
                retIndex=msg["actionList"].index(action)
        # _debug_print("选择动作：", retIndex, "动作为：", msg["actionList"][retIndex])
        return retIndex

    def parse(self, msg):
        self.action = msg["actionList"]
        self.act_range = msg["indexRange"]
        # _debug_print(self.action)
        # _debug_print("可选动作范围为：0至{}".format(self.act_range))
        return randint(0, self.act_range)

class Base2Agent(Player):
    ''' Baseline 2 agent.
    '''
    name = 'Base2'
    
    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)
        self.action = Action()
        self.use_raw = True
        self.begin = True
        self.strategy = Strategy()

    def step(self, state):
        if not state['actions']:
            return []
        msg = self.parse(state)
        
        if (self.strategy.greaterPos == -1 or msg["greaterPos"]==-1 or msg["greaterPos"] == self.player_id):
            self.strategy.UpdateCurRank(msg['curRank'])
            retValue = PlayCard().FreePlay(self.strategy, msg["handCards"], msg['curRank'], msg["actionList"])
        else:
            self.strategy.UpdateCurRank(msg['curRank'])
            formerAction={"type":msg["greaterAction"][0], "rank":msg["greaterAction"][1], "action":msg["greaterAction"][2]}
            retValue = PlayCard().RestrictedPlay(self.strategy, msg["handCards"], formerAction, msg['curRank'], msg["actionList"])
        
        act_index = self.action.GetIndexFromPlay(msg, retValue)
        
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
        msg['num_cards_left'] = state['num_cards_left']
        msg['trace'] = state['trace']
        msg['reactions'] = state['reactions']
        
        if self.begin == True:
            self.begin = False
            self.strategy.SetBeginning(self.player_id, msg["handCards"])
        self.strategy.UpdatePlay(msg, msg['num_cards_left'], msg['curPos'], msg['curAction'], msg["greaterPos"], msg["greaterAction"])
            
        return msg
    
    def back_act(self, actionLists, rank, tribute_result):
        bestPlay = []
        maxValue = -100
        curRank = rank
        for rank in config.cardRanks:
            if (rank!=curRank and rank<='9' and rank>='2'):
                for card in self.current_hand_str:
                    if (card[1]==rank):
                        action = [card]
                        restCards = CreateActionList().GetRestCards(action, self.current_hand_str)
                        restValue, restActions = CountValue().HandCardsValue(restCards, 0, curRank)
                        if (restValue>maxValue):
                            maxValue = restValue
                            bestPlay = {"action": action, "type": "back", "rank": rank}
                        #_debug_print(card, restValue)
                        break
        
        retIndex = 0
        retAction = bestPlay['action']
        for action in actionLists:
            if (action[2] == retAction):
                retIndex = actionLists.index(action)
                break
        
        act = actionLists[retIndex]
        self.execute_back(act)
        
        return act
    
    def tribute_act(self, actionLists, rank):
        """根据合法动作集选择动作执行"""
        act = random.choice(actionLists)
        self.execute_tribute(act)
        return act
    
    def reset(self):
        super().reset()
        self.strategy.Clear()
        self.begin = True
    
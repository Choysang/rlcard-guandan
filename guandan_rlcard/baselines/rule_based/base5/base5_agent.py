# -*- coding: utf-8 -*-
def _debug_print(*args, **kwargs):
    # Debug output from the original research code, silenced for
    # the open-source release.
    pass


# @Time       : 2020/10/1 16:30
# @Author     : Duofeng Wu

import copy
from random import randint, random
from guandan_rlcard.game.player import GuandanPlayer as Player


CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
    'A', 'BJ', 'RJ']

class Action(object):

    def __init__(self):
        self.action = []
        self.act_range = -1

    def rule_parse(self,msg,mypos,remaincards,history,pass_num,my_pass_num):
        self.action = msg["actionList"]
        if len(self.action) == 1:
            return 0
        if msg["greaterPos"] != mypos and msg["greaterPos"] != -1:
            # _debug_print("5被动")
            try:
            # 被动接牌
                numofplayers = [history['0']["remain"],history['1']["remain"],history['2']["remain"],history['3']["remain"]]
                self.act = passive(self.action, msg["handCards"], msg["curRank"], msg['curAction'], msg["greaterAction"],mypos,
                                        msg["greaterPos"],remaincards, numofplayers,pass_num,my_pass_num)
            except Exception as e:
                # _debug_print(str(e))
                self.act = 1

        elif msg["greaterPos"] == mypos or msg["greaterPos"] == -1:
            # "主动出牌"
            # _debug_print("5主动")
            try:
                numofplayers = [history['0']["remain"], history['1']["remain"], history['2']["remain"],
                                history['3']["remain"]]
                self.act = active(self.action, msg["handCards"], msg["curRank"],numofplayers,mypos,remaincards)
            except Exception as e:
                # _debug_print(e)
                self.act = 0
        else:
            _debug_print(msg["handCards"])
            # _debug_print(f"{mypos}, {msg["greaterPos"]}, {msg["curPos"]}, 5随机")
            self.act_range = msg["indexRange"]
            self.act = randint(0, self.act_range)

        return self.act


class Base5Agent(Player):
    ''' Baseline 5 agent.
    '''
    name = 'Base5'
    
    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)
        self.Action = Action()
        self.use_raw = True

    def step(self, state):
        if not state['actions']:
            return []
        msg = self.parse(state)
        
        act_index = self.Action.rule_parse(msg, self.player_id, msg['remain_cards'], msg['history'],
            msg['pass_num'], msg['my_pass_num'])
        
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
    
    def back_act(self, actionLists, rank, tribute_result):
        """根据合法动作集选择动作执行"""
        self.action = actionLists
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
            for i in range(len(self.action)):
                if self.action[i][2][0] == target:
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

    
'''active.py'''

def getlist( handcards, rank):
    single_actionlist = []
    pair_actionlist = []
    trips_actionlist = []
    threepair_actionlist = []
    threetwo_actionlist = []
    twotrips_actionlist = []
    straight_actionlist = []

    action2 = "None"
    action3 = "None"

    rank_card = 'H' + str(rank)

    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                      "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
    card_value_s2v2 = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                       "Q": 12, "K": 13, "B": 16, "R": 17}
    card_value_s2v[rank_card[-1]] = 15
    sorted_cards, bomb_info = combine_handcards(handcards, rank, card_value_s2v)

    def mysort(elem):
        return card_value_s2v[elem[0]]

    def mysort1(elem):
        return card_value_s2v2[elem[0]]

    if sorted_cards["Single"]:
        for singlecard in sorted_cards['Single']:
            single_actionlist.append([singlecard[-1], singlecard])
        single_actionlist.sort(key=mysort)

    if sorted_cards["Pair"]:
        for paircard in sorted_cards['Pair']:
            pair_actionlist.append([paircard[0][-1], paircard])
        pair_actionlist.sort(key=mysort)

    if sorted_cards['Trips']:
        for tripcard in sorted_cards['Trips']:
            trips_actionlist.append([tripcard[0][-1], tripcard])
        trips_actionlist.sort(key=mysort)

    if sorted_cards['Pair'] and sorted_cards['Trips']:
        for tripcard in sorted_cards['Trips']:
            for paircard in sorted_cards['Pair']:
                threetwo_actionlist.append([tripcard[0][-1], tripcard + paircard])
        threetwo_actionlist.sort(key=mysort)

    if len(sorted_cards['Pair']) >= 3:
        for i in range(len(pair_actionlist) - 2):
            if card_value_s2v[pair_actionlist[i][0]] == card_value_s2v[pair_actionlist[i + 1][0]] - 1 and \
                    card_value_s2v[pair_actionlist[i + 1][0]] == card_value_s2v[pair_actionlist[i + 2][0]] - 1:
                action2 = pair_actionlist[i][-1] + pair_actionlist[i + 1][-1] + pair_actionlist[i + 2][-1]
                threepair_actionlist.append([action2[0][-1], action2])
        threepair_actionlist.sort(key=mysort1)

    if len(sorted_cards['Trips']) >= 2:
        for i in range(len(trips_actionlist) - 1):
            if card_value_s2v[trips_actionlist[i][0]] == card_value_s2v[trips_actionlist[i + 1][0]] - 1:
                action3 = trips_actionlist[i][-1] + trips_actionlist[i + 1][-1]
                twotrips_actionlist.append([action3[0][-1], action3])
        twotrips_actionlist.sort(key=mysort1)

    if 'Straight' in sorted_cards.keys() and sorted_cards['Straight']:
        for straightcard in sorted_cards['Straight']:
            straight_actionlist.append([straightcard[0][-1], straightcard])
        straight_actionlist.sort(key=mysort1)

    return sorted_cards, single_actionlist, pair_actionlist, trips_actionlist, threepair_actionlist, threetwo_actionlist, twotrips_actionlist, straight_actionlist


def active(actionList, handcards, rank, numofplayers, mypos, remaincards):
    restcards = rest_cards(handcards, remaincards, rank)
    rank_card = 'H' + rank
    numofnext = numofplayers[(mypos + 1) % 4]
    if numofnext == 0:
        numofnext = numofplayers[(mypos - 1) % 4]

    cur = [9, 10, 9, 8, 10, 10, 2]
    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                      "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17, 'BJ':16, 'RJ':17}
    card_value_s2v2 = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                       "Q": 12, "K": 13, "B": 16, "R": 17, 'BJ':16, 'RJ':17}
    card_value_s2v[rank] = 15

    sorted_cards, single_actionlist, pair_actionlist, trips_actionlist, threepair_actionlist, threetwo_actionlist, twotrips_actionlist, straight_actionlist = getlist(  # FIX: was self.getlist - NameError silently disabled the leading-play logic
            
        handcards, rank)
    # _debug_print(len(single_actionlist), len(pair_actionlist), len(trips_actionlist), len(threetwo_actionlist),
    #       len(threepair_actionlist), len(twotrips_actionlist), len(straight_actionlist))

    max_val = card_value_s2v[restcards[-1][0][-1]]

    for i in actionList:
        if len(handcards) == len(i[2]):
            return actionList.index(i)

    twohand_candidatelist = []

    def mysort2(elem):
        return elem[1]

    if len(handcards) <= 12:
        for i in range(len(actionList)):
            for j in range(i + 1, len(actionList)):
                if len(actionList[i][-1]) + len(actionList[j][-1]) == len(handcards):
                    combine_list = actionList[i][-1] + actionList[j][-1]
                    if combine_list.sort(key=mysort2) == handcards.sort(key=mysort2):
                        twohand_candidatelist.append((i, j))

    if len(single_actionlist) and card_value_s2v[single_actionlist[0][0]] < cur[0]:
        if numofnext == 1:
            pass
        else:
            return getindex("Single", single_actionlist, actionList)

    if len(threepair_actionlist) or len(twotrips_actionlist):
        index = rankfour(twotrips_actionlist, threepair_actionlist, actionList, cur[1], cur[2])
        if index is None:
            pass
        else:
            return index

    if len(straight_actionlist) and card_value_s2v2[straight_actionlist[0][0]] < cur[4]:
        return getindex("Straight", straight_actionlist, actionList)

    if len(threetwo_actionlist):

        index = rankthree(single_actionlist, pair_actionlist, trips_actionlist, threetwo_actionlist, actionList,
                          numofnext,
                          rank, cur[0], cur[3], cur[4], cur[5], cur[-1])
        if index is None:
            pass
        else:
            return index
    if len(trips_actionlist):
        return rankone(single_actionlist, trips_actionlist, actionList, numofnext, rank)
    if len(pair_actionlist):
        return ranktwo(handcards, single_actionlist, pair_actionlist, trips_actionlist, actionList, numofnext, rank,
                       max_val)
    if len(single_actionlist):
        if numofnext == 1 and len(trips_actionlist) == 0 and len(pair_actionlist) == 0 and rank_card in handcards:
            for i in range(len(actionList)):
                if actionList[i][0] == 'Pair' and (
                        actionList[i][-1][0] in sorted_cards['Single'] or actionList[i][-1][-1] in sorted_cards['Single']):
                    return i
        if numofnext == 1:
            now_max_act_value = 0
            now_max_act_key = 0
            for acti in range(len(actionList)):
                if actionList[acti][0] == 'Single' and actionList[acti][-1][0] in sorted_cards['Single']:
                    if card_value_s2v[actionList[acti][1]] > now_max_act_value:
                        now_max_act_value = card_value_s2v[actionList[acti][1]]
                        now_max_act_key = acti

            return now_max_act_key
        return getindex("Single", single_actionlist, actionList)
    else:
        return 0


'''passive.py'''

def Single( actionList, curAction, rank_card, handcards, numofplayers, rest_cards, card_val, myPos, greaterPos,
           pass_num, my_pass_num):
    numofnext = numofplayers[(myPos + 1) % 4]
    numofgreaterPos = numofplayers[greaterPos]
    numoffri = numofplayers[(myPos + 2) % 4]
    numofmy = numofplayers[myPos]
    numofpre = numofplayers[(myPos - 1) % 4]

    sorted_cards, bomb_info = combine_handcards(handcards, rank_card[1], card_val)

    bomb_member = []
    pair_member = []
    trip_member = []
    single_member = sorted_cards["Single"]
    straight_member = []
    if len(sorted_cards["Straight"]) != 0:
        straight_member += sorted_cards["Straight"][0]
    if len(sorted_cards["StraightFlush"]) != 0:
        straight_member += sorted_cards["StraightFlush"][0]

    for pair in sorted_cards["Pair"]:
        pair_member += pair
    for trip in sorted_cards["Trips"]:
        trip_member += trip
    for bomb in sorted_cards["Bomb"]:
        bomb_member += bomb

    tag = 0
    single_actionList = []
    bomb_actionList = []
    for action in actionList[1:]:
        tag += 1
        if action[0] == 'Single':
            single_actionList.append((tag, action))
        else:
            bomb_actionList.append((tag, action))

    curVal = card_val[curAction[1]]

    max_val = card_val[rest_cards[-1][0][1]]

    if numofnext == 0:
        numofnext = numofplayers[(myPos - 1) % 4]

    if numofnext <= 4 or (numofpre <= 3 and numofpre >= 1):


        if (myPos + 2) % 4 == greaterPos and curVal >= max_val:
            return 0
        if (myPos + 2) % 4 == greaterPos and curVal >= 15 and numofnext != 1:
            return 0

        for action in single_actionList:
            Index = action[0]
            action = action[1]
            if card_val[action[1]] >= max_val and action[2][0] in single_member and rank_card not in action[2]:
                return Index

        for action in single_actionList:
            Index = action[0]
            action = action[1]
            if card_val[action[1]] >= max_val and action[2][0] not in bomb_member and rank_card not in action[2]:
                if is_inStraight(action, straight_member):
                    continue
                return Index

        index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
        if index != -1:
            return index

        for action in single_actionList:
            Index = action[0]
            action = action[1]
            if card_val[action[1]] >= max_val - 2 and action[2][0] not in bomb_member and rank_card not in action[2]:
                if is_inStraight(action, straight_member):
                    continue
                return Index

        for action in single_actionList:
            Index = action[0]
            action = action[1]
            if rank_card in action[2] and (len(sorted_cards["Pair"]) < 3 or numofnext == 1):
                return Index

    def normal(single_actionList, single_member, rank_card):
        for action in single_actionList:
            Index = action[0]
            action = action[1]
            # _debug_print(action)
            if (action[2][0] in single_member or card_val[action[1]] >= 15) and rank_card not in action[2]:
                return Index
        return -1

    def special(single_actionList, bomb_member, straight_member, rank_card):
        for action in single_actionList[::-1]:
            Index = action[0]
            action = action[1]
            if action[2][0] not in bomb_member and rank_card not in action[2]:
                if is_inStraight(action, straight_member):
                    continue
                return Index
        return -1

    if (myPos + 2) % 4 == greaterPos:
        if curVal >= 14 or curVal >= max_val - 2:
            return 0
        elif numoffri <= 4:
            index = normal(single_actionList, single_member, rank_card)
            if index == -1:
                return 0
            if curVal <= 10:
                return index
            else:
                # _debug_print(index)
                if card_val[actionList[index][1]] == curVal + 1:
                    return index
        else:
            index = normal(single_actionList, single_member, rank_card)
            if index != -1:
                return index
            else:
                return 0
    else:
        index = normal(single_actionList, single_member, rank_card)
        if index != -1:
            return index
        else:
            if pass_num >= 5 or my_pass_num >= 3:
                index = special(single_actionList, bomb_member, straight_member, rank_card)
                if index != -1:
                    return index
            cur_bomb_num = cal_bomb_num(sorted_cards, handcards, rank_card)
            if curVal >= max_val and numofgreaterPos >= 15 and cur_bomb_num > 1:
                p = random()
                index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                if p > 0.5:
                    if index != -1:
                        return index
            elif ((curVal >= 15 or curVal >= max_val - 2) and numofgreaterPos <= 15) or pass_num >= 7 or my_pass_num >= 5:
                index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                if index != -1:
                    return index
                else:
                    return 0

    return 0

def Pair( actionList, curAction, rank_card, handcards, numofplayers, rest_cards, card_val, myPos, greaterPos,
         pass_num, my_pass_num):
    numofnext = numofplayers[(myPos + 1) % 4]
    numofgreaterPos = numofplayers[greaterPos]
    numoffri = numofplayers[(myPos + 2) % 4]
    numofpre = numofplayers[(myPos - 1) % 4]
    sorted_cards, bomb_info = combine_handcards(handcards, rank_card[-1], card_val)

    bomb_member = []
    pair_member = []
    trip_member = []
    single_member = sorted_cards["Single"]
    straight_member = []
    if len(sorted_cards["Straight"]) != 0:
        straight_member += sorted_cards["Straight"][0]
    if len(sorted_cards["StraightFlush"]) != 0:
        straight_member += sorted_cards["StraightFlush"][0]

    for pair in sorted_cards["Pair"]:
        pair_member += pair
    for trip in sorted_cards["Trips"]:
        trip_member += trip
    for bomb in sorted_cards["Bomb"]:
        bomb_member += bomb

    pair_actionList = []
    bomb_actionList = []
    tag = 0
    for action in actionList[1:]:
        tag += 1
        if action[0] == 'Pair':
            pair_actionList.append((tag, action))
        else:
            bomb_actionList.append((tag, action))

    curVal = card_val[curAction[1]]
    rest_cards = rest_cards[::-1]
    max_val = 0
    for cards in rest_cards:
        if len(cards) >= 2:
            max_val = card_val[cards[0][1]]
            break

    if numofnext == 0:
        numofnext = numofplayers[(myPos - 1) % 4]

    if numofnext <= 4 or (numofpre <= 4 and numofpre >= 1):

        if (myPos + 2) % 4 == greaterPos and curVal >= max_val:
            return 0
        if (myPos + 2) % 4 == greaterPos and curVal >= 12 and numofnext != 2:
            return 0

        for action in pair_actionList:
            Index = action[0]
            action = action[1]
            if card_val[action[1]] >= max_val and action[2][0] in pair_member and rank_card not in action[2]:
                return Index

        for action in pair_actionList:
            Index = action[0]
            action = action[1]
            if card_val[action[1]] >= max_val and action[2][0] not in bomb_member and rank_card not in action[2]:
                if is_inStraight(action, straight_member):
                    continue
                return Index

        index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
        if index != -1:
            return index

        for action in pair_actionList[::-1]:
            Index = action[0]
            action = action[1]
            if card_val[action[1]] >= max_val - 2 and action[2][0] not in bomb_member and rank_card not in action[2]:
                if is_inStraight(action, straight_member):
                    continue
                return Index

        max_match = -1
        max_match_index = -1
        for action in pair_actionList:
            index = action[0]
            action = action[1]
            if rank_card in action[2] and card_val[action[1]] > max_match and action[2][0] not in bomb_member:
                if is_inStraight(action, straight_member):
                    continue
                max_match = card_val[action[1]]
                max_match_index = index
        if max_match_index != -1 and max_match >= max_val - 2:
            return max_match_index


    def normal(pair_actionList, pair_member, rank_card):
        for action in pair_actionList:
            Index = action[0]
            action = action[1]
            if (action[2][0] in pair_member or action[1] == rank_card[1]) and rank_card not in action[2]:
                return Index

        return -1

    def special(pair_actionList, bomb_member, straight_member, rank_card):
        for action in pair_actionList[::-1]:
            Index = action[0]
            action = action[1]

            if action[2][0] not in bomb_member and rank_card not in action[2]:
                if is_inStraight(action, straight_member):
                    continue
                return Index
        return -1

    if (myPos + 2) % 4 == greaterPos:
        if curVal >= 13 or curVal >= max_val - 2:
            return 0
        elif numoffri <= 4:
            index = normal(pair_actionList, pair_member, rank_card)
            if index == -1:
                return 0
            if curVal <= 10:
                return index
            else:
                if card_val[actionList[index][1]] == curVal + 1:
                    return index

        else:
            index = normal(pair_actionList, pair_member, rank_card)
            if index != -1:
                return index
            else:
                return 0
    else:
        index = normal(pair_actionList, pair_member, rank_card)
        if index != -1:
            return index
        else:
            if pass_num >= 5 or my_pass_num >= 3:
                index = special(pair_actionList, bomb_member, straight_member, rank_card)
                if index != -1:
                    return index
            cur_bomb_num = cal_bomb_num(sorted_cards, handcards, rank_card)
            if curVal >= max_val and numofgreaterPos >= 15 and cur_bomb_num > 1:
                p = random()
                index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                if p > 0.5:
                    if index != -1:
                        return index
            elif ((curVal >= 14 or curVal >= max_val - 2) and numofgreaterPos <= 15) or pass_num >= 6 or my_pass_num >= 5:
                index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                if index != -1:
                    return index
                else:
                    return 0

    return 0

def ThreeWithTwo( actionList, curAction, rank_card, handcards, numofplayers,
                 rest_cards, card_val, myPos, greaterPos, pass_num, my_pass_num):
    numofnext = numofplayers[(myPos + 1) % 4]
    numofgreaterPos = numofplayers[greaterPos]
    numoffri = numofplayers[(myPos + 2) % 4]
    numofpre = numofplayers[(myPos - 1) % 4]

    sorted_cards, bomb_info = combine_handcards(handcards, rank_card[-1], card_val)

    bomb_member = []
    pair_member = []
    trip_member = []
    single_member = sorted_cards["Single"]
    straight_member = []
    if len(sorted_cards["Straight"]) != 0:
        straight_member += sorted_cards["Straight"][0]
    if len(sorted_cards["StraightFlush"]) != 0:
        straight_member += sorted_cards["StraightFlush"][0]

    for pair in sorted_cards["Pair"]:
        pair_member += pair
    for trip in sorted_cards["Trips"]:
        trip_member += trip
    for bomb in sorted_cards["Bomb"]:
        bomb_member += bomb

    three2_actionList = []
    bomb_actionList = []
    tag = 0

    for action in actionList[1:]:
        tag += 1
        if (action[0] == 'ThreeWithTwo'):
            three2_actionList.append((tag, action))
        else:
            bomb_actionList.append((tag, action))

    curVal = card_val[curAction[1]]  # 当前牌的值
    max_val = 0
    for cards in rest_cards[::-1]:
        if len(cards) >= 3:
            max_val = card_val[cards[0][-1]]  # 另外三方的可能最大值
            break

    if numofnext == 0:
        numofnext = numofplayers[(myPos - 1) % 4]

    if numofnext <= 7 or (numofpre <= 7 and numofpre >= 1):

        if (myPos + 2) % 4 == greaterPos and curVal >= max_val:
            return 0
        if (myPos + 2) % 4 == greaterPos and curVal >= 11 and numofnext != 5:
            return 0

        three2_sorted = sorted(three2_actionList, key=lambda item: card_val[item[1][1]], reverse=True)
        for action in three2_sorted:
            index = action[0]
            action = action[1]
            trip = action[2][0]
            pair = action[2][3]
            if trip in trip_member and pair in pair_member and rank_card not in action[2] and card_val[pair[1]] <= 13:
                return index

        for action in three2_sorted:
            index = action[0]
            action = action[1]
            trip = action[2][0]
            pair = action[2][3]
            if trip in trip_member and pair in trip_member and rank_card not in action[2] and card_val[pair[1]] >= 10:
                return index

        index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
        if index != -1:
            return index
        for action in three2_sorted:
            index = action[0]
            action = action[1]
            trip = action[2][0]
            pair = action[2][3]
            if trip in pair_member and pair in pair_member and rank_card in action[2]:
                return index

    def normal(three2_actionList, trip_member, pair_member, rank_card):
        for action in three2_actionList:
            index = action[0]
            action = action[1]
            trip = action[2][0]
            pair = action[2][3]
            if trip in trip_member and pair in pair_member and rank_card not in action[2] and card_val[pair[-1]] <= 13:
                return index
        return -1

    if (myPos + 2) % 4 == greaterPos:
        if curVal >= 14 or curVal >= max_val - 2:
            return 0
        elif numoffri <= 5:
            index = normal(three2_actionList, trip_member, pair_member, rank_card)
            if index == -1:
                return 0
            if curVal <= 10:
                return index
            else:
                if card_val[actionList[index][1]] == curVal + 1:
                    return index
        else:
            index = normal(three2_actionList, trip_member, pair_member, rank_card)
            if index != -1:
                return index
            else:
                return 0
    else:
        index = normal(three2_actionList, trip_member, pair_member, rank_card)
        if index != -1:
            return index
        else:
            if curVal >= max_val and numofgreaterPos >= 15:
                p = random()
                if p > 0.5:
                    index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                    if index != -1:
                        return index
            if ((curVal >= 12 or curVal >= max_val - 2) and numofgreaterPos <= 15) or pass_num >= 5 or my_pass_num >= 3:
                index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                if index != -1:
                    return index
                else:
                    return 0
    return 0

def Trips( actionList, curAction, rank_card, handcards, numofplayers, rest_cards, card_val, myPos, greaterPos,
          pass_num, my_pass_num):
    numofnext = numofplayers[(myPos + 1) % 4]
    numofgreaterPos = numofplayers[greaterPos]
    numoffri = numofplayers[(myPos + 2) % 4]
    numofpre = numofplayers[(myPos - 1) % 4]

    sorted_cards, bomb_info = combine_handcards(handcards, rank_card[-1], card_val)

    bomb_member = []
    pair_member = []
    trip_member = []
    single_member = sorted_cards["Single"]
    straight_member = []
    if len(sorted_cards["Straight"]) != 0:
        straight_member += sorted_cards["Straight"][0]
    if len(sorted_cards["StraightFlush"]) != 0:
        straight_member += sorted_cards["StraightFlush"][0]

    for pair in sorted_cards["Pair"]:
        pair_member += pair
    for trip in sorted_cards["Trips"]:
        trip_member += trip
    for bomb in sorted_cards["Bomb"]:
        bomb_member += bomb

    trip_actionList = []
    bomb_actionList = []
    tag = 0
    for action in actionList[1:]:
        tag += 1
        if action[0] == 'Trips':
            trip_actionList.append((tag, action))
        else:
            bomb_actionList.append((tag, action))

    curVal = card_val[curAction[1]]
    rest_cards = rest_cards[::-1]
    max_val = 0
    for cards in rest_cards:
        if len(cards) >= 3:
            max_val = card_val[cards[0][-1]]
            break


    if numofnext == 0:
        numofnext = numofplayers[(myPos - 1) % 4]

    if numofnext <= 6 or (numofpre <= 5 and numofpre >= 1):

        if (myPos + 2) % 4 == greaterPos and curVal >= max_val:
            return 0
        if (myPos + 2) % 4 == greaterPos and curVal >= 12 and numofnext != 3:
            return 0

        for action in trip_actionList:
            Index = action[0]
            action = action[1]
            if card_val[action[1]] >= max_val and action[2][0] in trip_member and action[2] and rank_card not in action[
                2]:
                return Index

        index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
        if index != -1:
            return index

        for action in trip_actionList[::-1]:
            Index = action[0]
            action = action[1]
            if card_val[action[1]] >= max_val - 2 and action[2][0] in trip_member and rank_card not in action[2]:
                if is_inStraight(action, straight_member):
                    continue
                return Index
        max_match = -1
        max_match_index = -1
        for action in trip_actionList:
            index = action[0]
            action = action[1]
            if rank_card in action[2] and card_val[action[1]] > max_match and action[2][0] not in bomb_member:
                if is_inStraight(action, straight_member):
                    continue
                max_match = card_val[action[1]]
                max_match_index = index
        if max_match_index != -1:
            return max_match_index

    def normal(trip_actionList, trip_member, rank_card):
        for action in trip_actionList:
            Index = action[0]
            action = action[1]
            if action[2][0] in trip_member and rank_card not in action[2]:
                return Index
        return -1

    if (myPos + 2) % 4 == greaterPos:
        if curVal >= 13 or curVal >= max_val - 2:
            return 0
        elif numoffri <= 4:
            index = normal(trip_actionList, trip_member, rank_card)
            if index == -1:
                return 0
            if curVal <= 10:
                return index
            else:
                if card_val[actionList[index][1]] == curVal + 1:
                    return index
        else:
            index = normal(trip_actionList, trip_member, rank_card)
            if index != -1:
                return index
            else:
                return 0
    else:
        index = normal(trip_actionList, trip_member, rank_card)
        if index != -1:
            return index
        else:
            if curVal >= max_val and numofgreaterPos >= 15:
                p = random()
                if p > 0.5:
                    index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                    if index != -1:
                        return index
            if ((curVal >= 12 or curVal >= max_val - 2) and numofgreaterPos <= 15) or pass_num >= 5 or my_pass_num >= 3:
                index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                if index != -1:
                    return index

    return 0

def ThreePair( actionList, curAction, rank_card, handcards, numofplayers, rest_cards, card_val, myPos, greaterPos,
              pass_num, my_pass_num):
    numofnext = numofplayers[(myPos + 1) % 4]
    numofgreaterPos = numofplayers[greaterPos]
    numoffri = numofplayers[(myPos + 2) % 4]
    numofpre = numofplayers[(myPos - 1) % 4]
    sorted_cards, bomb_info = combine_handcards(handcards, rank_card[-1], card_val)


    card_origin = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                   "Q": 12, "K": 13}
    card_val['A'] = 1
    card_val[rank_card[1]] = card_origin[rank_card[1]]

    bomb_member = []
    pair_member = []
    trip_member = []
    single_member = sorted_cards["Single"]
    straight_member = []
    if len(sorted_cards["Straight"]) != 0:
        straight_member += sorted_cards["Straight"][0]
    if len(sorted_cards["StraightFlush"]) != 0:
        straight_member += sorted_cards["StraightFlush"][0]

    for pair in sorted_cards["Pair"]:
        pair_member += pair
    for trip in sorted_cards["Trips"]:
        trip_member += trip
    for bomb in sorted_cards["Bomb"]:
        bomb_member += bomb

    pair3_actionList = []
    bomb_actionList = []

    tag = 0
    for action in actionList[1:]:
        tag += 1
        if (action[0] == 'ThreePair'):
            pair3_actionList.append((tag, action))
        else:
            bomb_actionList.append((tag, action))

    curVal = card_val[curAction[1]]
    max_val = 0
    val_list = []
    for cards in rest_cards:
        if len(cards) >= 2:
            val_list.append(card_val[cards[0][1]])
    val_list = sorted(val_list)

    for i in range(0, len(val_list)):
        if i >= len(val_list) - 2:
            break
        if (val_list[i] + 1 == val_list[i + 1] and val_list[i] + 2 == val_list[i + 2]):
            max_val = max(max_val, val_list[i])

    if len(val_list) >= 3 and (val_list[0] == 1 and val_list[-2] == 12 and val_list[-1] == 13):
        max_val = 12


    def normal(pair3_actionList, pair_member, rank_card):
        for action in pair3_actionList:
            index = action[0]
            action = action[1]
            first = action[2][0]
            mid = action[2][2]
            last = action[2][4]

            if first in pair_member and mid in pair_member and last in pair_member and rank_card not in action[2]:
                return index
        return -1

    def special(pair3_actionList, trip_member, rank_card):
        for action in pair3_actionList:
            index = action[0]
            action = action[1]
            first = action[2][0]
            mid = action[2][2]
            last = action[2][4]
            if rank_card in action[2]:
                continue
            if first in pair_member and mid in pair_member and last in trip_member:
                return index
            if first in pair_member and mid in trip_member and last in pair_member:
                return index
            if first in trip_member and mid in pair_member and last in pair_member:
                return index
        return -1

    def match_rank_card(pair3_actionList, rank_card, pair_member):
        for action in pair3_actionList:
            index = action[0]
            action = action[1]
            first = action[2][1]
            mid = action[2][3]
            last = action[2][5]
            if first == rank_card and mid in pair_member and last in pair_member:
                return index
            if first in pair_member and mid == rank_card and last in pair_member:
                return index
            if first in pair_member and mid == rank_card and last in pair_member:
                return index
        return -1

    if (myPos + 2) % 4 == greaterPos:
        if curVal >= 10 or curVal >= max_val - 2:
            return 0
        elif numoffri <= 4:
            index = normal(pair3_actionList, pair_member, rank_card)
            if index == -1:
                return 0
            if curVal <= 7:
                return index
            else:
                if card_val[actionList[index][1]] == curVal + 1:
                    return index
        else:
            index = normal(pair3_actionList, pair_member, rank_card)
            if index != -1:
                return index
            else:
                return 0
    else:
        index = normal(pair3_actionList, pair_member, rank_card)
        if index != -1:
            return index
        else:
            index = special(pair3_actionList, trip_member, rank_card)
            if index != -1:
                return index
            if len(trip_member) == 0 and rank_card in handcards:
                index = match_rank_card(pair3_actionList, rank_card, pair_member)
                if index != -1:
                    return index
            if curVal >= max_val and numofgreaterPos >= 15:
                index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                if index != -1:
                    return index
            elif ((curVal >= 10 or curVal >= max_val - 2) and numofgreaterPos <= 15) or pass_num >= 5 or my_pass_num >= 3:
                index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                if index != -1:
                    return index
                else:
                    return 0
    return 0

def Straight( actionList, curAction, rank_card, handcards, numofplayers, card_val, pass_num, my_pass_num, myPos,
             greaterPos):


    numofnext = numofplayers[(myPos + 1) % 4]
    numofpre = numofplayers[(myPos - 1) % 4]
    if numofnext == 0:
        numofnext = numofplayers[(myPos - 1) % 4]

    sorted_cards, bomb_info = combine_handcards(handcards, rank_card[-1], card_val)

    card_origin = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                   "Q": 12, "K": 13, "R": 14, "B": 15}
    card_val['A'] = 1
    card_val[rank_card[1]] = card_origin[rank_card[1]]

    curVal = card_val[curAction[1]]

    bomb_member = []
    pair_member = []
    trip_member = []
    single_member = sorted_cards["Single"]
    straight_member = []
    if len(sorted_cards["Straight"]) != 0:
        straight_member += sorted_cards["Straight"][0]
    if len(sorted_cards["StraightFlush"]) != 0:
        straight_member += sorted_cards["StraightFlush"][0]

    for pair in sorted_cards["Pair"]:
        pair_member += pair
    for trip in sorted_cards["Trips"]:
        trip_member += trip
    for bomb in sorted_cards["Bomb"]:
        bomb_member += bomb

    straight_actionList = []
    bomb_actionList = []
    tag = 0
    for action in actionList[1:]:
        tag += 1
        if action[0] == 'Straight':
            straight_actionList.append((tag, action))
        else:
            bomb_actionList.append((tag, action))

    if len(sorted_cards["Straight"]) > 0:
        curStraight = sorted_cards["Straight"][0][0][1]
        for action in straight_actionList:
            Index = action[0]
            action = action[1]
            if curStraight == action[1] and rank_card not in action[2]:
                if (myPos + 2) % 4 == greaterPos:
                    if curVal <= 7 or card_val[curStraight] - curVal <= 2:
                        return Index
                else:
                    return Index
    elif (myPos + 2) != greaterPos:
        for action in straight_actionList:
            Index = action[0]
            action = action[1]
            if rank_card in action[2] and len(trip_member) == 0:
                if len(set(action[2]).intersection(set(bomb_member))) != 0:
                    continue
                if is_inStraight(action, straight_member):
                    continue
                new_handcards = []
                for card in handcards:
                    if card not in action[2]:
                        new_handcards.append(card)

                new_card_val = copy.deepcopy(card_val)
                new_card_val['A'] = 14
                new_card_val[rank_card[1]] = 15
                originSinglenum = len(single_member)
                new_sorted_cards, _ = combine_handcards(new_handcards, rank_card, new_card_val)
                curSinglenum = len(new_sorted_cards["Single"])
                if curSinglenum <= originSinglenum:
                    return Index

        if (numofnext <= 15 or curVal >= 9) or numofnext <= 10 or pass_num >= 5 or my_pass_num >= 3 or (
                numofpre <= 5 and numofpre >= 1):
            index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
            if index != -1:
                return index
    return 0

def TwoTrips( actionList, curAction, rank_card, handcards, numofplayers, rest_cards, card_val, myPos, greaterPos,
             pass_num, my_pass_num):
    numofnext = numofplayers[(myPos + 1) % 4]
    numofgreaterPos = numofplayers[greaterPos]
    numoffri = numofplayers[(myPos + 2) % 4]

    sorted_cards, bomb_info = combine_handcards(handcards, rank_card[-1], card_val)
    card_origin = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                   "Q": 12, "K": 13}
    card_val['A'] = 1
    card_val[rank_card[1]] = card_origin[rank_card[1]]

    bomb_member = []
    pair_member = []
    trip_member = []
    single_member = sorted_cards["Single"]
    straight_member = []
    if len(sorted_cards["Straight"]) != 0:
        straight_member += sorted_cards["Straight"][0]
    if len(sorted_cards["StraightFlush"]) != 0:
        straight_member += sorted_cards["StraightFlush"][0]

    for pair in sorted_cards["Pair"]:
        pair_member += pair
    for trip in sorted_cards["Trips"]:
        trip_member += trip
    for bomb in sorted_cards["Bomb"]:
        bomb_member += bomb

    twoTripsList = []
    bomb_actionList = []
    tag = 0

    for action in actionList[1:]:
        tag += 1
        if (action[0] == "TwoTrips"):
            twoTripsList.append((tag, action))
        else:
            bomb_actionList.append((tag, action))

    curVal = card_val[curAction[1]]
    max_val = 0
    val_list = []
    for cards in rest_cards:
        if len(cards) >= 3:
            val_list.append(card_val[cards[0][1]])
    val_list = sorted(val_list)
    for i in range(0, len(val_list)):
        if (i >= len(val_list) - 1):
            break
        if (val_list[i] + 1 == val_list[i + 1]):
            max_val = max(max_val, val_list[i])
    if len(val_list) >= 2 and val_list[0] == 1 and val_list[-1] == 13:
        max_val = 13


    def normal(twoTripsList, trip_member, rank_card):
        for action in twoTripsList:
            index = action[0]
            action = action[1]
            first = action[2][0]
            last = action[2][3]
            if first in trip_member and last in trip_member and rank_card not in action[2]:
                return index
        return -1

    if (myPos + 2) % 4 == greaterPos:
        if curVal >= 10 or curVal >= max_val - 2:
            return 0
        elif numoffri <= 4:
            index = normal(twoTripsList, trip_member, rank_card)
            if index == -1:
                return 0
            if curVal <= 10:
                return index
            else:
                if card_val[actionList[index][1]] == curVal + 1:
                    return index
        else:
            index = normal(twoTripsList, trip_member, rank_card)
            if index != -1:
                return index
            else:
                return 0
    else:
        index = normal(twoTripsList, trip_member, rank_card)
        if index != -1:
            return index
        else:
            if curVal >= max_val and numofgreaterPos >= 15:
                p = random()
                if p > 0.5:
                    index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                    if index != -1:
                        return index
            if ((curVal >= 10 or curVal >= max_val - 2) and numofgreaterPos <= 15) or pass_num >= 5 or my_pass_num >= 3:
                index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
                if index != -1:
                    return index
                else:
                    return 0
    return 0

def Bomb( actionList, curAction, rank_card, handcards, numofplayers, rest_cards, card_val, myPos, greaterPos):
    numofnext = numofplayers[(myPos + 1) % 4]
    numofgreaterPos = numofplayers[greaterPos]
    if (myPos + 2) % 4 == greaterPos:
        return 0

    sorted_cards, bomb_info = combine_handcards(handcards, rank_card[1], card_val)
    cur_Bomb_num = cal_bomb_num(sorted_cards, handcards, rank_card)

    bomb_member = []
    pair_member = []
    trip_member = []
    single_member = sorted_cards["Single"]
    straight_member = []
    if len(sorted_cards["Straight"]) != 0:
        straight_member += sorted_cards["Straight"][0]
    if len(sorted_cards["StraightFlush"]) != 0:
        straight_member += sorted_cards["StraightFlush"][0]

    for pair in sorted_cards["Pair"]:
        pair_member += pair
    for trip in sorted_cards["Trips"]:
        trip_member += trip
    for bomb in sorted_cards["Bomb"]:
        bomb_member += bomb
    bomb_actionList = []
    tag = 0
    for action in actionList[1:]:
        tag += 1
        bomb_actionList.append((tag, action))
    if cur_Bomb_num >= 3:
        index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
        if index != -1:
            return index
    elif numofgreaterPos <= 18:
        index = choose_bomb(bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val)
        if index != -1:
            return index
    return 0

def passive( actionList, handcards, rank, curAction, greaterAction, myPos, greaterPos, remaincards,
            numofplayers, pass_num, my_pass_num):
    rank_card = 'H' + str(rank)
    restcards = rest_cards(handcards, remaincards, rank)

    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                      "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17, "JOKER": 10000, 'BJ':16, 'RJ':17}

    card_value_s2v[rank_card[1]] = 15

    actIndex = 0
    if curAction[0] == "PASS":
        curAction = greaterAction
    # _debug_print(curAction)
    numofmy = numofplayers[myPos]
    if numofmy <= 10:
        numofnext = numofplayers[(myPos + 1) % 4]
        actIndex = one_hand(numofmy, numofnext, actionList, myPos, greaterPos, 7,
                            restcards, card_value_s2v, rank_card)
        if actIndex != -1:
            return actIndex
    
    if curAction[0] == "Single":

        actIndex = Single(actionList, curAction, rank_card, handcards, numofplayers, restcards,
                               card_value_s2v, myPos, greaterPos, pass_num, my_pass_num)

    elif curAction[0] == "Pair":
        actIndex = Pair(actionList, curAction, rank_card, handcards, numofplayers, restcards,
                             card_value_s2v, myPos, greaterPos, pass_num, my_pass_num)

    elif curAction[0] == "Trips":
        actIndex = Trips(actionList, curAction, rank_card, handcards, numofplayers, restcards,
                              card_value_s2v, myPos, greaterPos, pass_num, my_pass_num)

    elif curAction[0] == "ThreeWithTwo":
        actIndex = ThreeWithTwo(actionList, curAction, rank_card, handcards, numofplayers, restcards,
                                     card_value_s2v, myPos, greaterPos, pass_num, my_pass_num)

    elif curAction[0] == "ThreePair":
        actIndex = ThreePair(actionList, curAction, rank_card, handcards, numofplayers, restcards,
                                  card_value_s2v, myPos, greaterPos, pass_num, my_pass_num)

    elif curAction[0] == "TwoTrips":
        actIndex = TwoTrips(actionList, curAction, rank_card, handcards, numofplayers, restcards,
                                 card_value_s2v, myPos, greaterPos, pass_num, my_pass_num)

    elif curAction[0] == "Straight":
        actIndex = Straight(actionList, curAction, rank_card, handcards, numofplayers, card_value_s2v, pass_num,
                                 my_pass_num, myPos, greaterPos)
    elif curAction[0] == "Bomb" or curAction[0] == "StraightFlush":
        actIndex = Bomb(actionList, curAction, rank_card, handcards, numofplayers, restcards,
                             card_value_s2v, myPos, greaterPos)

    return actIndex


'''utils.py'''

def is_inStraight(action, straight_member):

    flag = 0
    # _debug_print(straight_member)
    if len(straight_member) != 0:
        for card in action[2]:
            if card in straight_member:
                flag = 1
                break
    return flag

def combine_handcards(handcards, rank, card_val):
    cards = {}
    cards["Single"] = []
    cards["Pair"] = []
    cards["Trips"] = []
    cards["Bomb"] = []
    bomb_info = {}

    handcards = sorted(handcards, key=lambda item: card_val[item[1]])
    start = 0
    for i in range(1, len(handcards) + 1):
        if i == len(handcards) or handcards[i][-1] != handcards[i - 1][-1]:
            if (i - start == 1):
                cards["Single"].append(handcards[i - 1])
            elif (i - start == 2):
                cards["Pair"].append(handcards[start:i])
            elif (i - start) == 3:
                cards["Trips"].append(handcards[start:i])
            else:
                cards["Bomb"].append(handcards[start:i])
                bomb_info[handcards[start][-1]] = i - start
            start = i

    rank = rank
    temp = []
    for i in handcards:
        if i[-1] != rank and i[-1] != 'B' and i[-1] != 'R':
            temp.append(i)
    for i in cards['Bomb']:
        if i[0][-1] != rank and i[0][-1] != 'B' and i[0][-1] != 'R':
            for j in i:
                temp.remove(j)
    cardre = [0] * 14
    for i in temp:
        if i[-1] == 'A':
            cardre[1] += 1
        if i[-1] == '2':
            cardre[2] += 1
        if i[-1] == '3':
            cardre[3] += 1
        if i[-1] == '4':
            cardre[4] += 1
        if i[-1] == '5':
            cardre[5] += 1
        if i[-1] == '6':
            cardre[6] += 1
        if i[-1] == '7':
            cardre[7] += 1
        if i[-1] == '8':
            cardre[8] += 1
        if i[-1] == '9':
            cardre[9] += 1
        if i[-1] == 'T':
            cardre[10] += 1
        if i[-1] == 'J':
            cardre[11] += 1
        if i[-1] == 'Q':
            cardre[12] += 1
        if i[-1] == 'K':
            cardre[13] += 1

    st = []
    minnum = 10
    mintwonum = 10

    for i in range(1, len(cardre) - 4):
        if 0 not in cardre[i:i + 5]:
            onenum = 0
            zeronum = 0
            twonum = 0
            for j in cardre[i:i + 5]:
                if j - 1 == 0:
                    zeronum += 1
                if j - 1 == 1:
                    onenum += 1
                if j - 1 == 2:
                    twonum += 1

            # 1020
            if zeronum > onenum and minnum >= onenum:
                if len(st) == 0:
                    if zeronum >= onenum + twonum:
                        st.append(i)
                        minnum = onenum
                        mintwonum = twonum
                else:
                    if minnum == onenum:
                        if i == 1:
                            if mintwonum > twonum:
                                if zeronum >= onenum + twonum:
                                    st = []
                                    st.append(i)
                                    minnum = onenum
                                    mintwonum = twonum
                        else:
                            if mintwonum >= twonum:
                                if zeronum >= onenum + twonum:
                                    st = []
                                    st.append(i)
                                    minnum = onenum
                                    mintwonum = twonum
                    else:
                        if zeronum >= onenum + twonum:
                            st = []
                            st.append(i)
                            minnum = onenum
                            mintwonum = twonum

    if 0 not in cardre[10:] and cardre[1] != 0:
        onenum = 0
        zeronum = 0
        twonum = 0
        for j in cardre[10:]:
            if j - 1 == 0:
                zeronum += 1
            if j - 1 == 1:
                onenum += 1
            if j - 1 == 2:
                twonum += 1
        if cardre[1] - 1 == 0:
            zeronum += 1
        if cardre[1] - 1 == 1:
            onenum += 1
        if cardre[1] - 1 == 2:
            twonum += 1

        if zeronum > onenum and minnum >= onenum:
            if len(st) == 0:

                if zeronum >= onenum + twonum:
                    st.append(10)
                    minnum = onenum
                    mintwonum = twonum
            else:

                if minnum == onenum:
                    if mintwonum >= twonum:
                        if zeronum >= onenum + twonum:
                            st = []
                            st.append(10)
                            minnum = onenum
                            mintwonum = twonum
                else:
                    if zeronum >= onenum + twonum:
                        st = []
                        st.append(10)
                        minnum = onenum
                        mintwonum = twonum

    tmp = []
    Flushtmp = []
    nowhandcards = []
    Straight = []
    if len(st) > 0:
        for i in range(st[0], st[0] + 5):
            if 1 < i < 10:
                Straight.append(str(i))
            if i % 13 == 1:
                Straight.append('A')
            if i == 10:
                Straight.append('T')
            if i == 11:
                Straight.append('J')
            if i == 12:
                Straight.append('Q')
            if i == 13:
                Straight.append('K')
    sttemp = []
    for i in range(4):
        sttemp.append([0] * 5)
    counttemp = 0

    colortemp = {"S": 0, "H": 1, "C": 2, "D": 3}
    rev_colortemp = {0: 'S', 1: 'H', 2: 'C', 3: 'D'}
    for i in range(0, len(handcards) - 1):
        if handcards[i][-1] in Straight:
            sttemp[colortemp[handcards[i][0]]][counttemp] += 1
            if handcards[i][-1] != handcards[i + 1][-1]:
                counttemp += 1

    StraightFlushflag = -1

    for i in range(4):
        if sttemp[i][0] > 0 and sttemp[i][1] > 0 and sttemp[i][2] > 0 and sttemp[i][3] > 0 and sttemp[i][4] > 0:
            StraightFlushflag = i
    if StraightFlushflag >= 0:
        for i in Straight:
            Flushtmp.append(rev_colortemp[StraightFlushflag] + i)
        for i in range(0, len(handcards)):
            if handcards[i] not in Flushtmp:
                nowhandcards.append(handcards[i])

    else:
        for i in range(0, len(handcards)):
            if handcards[i][-1] in Straight:
                tmp.append(handcards[i])
                Straight.remove(handcards[i][-1])
            else:
                nowhandcards.append(handcards[i])

    newcards = {}
    newcards["Single"] = []
    newcards["Pair"] = []
    newcards["Trips"] = []
    newcards["Bomb"] = []
    newcards['Straight'] = []
    newcards['StraightFlush'] = []

    if len(tmp) == 5:
        if tmp[-1][-1] == 'A' and tmp[-2][-1] == '5':
            tmpptmp = [tmp[-1]]
            for kkk in tmp[:-1]:
                tmpptmp.append(kkk)
            newcards['Straight'].append(tmpptmp)
        else:
            newcards['Straight'].append(tmp)
    if len(Flushtmp) == 5:
        newcards['StraightFlush'].append(Flushtmp)
    start = 0
    for i in range(1, len(nowhandcards) + 1):
        if i == len(nowhandcards) or nowhandcards[i][-1] != nowhandcards[i - 1][-1]:
            if (i - start == 1):
                newcards["Single"].append(nowhandcards[i - 1])
            elif (i - start == 2):
                newcards["Pair"].append(nowhandcards[start:i])
            elif (i - start) == 3:
                newcards["Trips"].append(nowhandcards[start:i])
            else:
                newcards["Bomb"].append(nowhandcards[start:i])
            start = i

    # return cards
    return newcards, bomb_info

def rest_cards(handcards,remaincards,rank):

    card_value_v2s = {0: "A", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8", 8: "9", 9: "T", 10: "J",
                      11: "Q", 12: "K"}
    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                      "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}

    card_index = {"A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "T": 9, "J": 10,
                  "Q": 11, "K": 12, "R": 13, "B": 13}
    new_remaincards = {}
    for key,val in remaincards.items():
        new_remaincards[key] = copy.deepcopy(val)
    for card in handcards:
        card_type = str(card[0])
        x = card_index[card[1]]
        new_remaincards[card_type][x] = remaincards[card_type][x]-1

    rest_cards = []

    for key,value in new_remaincards.items():
        for i  in range(0,len(value)):
            if value[i] ==0 :
                continue
            if i == 13 and key == 'S':
                val = 'B'
            elif i == 13 and key == 'H':
                val = 'R'
            else:
                val = card_value_v2s[i]
            if value[i]==1:
                rest_cards.append(key+val)
            elif value[i] == 2:
                rest_cards.append(key + val)
                rest_cards.append(key + val)
    
    card_value_s2v[str(rank)] = 15
    rest_cards = sorted(rest_cards,key = lambda item:card_value_s2v[item[1]])
    new_rest_cards = []
    tmp = []
    pre = rest_cards[0]
    tmp = [pre]
    for cards in rest_cards[1:]:
        if cards[1]!=pre[1]:
            new_rest_cards.append(tmp)
            tmp = [cards]
            pre = cards
        else:
            tmp.append(cards)
    new_rest_cards.append(tmp)
    return new_rest_cards

def choose_bomb( bomb_actionList, handcards, sorted_cards, bomb_info, rank_card, card_val):
    new_card_val = copy.deepcopy(card_val)
    new_card_val['A'] = 14
    new_card_val[rank_card[1]] = 15
    bomb_res = []

    new_card_val["JOKER"] = 10000
    straight_member = []
    if len(sorted_cards["Straight"]) != 0:
        straight_member += sorted_cards["Straight"][0]
    if len(sorted_cards["StraightFlush"]) != 0:
        straight_member += sorted_cards["StraightFlush"][0]

    for action in bomb_actionList:

        index = action[0]
        action = action[1]
        if action[0] == "Bomb":
            if action[1]==rank_card[1]:
                #级牌炸特别考虑
                prior = 0
                rank_card_num = 0
                for card in action[2]:
                    if card == rank_card:
                        rank_card_num += 1
                if rank_card_num == 1:
                    prior = 3
                elif rank_card_num == 2:
                    prior = 16
                l = len(action[2])
                bomb_res.append((index, new_card_val[action[1]] + (l - 4) * 16+prior))
            else:
                if action[1] in bomb_info:
                    if bomb_info[action[1]] == len(action[2]) and rank_card not in action[2]:
                        #纯炸的情况
                        l = len(action[2])
                        bomb_res.append((index, new_card_val[action[1]] + (l - 4) * 16))
                    elif len(sorted_cards["Trips"])==0:
                        if len(action[2])>bomb_info[action[1]] and rank_card in action[2]:
                            l = len(action[2])
                            rank_card_num = len(action[2]) - bomb_info[action[1]]
                            prior = 0
                            if rank_card_num == 1:
                                prior = 3
                            elif rank_card_num == 2:
                                prior = 16
                            bomb_res.append((index, new_card_val[action[1]] + (l - 4) * 16+prior))

                elif action[1] not in bomb_info and rank_card in action[2]:
                    #红桃配三个成炸 或者两个红桃配对子成炸
                    if is_inStraight(action, straight_member):
                        continue
                    prior = 0
                    rank_card_num = 0
                    for card in action[2]:
                        if card == rank_card:
                            rank_card_num += 1
                    if rank_card_num == 1:
                        prior = 3
                    elif rank_card_num == 2:
                        prior = 16
                    l = len(action[2])
                    bomb_res.append((index, new_card_val[action[1]] + (l - 4) * 16 + prior))
        elif action[0] == "StraightFlush":  # 出不出同花顺
            if len(sorted_cards["StraightFlush"]) > 0:
                curStraight = sorted_cards["StraightFlush"][0][0][1]
                if curStraight == action[1] and rank_card not in action[2]:
                    bomb_res.append((index, new_card_val[action[1]] + 32))

    if len(bomb_res) == 0:
        return -1
    else:
        bomb_res = sorted(bomb_res, key=lambda item: item[1])
        return bomb_res[0][0]

def one_hand(numofmy,numofnext,actionList,myPos,greaterPos,cards_num,restcards,card_val,rank_card):

    max_bomb = 0
    rank_card_num = 0
    for cards in restcards:
        if rank_card in cards:
            rank_card_num+=1
    for cards in restcards:
        if cards[0][1]==rank_card[1] and len(cards)>=4:
            l = len(cards)
            max_bomb = max(max_bomb,card_val[cards[0][1]]+(l-4)*14)
        elif cards[0][1]!=rank_card[1] and len(cards)>=4:
            l = len(cards)
            max_bomb = max(max_bomb, card_val[cards[0][1]] +(l+rank_card_num-4)*14)
        elif cards[0][1]!=rank_card[1] and len(cards)==3 and rank_card_num>=1:
            max_bomb = max(max_bomb, card_val[cards[0][1]] + (rank_card_num-1)*14)
        elif cards[0][1] != rank_card[1] and len(cards) == 2 and rank_card_num == 2:
            max_bomb = max(max_bomb, card_val[cards[0][1]])

    tag = 0
    if (myPos+2)%4 != greaterPos:#当前最大动作为非对家发出
        for action in actionList[1:]:
            tag +=1
            if numofmy == len(action[2]):
                return tag
    else:#当前动作为对家所出
        for action in actionList[1:]:
            tag += 1
            if action[0]!="Bomb" and action[0]!="StraightFlush" and numofmy == len(action[2]):
                return tag
            if (action[0]=="Bomb" or action[0]=="StraightFlush") and numofmy == len(action[2]):

                if action[0]=="Bomb":
                    l = len(action[2])
                    cur_level = card_val[action[1]]+(l-4)*14
                else:
                    cur_level = card_val[action[1]] + 14
                if numofnext>cards_num and cur_level>max_bomb:
                    return 0
                else:
                    return tag

    return -1

def cal_bomb_num(sorted_cards,handcards,rank_card):
    cur_Bomb_num = len(sorted_cards["Bomb"]) + len(sorted_cards["StraightFlush"])  # 没有考虑级牌在炸或者同花顺中
    rank_card_num = 0
    for card in handcards:
        if card == rank_card:
            rank_card_num += 1

    if rank_card_num == 1:
        for trip in sorted_cards["Trips"]:
            if rank_card not in trip:
                cur_Bomb_num += 1
                break
    if rank_card_num == 2:
        if len(sorted_cards["Trips"])==1 and rank_card not in sorted_cards["Trips"][0]:
            cur_Bomb_num += 1
        elif len(sorted_cards["Trips"])==2 and rank_card in sorted_cards["Trips"][1]:
            cur_Bomb_num += 1
        elif len(sorted_cards["Trips"])==2 and rank_card not in sorted_cards["Trips"][1]:
            cur_Bomb_num += 2
        elif len(sorted_cards["Trips"])>=2: #计算写错
            cur_Bomb_num += 2

    for bomb in sorted_cards["Bomb"]:
        if rank_card in bomb:
            cur_Bomb_num -= 1

    return cur_Bomb_num

def combine_ThreePair(handcards,rank_card,sorted_cards,card_val):
    card_origin = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                   "Q": 12, "K": 13}
    card_val['A'] = 1
    card_val[rank_card[1]] = card_origin[rank_card[1]]
    Pairs = {}
    Trips = {}
    for pair in sorted_cards["Pair"]:
        Pairs[card_val[pair[0][1]]] = pair
    for trips in sorted_cards["Trips"]:
        Trips[card_val[trips[0][1]]] = trips

    for key,val in card_origin.items():
        if val >12 or val == 1:
            continue

#计算牌距2
def caldistance2(trips_actionlist,pair_actionlist,rank):
    rank_card = 'H' + str(rank)
    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                      "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
    card_value_s2v[rank_card[-1]] = 15
    return card_value_s2v[trips_actionlist[0][0]],card_value_s2v[trips_actionlist[0][0]] - card_value_s2v[pair_actionlist[1][0]],card_value_s2v[pair_actionlist[1][0]],card_value_s2v[pair_actionlist[0][0]]

#计算牌距3
def caldistance3(trips_actionlist,pair_actionlist,rank):
    rank_card = 'H' + str(rank)
    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                      "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
    card_value_s2v[rank_card[-1]] = 15
    return card_value_s2v[pair_actionlist[0][0]],card_value_s2v[trips_actionlist[0][0]] - card_value_s2v[pair_actionlist[0][0]],card_value_s2v[trips_actionlist[1][0]],card_value_s2v[trips_actionlist[0][0]]

#获得动作索引
def getindex(tag, actList, actionList):
    myaction = tag
    mynumber = actList[0][0]
    mycard = "None"
    if myaction == "Single":
        mycard = [actList[0][1]]
    else:
        mycard = actList[0][1]
    my_act = []
    my_act.append(myaction)
    my_act.append(mynumber)
    my_act.append(mycard)
    # _debug_print(my_act)
    if my_act in actionList:
        return actionList.index(my_act)
    else:
        return 0

#获得第二小的下标
def getindex1(tag, actList, actionList):
    myaction = tag
    mynumber = actList[1][0]
    mycard = "None"
    if myaction == "Single":
        mycard = [actList[1][1]]
    else:
        mycard = actList[1][1]
    my_act = []
    my_act.append(myaction)
    my_act.append(mynumber)
    my_act.append(mycard)
    # _debug_print(my_act)
    if my_act in actionList:
        return actionList.index(my_act)
    else:
        return 0

#等级四（连三，连对至少存在一个）
def rankfour(twotrips_actionlist,threepair_actionlist,actionList,cur2,cur3):#cur2是连对，cur3是连三

    card_value_s2v2 = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                       "Q": 12, "K": 13, "B": 16, "R": 17}
    minvalue = [100,100]

    #0是连对，1是连三
    if len(threepair_actionlist):
        minvalue[0] = card_value_s2v2[threepair_actionlist[0][0]]

    if len(twotrips_actionlist):
        minvalue[1] = card_value_s2v2[twotrips_actionlist[0][0]]

    #找到最小值的位置
    minpos = minvalue.index(min(minvalue))
    #映射类型
    if minpos == 0 and minvalue[0]<=cur2:
        return getindex("ThreePair",threepair_actionlist,actionList)

    if minpos == 1 and minvalue[1]<=cur3:
        return getindex("TwoTrips",twotrips_actionlist,actionList)

#等级三，考虑三带二，考虑有压的情况下
def rankthree(single_actionlist,pair_actionlist,trips_actionlist,threetwo_actionlist,actionList,numofnext,rank,cur1,cur4,cur5,cur6,curp2):
    #cur4是三带二，cur5是对子，cur6是三个，curp2是牌的距离
    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                      "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
    card_value_s2v[rank] = 15
    # 此时出三带二需要符合长度条件
    if len(pair_actionlist) == len(trips_actionlist) or (
            len(pair_actionlist) >= 2 and len(trips_actionlist) >= 2):
        if card_value_s2v[threetwo_actionlist[0][0]] < cur4 or numofnext==1:
            #出最小的三带二
            return getindex("ThreeWithTwo",threetwo_actionlist,actionList)
        elif len(single_actionlist) and card_value_s2v[single_actionlist[0][0]] < cur1 and numofnext==5:
            return getindex("Single",single_actionlist,actionList)
        else:
            #看环境（补）
            #有压
            #一个有压
            minvalue = [100, 100]
            # 0是三带二，1是单张
            if len(threetwo_actionlist):
                minvalue[0] = card_value_s2v[threetwo_actionlist[0][0]]

            if len(single_actionlist):
                minvalue[1] = card_value_s2v[single_actionlist[0][0]]

            if len(threetwo_actionlist)>1 and len(single_actionlist) == 1:#三带二有压
                minvalue[0] = minvalue[0] + 1
                # 找到最小值的位置
                minpos = minvalue.index(min(minvalue))
                # 映射类型
                if minpos == 0 :
                    return getindex("ThreeWithTwo", threetwo_actionlist, actionList)
                if minpos == 1 :
                    return getindex("Single", single_actionlist, actionList)
            elif len(threetwo_actionlist) == 1 and len(single_actionlist)>1:#单张有压
                minvalue[1] = minvalue[1] + 1
                # 找到最小值的位置
                minpos = minvalue.index(min(minvalue))
                # 映射类型
                if minpos == 0 :
                    return getindex("ThreeWithTwo", threetwo_actionlist, actionList)
                if minpos == 1 :
                    return getindex("Single", single_actionlist, actionList)
            else:
                #两个都有压或者都无压的情况下，直接出最小值
                # 找到最小值的位置
                minpos = minvalue.index(min(minvalue))
                # 映射类型
                if minpos == 0:
                    return getindex("ThreeWithTwo", threetwo_actionlist, actionList)
                if minpos == 1:
                    return getindex("Single", single_actionlist, actionList)

    #当不符合三带二的条件时，考虑对子和三个
    else:
        len3 = len(pair_actionlist) #对子个数
        len4 = len(trips_actionlist) #三个个数

        if numofnext == 3 and len(pair_actionlist) and card_value_s2v[pair_actionlist[0][0]] < cur4:
            return getindex("Pair",pair_actionlist,actionList)
        if numofnext == 2 and len(trips_actionlist) and card_value_s2v[trips_actionlist[0][0]] < cur5:
            return getindex("ThreeWithTwo",threetwo_actionlist,actionList)

        if len3 > len4:#只有一个三个
            mint,_,minp2,minp1= caldistance2(trips_actionlist,pair_actionlist,rank)
            if minp2 <= cur5 and mint <= cur6:
                if minp1 > mint:
                    return getindex("ThreeWithTwo",threetwo_actionlist,actionList)
                else:return getindex1("Pair",pair_actionlist,actionList)
            elif minp2 > cur5 and mint <= cur6:
                if minp2 > cur5 + curp2:
                    if minp1 < cur5:
                        return getindex("Pair",pair_actionlist,actionList)
                    else:
                        return getindex("ThreeWithTwo",threetwo_actionlist,actionList)
                else:
                    return getindex("ThreeWithTwo",threetwo_actionlist,actionList)
            elif minp2 <= cur5 and mint > cur6:
                return getindex("Pair",pair_actionlist,actionList)
            else:
                return getindex("ThreeWithTwo",threetwo_actionlist,actionList)

        else:
            minp, _, mint2, mint1 = caldistance3(trips_actionlist, pair_actionlist, rank)
            if minp <= cur5:
                return getindex("ThreeWithTwo",threetwo_actionlist,actionList)
            elif mint1 < cur6 and minp > cur5:
                return getindex("Trips",trips_actionlist,actionList)
            else:
                if minp >= cur5 + curp2:
                    return getindex("Trips",trips_actionlist,actionList)
                else:
                    return getindex("ThreeWithTwo",threetwo_actionlist,actionList)

#等级二，还剩对子的时候
def ranktwo(hand_cards,single_actionlist,pair_actionlist,trips_actionlist,actionList,numofnext,rank,max_val):
    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                      "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
    card_value_s2v[rank] = 15
    rank_card = 'H'+rank
    if len(single_actionlist):
        # 如果红桃级牌在列表中
        # if rank_card in hand_cards and len(trips_actionlist)==0:
        #     for i in range(len(actionList)):
        #         if actionList[i][0]=='ThreePair' and rank_card in actionList[i][-1]:
        #             if card_value_s2v[actionList[i][1]] > card_value_s2v[single_actionlist[0][0]] and numofnext > 1 and card_value_s2v[single_actionlist[-1][0]] == max_val:
        #                 return getindex("Single",single_actionlist,actionList)
        #             else:
        #                 return i
        #如果下家只剩一张牌
        if numofnext == 1:
            return getindex("Pair", pair_actionlist, actionList)
        if numofnext == 2:
            return getindex("Single",single_actionlist,actionList)
        else:
            # 一个有压
            minvalue = [100, 100]
            # 0是对子，1是单张
            if len(pair_actionlist):
                minvalue[0] = card_value_s2v[pair_actionlist[0][0]]

            if len(single_actionlist):
                minvalue[1] = card_value_s2v[single_actionlist[0][0]]

            if len(pair_actionlist) > 1 and len(single_actionlist) == 1:  # 三带二有压
                minvalue[0] = minvalue[0] + 1
                # 找到最小值的位置
                minpos = minvalue.index(min(minvalue))
                # 映射类型
                if minpos == 0:
                    return getindex("Pair", pair_actionlist, actionList)
                if minpos == 1:
                    return getindex("Single", single_actionlist, actionList)
            elif len(pair_actionlist) == 1 and len(single_actionlist) > 1:  # 单张有压
                minvalue[1] = minvalue[1] + 1
                # 找到最小值的位置
                minpos = minvalue.index(min(minvalue))
                # 映射类型
                if minpos == 0:
                    return getindex("Pair", pair_actionlist, actionList)
                if minpos == 1:
                    return getindex("Single", single_actionlist, actionList)
            else:
                # 两个都有压或者都无压的情况下，直接出最小值
                # 找到最小值的位置
                minpos = minvalue.index(min(minvalue))
                # 映射类型
                if minpos == 0:
                    return getindex("Pair", pair_actionlist, actionList)
                if minpos == 1:
                    return getindex("Single", single_actionlist, actionList)
    else:
        return getindex("Pair", pair_actionlist, actionList)

def rankone(single_actionlist,trips_actionlist,actionList,numofnext,rank):
    card_value_s2v = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                      "Q": 12, "K": 13, "A": 14, "B": 16, "R": 17}
    card_value_s2v[rank] = 15
    if len(single_actionlist):
        # 如果下家只剩一张牌
        if numofnext == 1:
            return getindex("Trips", trips_actionlist, actionList)
        if numofnext == 3:
            return getindex("Single", single_actionlist, actionList)
        else:
            # 一个有压
            minvalue = [100, 100]
            # 0是三个，1是单张
            if len(trips_actionlist):
                minvalue[0] = card_value_s2v[trips_actionlist[0][0]]

            if len(single_actionlist):
                minvalue[1] = card_value_s2v[single_actionlist[0][0]]

            if len(trips_actionlist) > 1 and len(single_actionlist) == 1:  # 三带二有压
                minvalue[0] = minvalue[0] + 1
                # 找到最小值的位置
                minpos = minvalue.index(min(minvalue))
                # 映射类型
                if minpos == 0:
                    return getindex("Trips", trips_actionlist, actionList)
                if minpos == 1:
                    return getindex("Single", single_actionlist, actionList)
            elif len(trips_actionlist) == 1 and len(single_actionlist) > 1:  # 单张有压
                minvalue[1] = minvalue[1] + 1
                # 找到最小值的位置
                minpos = minvalue.index(min(minvalue))
                # 映射类型
                if minpos == 0:
                    return getindex("Trips", trips_actionlist, actionList)
                if minpos == 1:
                    return getindex("Single", single_actionlist, actionList)
            else:
                # 两个都有压或者都无压的情况下，直接出最小值
                # 找到最小值的位置
                minpos = minvalue.index(min(minvalue))
                # 映射类型
                if minpos == 0:
                    return getindex("Trips", trips_actionlist, actionList)
                if minpos == 1:
                    return getindex("Single", single_actionlist, actionList)
    else:
        return getindex("Trips", trips_actionlist, actionList)
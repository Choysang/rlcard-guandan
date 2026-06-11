# -*- coding: utf-8 -*-
def _debug_print(*args, **kwargs):
    # Debug output from the original research code, silenced for
    # the open-source release.
    pass


# @Time       : 2020/10/1 21:32
# @Author     : Duofeng Wu
# @File       : action.py
# @Description: 动作类

from random import randint
import numpy as np
import json
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
        self.shoupaijuzhen = np.zeros(shape=(4,16),dtype='i1')

    def opened(self):
        pass

    def closed(self, code, reason=None):
        _debug_print("Closed down", code, reason)

    def parse(self, msg, mypos):
        self.action = msg["actionList"]
        self.act_range = msg["indexRange"]
        # _debug_print(self.action)

        if "actionList" in msg:         # 需要做出动作选择时调用动作对象进行解析
            rank = self.handsort(msg)
            # 同花顺
            #straight_flush, flush_i, flush_j = self.tonghuashun()
            # 通配牌同花顺
            #straight_tong_flush, flush_tong_i, flush_tong_j = self.straight_tong_flush(rank)
            #straight = straight_tong_flush + straight_flush
            straight, flush_i, flush_j = self.tonghuashun(msg["actionList"])
            # 炸弹
            bomb, bomb_i, bomb_j = self.zhadan(msg["actionList"])

            zongzha = bomb + straight
            # 顺子
            Straight, straight_i, hangwei, liewei = self.shunzi(msg["actionList"])
            # 钢板
            TwoTrips, TwoTrips_i, TwoTrips_j = self.gangban(msg["actionList"])
            # 三对
            ThreePair, ThreePair_i, ThreePair_j = self.sandui(msg["actionList"])
            # 三张
            Trips, Trips_i, Trips_j = self.trips(msg["actionList"])
            # 对子
            Pair, Pair_, Pair_j = self.duizi(msg["actionList"])
            # 单牌
            Single, Single_, Single_j = self.danpai(msg["actionList"])
            # ThreeWithTwo
            ThreeWithTwo, ThreeWithTwo_i, ThreeWithTwo_j = self.sandaier(msg["actionList"])

            actionlist = msg["actionList"]

            action_dict = {
                2: '2', 3: '3', 4: '4', 5: '5',
                6: '6', 7: '7', 8: '8', 9: '9',
                10: 'T', 11: 'J', 12: 'Q', 13: 'K',
                14: 'A', 16: 'B', 17: 'R'
            }
            # 修正级牌牌力
            action_dict[15] = CARD_RANK[rank]

            hua_dict = {0:'S',1:'H',2:'C',3:'D'}

            #_debug_print('我的位置是：', mypos)
            public = msg.get("publicInfo")
            #_debug_print('当前公共剩余牌和打牌信息', public)
            rest_me = public[mypos]['rest']
            greatePos = msg["greaterPos"]
            #_debug_print('最大动作位置是：',greatePos)
            greateacion = msg["greaterAction"]
            #_debug_print('当前最大牌信息', greateacion)
            cha = abs(greatePos - mypos) if greatePos is not None else -1
            row_dict = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14,'B':16,'R':17}

            # 自己先出牌
            actionlist = msg["actionList"]
            if msg["greaterPos"] == mypos or msg["greaterPos"] == -1:
                if Straight > 0:
                    for i in range(len(actionlist)):
                        if actionlist[i][0] == 'Straight' and actionlist[i][1] == action_dict[straight_i[0]+2]:
                            #_debug_print('自己先出顺子的行和列分别为', hangwei, liewei)
                            if actionlist[i][2][0][0] == hua_dict[hangwei[0]] and actionlist[i][2][0][1] == action_dict[liewei[0]+2] and actionlist[i][2][1][0] == hua_dict[hangwei[1]] and actionlist[i][2][1][1] == action_dict[liewei[1]+2] and  actionlist[i][2][2][0] == hua_dict[hangwei[2]] and actionlist[i][2][2][1] == action_dict[liewei[2]+2] and  actionlist[i][2][3][0] == hua_dict[hangwei[3]] and actionlist[i][2][3][1] == action_dict[liewei[3]+2] and actionlist[i][2][4][0] == hua_dict[hangwei[4]] and actionlist[i][2][4][1] == action_dict[liewei[4]+2]:
                                act_index = i
                                #_debug_print('自己先出顺子,剩余通配牌数量是', self.shoupaijuzhen[1][rank-2])
                                Straight -= 1
                                return act_index
                                
                # HR
                elif self.shoupaijuzhen[1][15] > 0 and Single > 0:
                    for j in range(len(Single_)):
                        for i in range(len(actionlist)):
                            if actionlist[i][0] == 'Single' and actionlist[i][1] == action_dict[Single_[j]+2] and actionlist[i][2][0][0] == hua_dict[Single_j[j]]:
                                act_index = i
                                #_debug_print('有大王自己先出单牌')
                                Single -= 1
                                return act_index
                                
                elif ThreePair > 0:
                    for j in range(len(ThreePair_i)-5):
                        for i in range(len(actionlist)):
                            if ThreePair_i[j] < 7 or rest_me < 13:
                                if actionlist[i][0] == 'ThreePair' and actionlist[i][1] == action_dict[ThreePair_i[0]+2] and actionlist[i][2][0][0] == hua_dict[ThreePair_j[j]] and actionlist[i][2][1][0] == hua_dict[ThreePair_j[j+1]] and actionlist[i][2][2][0] == hua_dict[ThreePair_j[j+2]] and actionlist[i][2][3][0] == hua_dict[ThreePair_j[j+3]] and actionlist[i][2][4][0] == hua_dict[ThreePair_j[j+4]] and actionlist[i][2][5][0] == hua_dict[ThreePair_j[j+5]]:
                                    act_index = i
                                    #_debug_print('自己先出三对')
                                    ThreePair -= 1
                                    return act_index
                                    
                elif TwoTrips > 0:
                    for j in range(len(TwoTrips_i)-5):
                        for i in range(len(actionlist)):
                            if TwoTrips_i[j] < 7 or rest_me < 13:
                                if actionlist[i][0] == 'TwoTrips' and actionlist[i][1] == action_dict[TwoTrips_i[0]+2] and actionlist[i][2][0][0] == hua_dict[TwoTrips_j[j]] and actionlist[i][2][1][0] == hua_dict[TwoTrips_j[j+1]] and actionlist[i][2][2][0] == hua_dict[TwoTrips_j[j+2]] and actionlist[i][2][3][0] == hua_dict[TwoTrips_j[j+3]] and actionlist[i][2][4][0] == hua_dict[TwoTrips_j[j+4]] and actionlist[i][2][5][0] == hua_dict[TwoTrips_j[j+5]]:
                                    act_index = i
                                    #_debug_print('自己先出钢板')
                                    TwoTrips -= 1
                                    return act_index
                                    
                elif (ThreeWithTwo > 0 and Pair_[0]+2 < 11) or (ThreeWithTwo > 0 and rest_me < 12):
                    if Trips_i[0] < 8:
                        for j in range(len(Trips_j)-2):
                            for i in range(len(actionlist)):
                                if actionlist[i][0] == 'ThreeWithTwo' and actionlist[i][1] == action_dict[Trips_i[j]+2] and actionlist[i][2][-1][-1] == action_dict[Pair_[0]+2]:
                                    if actionlist[i][2][0][0] == hua_dict[Trips_j[j]] and actionlist[i][2][1][0] == hua_dict[Trips_j[j+1]] and actionlist[i][2][2][0] == hua_dict[Trips_j[j+2]]:
                                        for x in range(len(Pair_j)-1):
                                            if actionlist[i][2][3][0] == hua_dict[Pair_j[x]] and actionlist[i][2][4][0] == hua_dict[Pair_j[x + 1]]:
                                                act_index = i
                                                #_debug_print('自己先出三带二')
                                                Trips -= 1
                                                Pair -= 1
                                                ThreeWithTwo = Trips * Pair
                                                return act_index
                                                
                elif Trips > 0:
                    for j in range(len(Trips_j)-2):
                        for i in range(len(actionlist)):
                            if Trips_i[j] < 8 or rest_me < 10:
                                if actionlist[i][0] == 'Trips' and actionlist[i][1] == action_dict[Trips_i[j]+2]:
                                    if actionlist[i][2][0][0] == hua_dict[Trips_j[j]] and actionlist[i][2][1][0] == hua_dict[Trips_j[j+1]] and actionlist[i][2][2][0] == hua_dict[Trips_j[j+2]]:
                                        act_index = i
                                        #_debug_print('自己先出三张')
                                        Trips -= 1
                                        return act_index
                                    
                elif Pair > 0:
                    for j in range(len(Pair_j)-1):
                        for i in range(len(actionlist)):
                            if Pair_j[j] < 7 or rest_me < 9:
                                if actionlist[i][0] == 'Pair' and actionlist[i][1] == action_dict[Pair_[j]+2]:
                                    if actionlist[i][2][0][0] == hua_dict[Pair_j[j]] and actionlist[i][2][1][0] == hua_dict[Pair_j[j+1]]:
                                        act_index = i
                                        #_debug_print('自己先出对子')
                                        Pair -= 1
                                        return act_index
                                        
                elif Single > 0:
                    for j in range(len(Single_)):
                        for i in range(len(actionlist)):
                            if (Single_[j] != 14 and Single_[j] != 15) or rest_me < 8:
                                if actionlist[i][0] == 'Single' and actionlist[i][1] == action_dict[Single_[j]+2] and actionlist[i][2][0][0] == hua_dict[Single_j[j]]:
                                    act_index = i
                                    #_debug_print('自己先出单牌')
                                    Single -= 1
                                    return act_index
                                    
                elif zongzha > 0:
                    for i in range(len(actionlist)):
                        if actionlist[i][0] == 'Bomb' and rest_me < 10:
                            act_index = i
                            #_debug_print('其他牌型都没有自己先出炸弹')
                            Single -= 1
                            return act_index
                            
                else:
                    #_debug_print('没有自己先出的牌，随机打出')
                    self.act_range = msg["indexRange"]
                    return randint(0, self.act_range)
                    #return act_index
                    

            # 当前队友牌型最大 +++判断单牌和对子是否小于11
            elif cha == 2:
                if greateacion[0] == 'Single' and row_dict[greateacion[1]] < 11 and row_dict[greateacion[1]] != rank and Single > 0:
                    for i in range(len(Single_)):
                        if Single_[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                for x in range(len(Single_j)):
                                    if actionlist[j][0] == 'Single' and actionlist[j][1] == action_dict[Single_[i]+2] and actionlist[j][2][0][0] == hua_dict[Single_j[x]]:
                                        act_index = j
                                        # _debug_print('队友出单牌最大出单牌')
                                        Single -= 1
                                        return act_index
                                        
                elif greateacion[0] == 'Pair' and Pair > 0 and row_dict[greateacion[1]] < 11 and row_dict[greateacion[1]] != rank:
                    for i in range(len(Pair_)-1):
                        if Pair_[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Pair' and actionlist[j][1] == action_dict[Pair_[i]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[Pair_j[i]] and actionlist[j][2][1][0] == hua_dict[Pair_j[i+1]]:
                                        act_index = j
                                        # _debug_print('队友出对子最大出对子')
                                        Pair -= 1
                                        return act_index
                                        
                else:
                    # _debug_print('队友玩家牌最大，pass')
                    return 0
            # 当前局面对手牌最大
            else:
                if (greateacion[0] == 'StraightFlush' or greateacion[0] == 'Bomb') and zongzha > 0:
                    if len(actionlist) > 1:
                        act_index = 1
                        # _debug_print('对手出炸弹或同花顺最大出炸或同花顺')
                        zongzha -= 1
                        return act_index
                        
                elif greateacion[0] == 'Straight' and Straight > 0:
                    for i in range(len(straight_i)):
                        if straight_i[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Straight' and actionlist[j][1] == action_dict[straight_i[i]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[hangwei[i*4+1]] and actionlist[j][2][0][1] == action_dict[liewei[i*4+1]+2] and actionlist[j][2][1][0] == hua_dict[hangwei[i*4+2]] and actionlist[j][2][1][1] == action_dict[liewei[i*4+2]+2] and  actionlist[j][2][2][0] == hua_dict[hangwei[i*4+3]] and actionlist[j][2][2][1] == action_dict[liewei[i*4+3]+2] and  actionlist[j][2][3][0] == hua_dict[hangwei[i*4+4]] and actionlist[j][2][3][1] == action_dict[liewei[i*4+4]+2] and actionlist[j][2][4][0] == hua_dict[hangwei[i*4+5]] and actionlist[j][2][4][1] == action_dict[liewei[i*4+5]+2]:
                                        act_index = j
                                        # _debug_print('对手出顺子最大出顺子')
                                        Straight -= 1
                                        return act_index
                                        
                elif greateacion[0] == 'ThreePair' and ThreePair > 0:
                    for j in range(len(ThreePair_i)-5):
                        if ThreePair_i[j]+2 > row_dict[greateacion[1]]:
                            for i in range(len(actionlist)):
                                if actionlist[i][0] == 'ThreePair' and actionlist[i][1] == action_dict[ThreePair_i[j]+2]  and actionlist[i][2][0][0] == hua_dict[ThreePair_j[j]] and actionlist[i][2][1][0] == hua_dict[ThreePair_j[j+1]] and actionlist[i][2][2][0] == hua_dict[ThreePair_j[j+2]] and actionlist[i][2][3][0] == hua_dict[ThreePair_j[j+3]] and actionlist[i][2][4][0] == hua_dict[ThreePair_j[j+4]] and actionlist[i][2][5][0] == hua_dict[ThreePair_j[j+5]]:
                                    act_index = i
                                    # _debug_print('队手出三连队最大出三连队')
                                    ThreePair -= 1
                                    return act_index
                                    
                elif greateacion[0] == 'TwoTrips' and TwoTrips > 0:
                    for j in range(len(TwoTrips_i)-5):
                        if TwoTrips_i[j]+2 > row_dict[greateacion[1]]:
                            for i in range(len(actionlist)):
                                if actionlist[i][0] == 'TwoTrips' and actionlist[i][1] == action_dict[TwoTrips_i[j]+2] and actionlist[i][2][0][0] == hua_dict[TwoTrips_j[j]] and actionlist[i][2][1][0] == hua_dict[TwoTrips_j[j+1]] and actionlist[i][2][2][0] == hua_dict[TwoTrips_j[j+2]] and actionlist[i][2][3][0] == hua_dict[TwoTrips_j[j+3]] and actionlist[i][2][4][0] == hua_dict[TwoTrips_j[j+4]] and actionlist[i][2][5][0] == hua_dict[TwoTrips_j[j+5]]:
                                    act_index = i
                                    # _debug_print('对手出钢板最大出钢板')
                                    TwoTrips -= 1
                                    return act_index
                                    
                elif greateacion[0] == 'Trips' and Trips > 0:
                    for i in range(len(Trips_i)-2):
                        if Trips_i[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Trips' and actionlist[j][1] == action_dict[Trips_i[i]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[Trips_j[i]] and actionlist[j][2][1][0] == hua_dict[Trips_j[i+1]] and actionlist[j][2][2][0] == hua_dict[Trips_j[i+2]]:
                                        act_index = j
                                        # _debug_print('对手出三张最大出三张')
                                        Trips -= 1
                                        return act_index
                                        
                elif greateacion[0] == 'ThreeWithTwo' and ThreeWithTwo > 0:
                    for i in range(len(Trips_j)-2):
                        if Trips_i[i]+2 > row_dict[greateacion[1]]:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'ThreeWithTwo' and actionlist[j][1] == action_dict[Trips_i[i]+2] and actionlist[j][2][-1][-1] == action_dict[Pair_[0]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[Trips_j[i]] and actionlist[j][2][1][0] == hua_dict[Trips_j[i+1]] and actionlist[j][2][2][0] == hua_dict[Trips_j[i+2]]:
                                        for x in range(len(Pair_j)-1):
                                            if actionlist[j][2][3][0] == hua_dict[Pair_j[x]] and actionlist[j][2][4][0] == hua_dict[Pair_j[x + 1]]:
                                                act_index = j
                                                # _debug_print('对手出三带二最大出三带二')
                                                Trips -= 1
                                                Pair -= 1
                                                ThreeWithTwo = Trips * Pair
                                                return act_index
                                                
                elif zongzha > 3 or (zongzha > 2 and rest_me < 20) or (zongzha > 1 and rest_me < 15) or (zongzha > 0 and rest_me < 10):
                    if len(bomb_i) > 0:
                        for i in range(len(bomb_i)):
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Bomb' and actionlist[j][1] == action_dict[bomb_i[i]+2]:
                                    act_index = j
                                    # _debug_print('对手出的牌型没有出炸弹')
                                    bomb -= 1
                                    return act_index
                                    
                    elif len(flush_j) > 0:
                        for i in range(len(flush_j)):
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'StraightFlush' and actionlist[j][1] == action_dict[flush_j[i]+2]:
                                    act_index = j
                                    # _debug_print('对手出的牌型没有出同花顺')
                                    bomb -= 1
                                    return act_index
                                    
                    elif len(flush_j) > 0:
                        for i in range(len(flush_j)):
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'StraightFlush' and actionlist[j][1] == action_dict[flush_j[i]+2]:
                                    act_index = j
                                    # _debug_print('对手出的牌型没有出通配同花顺')
                                    bomb -= 1
                                    return act_index
                                    
                    elif len(bomb_j) > 0 and rest_me < 9:
                        for j in range(len(bomb_j)):
                            for i in range(len(actionlist)):
                                if actionlist[i][0] == 'Bomb' and actionlist[i][1] == action_dict[bomb_j[j]+2]:
                                    act_index = i
                                    # _debug_print('对手出的牌型没有出6炸上')
                                    bomb -= 1
                                    return act_index
                                    
                elif greateacion[0] == 'Pair' and Pair > 0:
                    for i in range(len(Pair_)-1):
                        if Pair_[i]+2 > row_dict[greateacion[1]] or Pair_[i]+2 == rank:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Pair' and actionlist[j][1] == action_dict[Pair_[i]+2]:
                                    if actionlist[j][2][0][0] == hua_dict[Pair_j[i]] and actionlist[j][2][1][0] == hua_dict[Pair_j[i+1]]:
                                        act_index = j
                                        # _debug_print('队手出对子最大出对子')
                                        Pair -= 1
                                        return act_index
                                    
                elif greateacion[0] == 'Single' and Single > 0:
                    for i in range(len(Single_)):
                        if Single_[i]+2 > row_dict[greateacion[1]] or Single_[i]+2 == rank:
                            for j in range(len(actionlist)):
                                if actionlist[j][0] == 'Single' and actionlist[j][1] == action_dict[Single_[i]+2] and actionlist[j][2][0][0] == hua_dict[Single_j[i]]:
                                    act_index = j
                                    # _debug_print('对手出单牌最大出单牌')
                                    Single -= 1
                                    return act_index
                                    
            # _debug_print('自己的判断后没有牌型可以打选择pass 不考虑拆牌')
            return 0


        # _debug_print("可选动作范围为：0至{}".format(self.act_range))
        return randint(0, self.act_range)
    
    def handsort(self, msg):
        hand = np.zeros(shape=(4, 16), dtype='i1')
        self.shoupaijuzhen = hand
        shoupai = msg["handCards"]
        #_debug_print('当前手牌是：',shoupai)
        line_dict = {'S':0,'H':1,'C':2,'D':3}
        row_dict = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14,'B':16,'R':17}
        for i in range(len(shoupai)):
            self.shoupaijuzhen[line_dict[shoupai[i][0]], row_dict[shoupai[i][1]] - 2] += 1

        rank = msg["curRank"]
        #_debug_print('当前打牌等级是：',row_dict[rank])

        #_debug_print('手牌矩阵是\n',self.shoupaijuzhen)
        return row_dict[rank]

    def straight_tong_flush(self, rank):
        row_sum = self.shoupaijuzhen.sum(axis=0)
        _debug_print('stf',row_sum)
        Single = 0
        Bomb_4 = 0
        for i in range(len(row_sum)):
            if row_sum[i]==4:
                Bomb_4 += 1
            if row_sum[i]==1:
                Single += 1

        Straight_tong_flush = 0
        flush_tong_i = []
        flush_tong_j = []
        dingwei = 0
        if self.shoupaijuzhen[1][rank-2] > 0:
            # _debug_print('存在{}张通配牌'.format(self.shoupaijuzhen[1][rank-2]))
            # 存在通配牌先检验不减少炸弹数量的情况下能否添加同花顺
            # row_sum_[num] -= tongpei[0][num]
            # _debug_print('考虑通配牌时只考虑单牌有同花顺')
            for i in range(4):
                for j in range(9):
                    if self.shoupaijuzhen[i][j] != self.shoupaijuzhen[1][rank-2]:
                        if self.shoupaijuzhen[i][j] == 1 and self.shoupaijuzhen[i][j + 1] == 1 and self.shoupaijuzhen[i][j + 2] == 1 and self.shoupaijuzhen[i][j + 3] == 1:
                            dingwei = 1
                            Straight_tong_flush += 1
                            flush_tong_i.append(i)
                            flush_tong_j.append(j)
                            self.shoupaijuzhen[1][rank-2] -= 1
                            self.shoupaijuzhen[i][j] -= 1
                            self.shoupaijuzhen[i][j+1] -= 1
                            self.shoupaijuzhen[i][j+2] -= 1
                            self.shoupaijuzhen[i][j+3] -= 1
                        if self.shoupaijuzhen[i][j] == 1 and self.shoupaijuzhen[i][j + 2] == 1 and self.shoupaijuzhen[i][j + 3] == 1 and self.shoupaijuzhen[i][j + 4] == 1:
                            dingwei = 2
                            Straight_tong_flush += 1
                            flush_tong_i.append(i)
                            flush_tong_j.append(j)
                            self.shoupaijuzhen[1][rank-2] -= 1
                            self.shoupaijuzhen[i][j] -= 1
                            self.shoupaijuzhen[i][j+2] -= 1
                            self.shoupaijuzhen[i][j+3] -= 1
                            self.shoupaijuzhen[i][j+4] -= 1
                        if self.shoupaijuzhen[i][j] == 1 and self.shoupaijuzhen[i][j + 1] == 1 and self.shoupaijuzhen[i][j + 3] == 1 and self.shoupaijuzhen[i][j + 4] == 1:
                            dingwei = 3
                            Straight_tong_flush += 1
                            flush_tong_i.append(i)
                            flush_tong_j.append(j)
                            self.shoupaijuzhen[1][rank-2] -= 1
                            self.shoupaijuzhen[i][j] -= 1
                            self.shoupaijuzhen[i][j+1] -= 1
                            self.shoupaijuzhen[i][j+3] -= 1
                            self.shoupaijuzhen[i][j+4] -= 1
                        if self.shoupaijuzhen[i][j] == 1 and self.shoupaijuzhen[i][j + 1] == 1 and self.shoupaijuzhen[i][j + 2] == 1 and self.shoupaijuzhen[i][j + 4] == 1:
                            dingwei = 4
                            Straight_tong_flush += 1
                            flush_tong_i.append(i)
                            flush_tong_j.append(j)
                            self.shoupaijuzhen[1][rank-2] -= 1
                            self.shoupaijuzhen[i][j] -= 1
                            self.shoupaijuzhen[i][j+1] -= 1
                            self.shoupaijuzhen[i][j+2] -= 1
                            self.shoupaijuzhen[i][j+4] -= 1

        if flush_tong_i != []:
            # 判断同花顺是否减少4张炸弹数量和增加的单牌数量
            row_sum_ = self.shoupaijuzhen.sum(axis=0)
            Bomb_4_ = 0
            Single_ = 0
            for i in range(len(row_sum_)):
                if row_sum_[i]==4:
                    Bomb_4_ += 1
                if row_sum_[i]==1:
                    Single_ += 1
            for j in range(len(flush_tong_j)):
                if (Bomb_4 - Bomb_4_ > len(flush_tong_j)-1 and Single - Single_ > 1) or (Bomb_4 - Bomb_4_ > len(flush_tong_j)):
                    Straight_tong_flush -= 1
                    self.shoupaijuzhen[1][rank-2] += 1
                    # _debug_print('同花顺拆除了2个炸弹并且单牌比原来多时不考虑组同花顺')
                    if dingwei == 1:
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+1] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+2] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+3] += 1
                        flush_tong_i.remove(flush_tong_i[j])
                        flush_tong_j.remove(flush_tong_j[j])
                    elif dingwei == 2:
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+2] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+3] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+4] += 1
                        flush_tong_i.remove(flush_tong_i[j])
                        flush_tong_j.remove(flush_tong_j[j])
                    elif dingwei == 3:
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+1] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+3] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+4] += 1
                        flush_tong_i.remove(flush_tong_i[j])
                        flush_tong_j.remove(flush_tong_j[j])
                    elif dingwei == 4:
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+1] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+2] += 1
                        self.shoupaijuzhen[flush_tong_i[j]][flush_tong_j[j]+4] += 1
                        flush_tong_i.remove(flush_tong_i[j])
                        flush_tong_j.remove(flush_tong_j[j])

        # 输出同花顺后的手牌
        if Straight_tong_flush > 0:
            # _debug_print('tongpei同花顺更新后的手牌矩阵为\n',self.shoupaijuzhen)
            # _debug_print('stf',row_sum)
            # _debug_print('含有通配牌的同花顺为：', flush_tong_i, flush_tong_j)
            return Straight_tong_flush, flush_tong_i, flush_tong_j
        else:
            return 0, [], []
  
    def danpai(self, action_list):
        Single = 0
        Single_ = []
        Single_j = []

        suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14}

        for action in action_list:
            if action[0] == 'Single':
                rank = action[1]         # 比如 'Q', 'K'
                cards = action[2]        # 比如 ['HQ']

                if cards and len(cards) == 1:
                    suit = cards[0][0]   # 花色，如 'H'
                    rank_char = cards[0][1]  # 点数，如 'Q'

                    suit_idx = suit_map.get(suit, -1)
                    rank_idx = rank_map.get(rank_char, -1)

                    if suit_idx != -1 and rank_idx != -1:
                        Single += 1
                        Single_.append(rank_idx)
                        Single_j.append(suit_idx)

        if Single > 0:
            return Single, Single_, Single_j
        else:
            return 0, [], []
    
    def duizi(self, action_list):
        Pair = 0
        Pair_ = []
        Pair_j = []

        suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14}

        for action in action_list:
            if action[0] == 'Pair' and len(action[2]) == 2:
                cards = action[2]
                Pair += 1
                for card in cards:
                    rank_char = card[1]
                    suit_char = card[0]
                    rank = rank_map.get(rank_char, -1)
                    suit = suit_map.get(suit_char, -1)
                    if rank != -1 and suit != -1:
                        Pair_.append(rank)
                        Pair_j.append(suit)

        return (Pair, Pair_, Pair_j) if Pair else (0, [], [])

    def trips(self, action_list):
        Trips = 0
        Trips_ = []
        Trips_j = []

        suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14}

        for action in action_list:
            if action[0] == 'Trips' and len(action[2]) == 3:
                cards = action[2]
                Trips += 1
                for card in cards:
                    rank_char = card[1]
                    suit_char = card[0]
                    rank = rank_map.get(rank_char, -1)
                    suit = suit_map.get(suit_char, -1)
                    if rank != -1 and suit != -1:
                        Trips_.append(rank)
                        Trips_j.append(suit)

        return (Trips, Trips_, Trips_j) if Trips else (0, [], [])

    def sandui(self, action_list):
        ThreePair = 0
        ThreePair_ = []
        ThreePair_j = []

        suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14}

        for action in action_list:
            if action[0] == 'ThreePair':
                cards = action[2]
                ThreePair += 1
                for card in cards:
                    rank = rank_map.get(card[1], -1)
                    suit = suit_map.get(card[0], -1)
                    if rank != -1 and suit != -1:
                        ThreePair_.append(rank)
                        ThreePair_j.append(suit)

        return (ThreePair, ThreePair_, ThreePair_j) if ThreePair else (0, [], [])

    def sandaier(self, action_list):
        ThreeWithTwo = 0
        ThreeWithTwo_ = []
        ThreeWithTwo_j = []

        suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14}

        for action in action_list:
            if action[0] == 'ThreeWithTwo':
                cards = action[2]
                ThreeWithTwo += 1
                for card in cards:
                    ThreeWithTwo_.append(rank_map[card[1]])
                    ThreeWithTwo_j.append(suit_map[card[0]])

        return (ThreeWithTwo, ThreeWithTwo_, ThreeWithTwo_j) if ThreeWithTwo else (0, [], [])

    def gangban(self, action_list):
        TwoTrips = 0
        TwoTrips_ = []
        TwoTrips_j = []

        suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14}

        for action in action_list:
            if action[0] == 'TwoTrips':
                cards = action[2]
                TwoTrips += 1
                for card in cards:
                    rank = rank_map.get(card[1], -1)
                    suit = suit_map.get(card[0], -1)
                    if rank != -1 and suit != -1:
                        TwoTrips_.append(rank)
                        TwoTrips_j.append(suit)

        return (TwoTrips, TwoTrips_, TwoTrips_j) if TwoTrips else (0, [], [])

    def shunzi(self, action_list):
        Straight = 0
        Straight_i = []  # 顺子起始点（最小点数）
        hangwei = []     # 每张牌的花色索引
        liewei = []      # 每张牌的点数索引

        suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14}

        for action in action_list:
            if action[0] == 'Straight':
                cards = action[2]
                Straight += 1

                # 获取当前顺子的最小点数作为起始索引
                min_point = min(rank_map[card[1]] for card in cards)
                Straight_i.append(min_point)

                for card in cards:
                    liewei.append(rank_map[card[1]])
                    hangwei.append(suit_map[card[0]])

        if Straight > 0:
            return Straight, Straight_i, hangwei, liewei
        else:
            return 0, [], [], []

    def zhadan(self, action_list):
        Bomb = 0
        Bomb_ = []
        Bomb_j = []
        suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14}

        for action in action_list:
            if action[0] in ['Bomb', 'Boom']:
                cards = action[2]
                Bomb += 1
                for card in cards:
                    Bomb_.append(rank_map[card[1]])
                    Bomb_j.append(suit_map[card[0]])

        return (Bomb, Bomb_, Bomb_j) if Bomb else (0, [], [])

    def tonghuashun(self, action_list):
        StraightFlush = 0
        flush_i = []
        flush_j = []

        suit_map = {'S': 0, 'H': 1, 'C': 2, 'D': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                    'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12, 'B': 13, 'R': 14}

        for action in action_list:
            if action[0] == 'StraightFlush':
                cards = action[2]
                StraightFlush += 1
                for card in cards:
                    flush_i.append(rank_map.get(card[1], -1))
                    flush_j.append(suit_map.get(card[0], -1))

        return (StraightFlush, flush_i, flush_j) if StraightFlush else (0, [], [])


class Base8Agent(Player):
    ''' Baseline 8 agent.
    '''
    name = 'Base8'
    
    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)
        self.action = Action()
        self.use_raw = True
        self.begin = True
    
    def step(self, state):
        if not state['actions']:
            return []
        msg = self.parse(state)
        act_index = self.action.parse(msg, self.player_id)

         #我方先手 AI选择部分开始

        final_action = state['actions'][act_index]
        #_debug_print("我打出来的下标是：", act_index)
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
    
    def tribute_act(self, actionlist, rank):
        """根据合法动作集选择动作执行"""
        message = {"handCards":self.current_hand_str, "curRank":rank}
        rank = self.action.handsort(message)
        # 同花顺
        straight_flush, flush_i, flush_j = self.action.tonghuashun(actionlist)
        # 通配牌同花顺
        straight_tong_flush, flush_tong_i, flush_tong_j = self.action.straight_tong_flush(rank)
        straight = straight_tong_flush + straight_flush

        act_index = randint(0, len(actionlist)-1)

        action_dict = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A',16:'B',17:'R'}
        hua_dict = {0:'S',1:'H',2:'C',3:'D'}
        if self.action.shoupaijuzhen[1][15] == 0 and self.action.shoupaijuzhen[0][14] == 0:
            if len(actionlist) > 1 and straight > 0:
                for i in range(len(flush_j)):
                    for j in range(4):
                        if self.action.shoupaijuzhen[j][rank-2] > 0 and (flush_j[i] > self.action.shoupaijuzhen[j][rank-2] or flush_j[i]+4 < self.action.shoupaijuzhen[j][rank-2]):
                            for x in range(len(actionlist)):
                                if actionlist[x][-1][0][0] == hua_dict[flush_i[i]] and actionlist[x][-1][0][1] == action_dict[flush_j[i]+2]:
                                    act_index = x
                                    act = actionlist[act_index]
                                    self.execute_tribute(act)
                                    return act
                for i in range(len(flush_tong_j)):
                    for j in range(4):
                        if self.action.shoupaijuzhen[j][rank-2] > 0 and (flush_tong_j[i] > self.action.shoupaijuzhen[j][rank-2] or flush_tong_j[i]+4 < self.action.shoupaijuzhen[j][rank-2]):
                            for x in range(len(actionlist)):
                                if actionlist[x][-1][0][0] == hua_dict[flush_tong_i[i]] and actionlist[x][-1][0][1] == action_dict[flush_tong_j[i]+2]:
                                    act_index = x
                                    act = actionlist[act_index]
                                    self.execute_tribute(act)
                                    return act
                                
        act = actionlist[act_index]
        self.execute_tribute(act)
        return act
        
    def back_act(self, actionlist, curRank, tribute_result):
        """根据合法动作集选择动作执行"""
        message = {"handCards":self.current_hand_str, "curRank":curRank}
        rank = self.action.handsort(message)
        # 单牌
        Single, Single_, Single_j = self.action.danpai(actionlist)
        
        action_dict = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A',16:'B',17:'R'}
        hua_dict = {0:'S',1:'H',2:'C',3:'D'}
        
        act_index = randint(0, len(actionlist)-1)
        
        if Single > 0:
            # _debug_print('还贡单牌选项有', Single_)
            for i in range(len(Single_)):
                for j in range(len(actionlist)):
                    if actionlist[j][-1][0][0] == hua_dict[Single_j[i]] and actionlist[j][-1][0][1] == action_dict[Single_[i]+2]:
                        act_index = j
                        # _debug_print('自己选择还贡的牌')
                        act = actionlist[act_index]
                        self.execute_tribute(act)
                        return act
                    
        act = actionlist[act_index]
        self.execute_back(act)
        return act
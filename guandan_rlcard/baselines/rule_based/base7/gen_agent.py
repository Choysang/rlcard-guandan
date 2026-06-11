# -*- coding: utf-8 -*-
# @Time       : 2020/10/19 19:30
# @Author     : Duofeng Wu  &&  Zenghui Qian


from guandan_rlcard.game.player import GuandanPlayer as Player
import json
import os

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

CARD_TYPE = [
    "Single",
    "Pair",
    "Trips",
    "ThreePair",
    "ThreeWithTwo",
    "TwoTrips",
    "Straight",
    "StraightFlush",
    "Bomb",
    "PASS"
]


CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
        'A', 'BJ', 'RJ']

#              0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
str_to_ind = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'L', 'B', 'R']
#              0    1    2    3
str_to_flo = ['S', 'H', 'C', 'D']


CARDTYPE_PROMPT = '''
Now it is your turn to act. The current game state is described as follows:

```Current state information:
Player IDs are [0, 1, 2, 3], your ID is {id}, your teammate's ID is {teammate_id}, and your team ID is {teamid}.
Team 0's levle is {team0_rank}, and Team 1's level is {team1_rank}. Currently, We are playing team {play_team}'s level {cur_rank}, and the wild card is 'H{cur_rank}'.
```

```Your current hand card:
{current_hand}
```

```Each player's remaining hand count (in order of player ID):
{num_cards_left}
```

```Recent greatest action:
{current_action}
```  

```Legal card types:
{legal_cardtype}
``` 

During the game, you must adhere to the following unbreakable rules:
- Choose a card type from the legal card types as your final answer., do not output any other information.;
- If the recent greatest action is 'NULL', then you can not choose 'PASS' as your answer;
- If your answer is not 'PASS', make sure your hand cards are able to make up a stronger action to cover the recent greatest action.
- If your answer is not 'PASS', then it can only be the same as Current acion's CardType or 'Bomb' or 'StraightFlush';
'''

Guandan_Player_PROMPT = '''
Now it is your turn to act. The current game state is described as follows:

```Current state information:
Player IDs are [0, 1, 2, 3], your ID is {id}, your teammate's ID is {teammate_id}, and your team ID is {teamid}.
Currently, Team 0's rank is {team0_rank}, and Team 1's rank is {team1_rank}.
You are now playing against Team {play_team} with a rank of {cur_rank}, and the wild card is 'H{cur_rank}'.
```

```Your current hand card:
{current_hand}
```

```Each player's remaining hand count (in order of player ID):
{num_cards_left}
```

```All the played cards:
{played_cards}
```

```Recent action history (Listed in chronological order from earliest to latest, [player ID, action]):
{trace}
```

During the game, you must adhere to the following unbreakable rules:
- Strictly follow the action format, where the specific list of cards includes the same number of cards as required by the card type.
- The cards included in the action must be within your current hand cards.

'''

class Action(object):

    def __init__(self):
        self.action = []
        self.act_range = -1

    def parse(self, msg, mate_pos):     # 增加了一个新参数mate_pos，表示队友的位置
        
        act = solve(msg, mate_pos) # 注意，此处返回的直接是动作而非序号了
        return act


class GenAgent(Player):
    ''' Generate data agent.
    '''
    def __init__(self, player_id, np_random, output_path=None):
        super().__init__(player_id, np_random)
        # FIX: dataset path was hardcoded to a developer machine.
        self.output_path = (output_path
                            or os.environ.get('GUANDAN_DATASET_PATH',
                                              'guandan_dataset.jsonl'))
        self.action = Action()
        self.my_pos = self.player_id                   # 增加了一个属性，用来记录自己的位置
        self.mate_pos = (self.my_pos + 2) % 4          # 增加了一个属性，用来记录队友的位置
        self.use_raw = True

    def step(self, state):
        if not state['actions']:
            return []
        
        # data_gen
        id = state['self']
        teammate_id = (id+2) % 4
        teamid = state['teamid']
        played_cards = state['played_cards']
        legal_actions = state['actions']
        rank_list = state['rank_list']
        team0_rank = CARD_RANK[rank_list[0]]
        team1_rank = CARD_RANK[rank_list[1]]
        play_team = state['play_team']
        cur_rank = CARD_RANK[rank_list[play_team]]
        current_hand = state['current_hand']
        action_index_range = len(legal_actions)-1
        num_cards_left = state['num_cards_left']
        
        if len(state['trace']) < 5:
            trace = state['trace']
        else:
            trace = state['trace'][-5:]
          
        current_action = 'NULL'
        for action in trace:
            if action[0] != id and action[1][0] != 'PASS':
                current_action = action
                
        legal_cardtype = set(item[0] for item in legal_actions)
        
        prompt = CARDTYPE_PROMPT.format(id=id, teammate_id=teammate_id, teamid=teamid, 
                    team0_rank=team0_rank, team1_rank=team1_rank,
                    play_team=play_team, cur_rank=cur_rank, 
                    current_hand=current_hand, 
                    num_cards_left=num_cards_left,
                    current_action=current_action,
                    legal_cardtype=legal_cardtype)
        
        msg = self.parse(state)
        act = self.action.parse(msg, self.mate_pos)
        
        meta_data = {'prompt': prompt, "legal_list": str(legal_cardtype)}
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta_data, ensure_ascii=False) + "\n")
        
        return act
    
    def parse(self, state):
        assert type(state) == dict
        msg = {}
        rank_list = state['rank_list']
        play_team = state['play_team']
        msg['curRank'] = CARD_RANK[rank_list[play_team]]
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
        zero_s_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # 2 -- 王
        zero_h_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zero_c_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          # 2 -- A
        zero_d_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zero_number_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # 2 -- A
        zero_rank_cards = [0, 0]            # 非红桃 红桃

        has = [zero_s_cards, zero_h_cards, zero_c_cards, zero_d_cards, zero_number_cards, zero_rank_cards]
        now_rank = rank           # 获取等级

        for card in self.current_hand_str:       # 统计当前手牌
            if card[1] == now_rank and card[0] == 'H':
                has[5][1] += 1
            elif card[1] == now_rank and card[0] != 'H':
                has[5][0] += 1
            card_to_list(card, has, 1)

        act_score = []                  # 存放所有行动选项的评分
        for action in actionLists:
            values = []
            for one in action[2]:
                values.append(getval(one, now_rank, has))
            value = max(values)
            poss = 1
            score = cac(1, value, poss)
            act_score.append(score)
            
        act_index = act_score.index(max(act_score))
        act = actionLists[act_index]
        self.execute_tribute(act)
        
        return act
    
    def back_act(self, actionLists, rank, tribute_result):
        """根据合法动作集选择动作执行"""
        zero_s_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # 2 -- 王
        zero_h_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zero_c_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          # 2 -- A
        zero_d_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        zero_number_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # 2 -- A
        zero_rank_cards = [0, 0]            # 非红桃 红桃

        has = [zero_s_cards, zero_h_cards, zero_c_cards, zero_d_cards, zero_number_cards, zero_rank_cards]
        now_rank = rank           # 获取等级

        for card in self.current_hand_str:       # 统计当前手牌
            if card[1] == now_rank and card[0] == 'H':
                has[5][1] += 1
            elif card[1] == now_rank and card[0] != 'H':
                has[5][0] += 1
            card_to_list(card, has, 1)

        act_score = []                  # 存放所有行动选项的评分
        for action in actionLists:
            values = []
            for one in action[2]:
                values.append(getval(one, now_rank, has))
            value = max(values)
            poss = 1
            score = cac(1, value, poss)
            act_score.append(score)
        
        act_index = act_score.index(max(act_score))
        act = actionLists[act_index]
        self.execute_back(act)
        
        return act

def card_to_list(card, my_list, step):
    if card == 'RJ':
        card = 'HR'
    elif card == 'BJ':
        card = 'SB'
    des = str_to_flo.index(card[0])
    #  数字
    if card[1] == 'T':
        my_list[des][8] += step
        my_list[4][8] += step
    elif card[1] == 'J':
        my_list[des][9] += step
        my_list[4][9] += step
    elif card[1] == 'Q':
        my_list[des][10] += step
        my_list[4][10] += step
    elif card[1] == 'K':
        my_list[des][11] += step
        my_list[4][11] += step
    elif card[1] == 'A':
        my_list[des][12] += step
        my_list[4][12] += step
    elif card[1] == 'B':
        my_list[des][13] += step
    elif card[1] == 'R':
        my_list[des][13] += step
    else:
        my_list[des][int(card[1]) - 2] += step
        my_list[4][int(card[1]) - 2] += step


def getval(card, rank, has):               # 评估一张牌在手牌中的价值，card: 牌面 如"H2"，rank 当前等级，has 手牌情况
    if card == 'RJ':
        card = 'HR'
    elif card == 'BJ':
        card = 'SB'
    des = str_to_flo.index(card[0])
    val = str_to_ind.index(card[1]) + 1
    index = str_to_ind.index(card[1])
    if card[1] == 'B' or card[1] == 'R':
        index = 13
    if card[1] == rank:                     # 对级牌重新赋予价值
        val = 14
    if card == 'SB':                        # 对大王小王重新赋予价值
        if has[0][13] == 2 and has[1][13] == 2:
            val *= 100
        elif has[0][13] == 2 and has[1][13] != 2:
            val += 20
    elif card == 'HR':
        if has[0][13] == 2 and has[1][13] == 2:
            val *= 100
        elif has[0][13] != 2 and has[1][13] == 2:
            val += 20
    elif card[0] == 'H' and card[1] == rank:    # 对红桃级牌重新赋予价值
        val = 340
    else:
        ans = 0                         # ans 暂时记录构成同花顺的牌的新价值
        for num in range(index-4, index+1):   # 搜寻同花顺
            if num >= -1 and num+4 <= 12:
                if num == -1:
                    if has[des][12] >= 1 and has[des][0] >= 1 and has[des][1] >= 1 and has[des][2] >= 1 \
                            and has[des][3] >= 1:
                        ans = 320+val
                else:
                    if has[des][num] >= 1 and has[des][num+1] >= 1 and has[des][num+2] >= 1 and has[des][num+3] >= 1 \
                            and has[des][num+4] >= 1:
                        ans = 320+val
        if has[4][index] <= 3:          # 查看同点数的牌的数目，根据数目重新赋予价值
            val += 20 * (has[4][index]-1)
        elif has[4][index] == 4:
            val += 220
        elif has[4][index] == 5:
            val += 300
        elif has[4][index] == 6:        # 修改了超过5张的赋值，是更符合出牌逻辑 2020.10.21
            val += 400                  # 再次修改  2020.10.23
        elif has[4][index] == 7:
            val += 500
        elif has[4][index] == 8:
            val += 600
        if val < ans:                   # 取val 和 ans 中较大者为最终价值
            val = ans
    return val                          # 评估完成，返回该牌的价值


def cac(gain, value, poss):             # 评分计算公式 有待完善
    return gain*(1 + poss)/value        # Score = Gain * ( 1 + Possibility ) / Value;（算法会优先打出评分较高的打法）


def solve(msg, mate_pos):                   # 主体函数  生成记录手牌情况的列表
    zero_s_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # 2 -- 王
    zero_h_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    zero_c_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          # 2 -- A
    zero_d_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    zero_number_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]     # 2 -- A
    zero_rank_cards = [0, 0]            # 非红桃 红桃

    has = [zero_s_cards, zero_h_cards, zero_c_cards, zero_d_cards, zero_number_cards, zero_rank_cards]
    now_rank = msg["curRank"]           # 获取等级
    now_greater_pos = msg["greaterPos"]
    op1_pos = (mate_pos + 1) % 4
    op2_pos = (mate_pos + 3) % 4

    for card in msg["handCards"]:       # 统计当前手牌
        if card[1] == now_rank and card[0] == 'H':
            has[5][1] += 1
        elif card[1] == now_rank and card[0] != 'H':
            has[5][0] += 1
        card_to_list(card, has, 1)

    act_score = []                  # 存放所有行动选项的评分

    for action in msg["actionList"]:   # 对所有行动选项进行评估
        if len(action[2]) == len(msg["handCards"]):
            act_score.append(100000)
        else:
            if action[0] == "Single":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(1, value, poss)
                n_score = score
                if now_greater_pos == -1:
                    if msg["publicInfo"][op1_pos]['rest'] == 1:
                        n_score = -1 * score
                    if msg["publicInfo"][mate_pos]['rest'] == 1:
                        n_score = score + 100
                    if msg["publicInfo"][op2_pos]['rest'] == 1:
                        n_score = -1 * score
                    act_score.append(n_score)
                else:
                    if msg["publicInfo"][op1_pos]['rest'] == 1:
                        n_score = score
                    if msg["publicInfo"][mate_pos]['rest'] == 1:
                        n_score = -10000
                    if msg["publicInfo"][op2_pos]['rest'] == 1:
                        n_score = -1 * score
                    act_score.append(n_score)

            elif action[0] == "Pair":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(2, value, poss)
                n_score = score
                if now_greater_pos == -1:
                    if msg["publicInfo"][op1_pos]['rest'] == 2:
                        n_score = -1 * score
                    if msg["publicInfo"][mate_pos]['rest'] == 2:
                        n_score = score + 100
                    if msg["publicInfo"][op2_pos]['rest'] == 2:
                        n_score = -1 * score
                    act_score.append(n_score)
                else:
                    if msg["publicInfo"][op2_pos]['rest'] == 2 or msg["publicInfo"][op2_pos]['rest'] == 2:
                        n_score = -1 * score
                    act_score.append(n_score)

            elif action[0] == "Trips":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(3, value, poss)
                n_score = score
                if now_greater_pos == -1:
                    if msg["publicInfo"][op1_pos]['rest'] == 3:
                        n_score = -1 * score
                    if msg["publicInfo"][op2_pos]['rest'] == 3:
                        n_score = -1 * score
                    act_score.append(n_score)
                else:
                    if msg["publicInfo"][op1_pos]['rest'] == 3 or msg["publicInfo"][op2_pos]['rest'] == 3:
                        n_score = -1 * score
                    act_score.append(n_score)

            elif action[0] == "ThreePair":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(6, value, poss)
                act_score.append(score)

            elif action[0] == "ThreeWithTwo":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = (max(values) + min(values)) / 2
                poss = 1
                score = cac(5, value, poss)
                act_score.append(score)

            elif action[0] == "TwoTrips":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(6, value, poss)
                act_score.append(score)

            elif action[0] == "Straight":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = sum(values)
                poss = 1
                score = cac(5, value, poss)
                act_score.append(score)

            elif action[0] == "StraightFlush":
                if now_greater_pos == mate_pos or now_greater_pos == -1:
                    act_score.append(-10000)    # 如果是队友，就不会出同花顺  解决 接队友单张时不能排除的问题  2020.10.20
                elif now_greater_pos == op1_pos or now_greater_pos == op2_pos:
                    if msg["publicInfo"][now_greater_pos]['rest'] <= 14:
                        values = []
                        for one in action[2]:
                            values.append(getval(one, now_rank, has))
                        value = max(values)
                        poss = 1
                        score = cac(5, value, poss)
                        act_score.append(score)
                    else:
                        act_score.append(-1)

            elif action[0] == "Bomb":
                if now_greater_pos == mate_pos or now_greater_pos == -1:
                    act_score.append(-10000)    # 如果是队友，就不会出炸弹    解决 接队友单张时不能排除的问题  2020.10.20
                elif now_greater_pos == op1_pos or now_greater_pos == op2_pos:
                    if msg["publicInfo"][now_greater_pos]['rest'] <= 14:
                        values = []
                        for one in action[2]:
                            values.append(getval(one, now_rank, has))
                        value = max(values)
                        poss = 1
                        score = cac(len(values), value, poss)
                        act_score.append(score)
                    else:
                        act_score.append(-1)

            elif action[0] == "PASS":
                if now_greater_pos == mate_pos:
                    if msg["publicInfo"][mate_pos]['rest'] <= 6:  # 队友的牌数小于6时，不会压队友  PASS优先级最高
                        value = 1
                        poss = 1
                        score = cac(1, value, poss)
                        act_score.append(score)
                    else:
                        if msg["greaterAction"][0] == "Single":  # 单张情况  酌情设置PASS优先级
                            value = 25
                            poss = 1
                            score = cac(2, value, poss)
                            act_score.append(score)
                        elif msg["greaterAction"][0] == "Pair":  # 对子情况  酌情设置PASS优先级
                            value = 65
                            poss = 1
                            score = cac(4, value, poss)
                            act_score.append(score)
                        elif msg["greaterAction"][0] == "Trips":  # 对子情况  酌情设置PASS优先级
                            value = 103
                            poss = 1
                            score = cac(6, value, poss)
                            act_score.append(score)
                        else:                               # 不会压队友  PASS优先级最高
                            value = 1
                            poss = 1
                            score = cac(1, value, poss)
                            act_score.append(score)
                else:                                       # 对方牌权  PASS优先级最低
                    if 1 <= msg["publicInfo"][op2_pos]['rest'] <= 3 :
                        act_score.append(-9999)
                    else:
                        value = 1
                        poss = 1
                        score = cac(0, value, poss)
                        act_score.append(score)

            elif action[0] == "back":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(1, value, poss)
                act_score.append(score)

            elif action[0] == "tribute":
                values = []
                for one in action[2]:
                    values.append(getval(one, now_rank, has))
                value = max(values)
                poss = 1
                score = cac(1, value, poss)
                act_score.append(score)

    result_index = act_score.index(max(act_score))     # 返回所有选项中分数最高的选项的下标

    return msg["actionList"][result_index]

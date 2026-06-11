"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch 
from collections import Counter

RANK = {
    '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8,
    'T':9, 'J':10, 'Q':11, 'K':12, 'A':13
}

# 动作向量化
TYPPES = ["Single", "Pair", "Trips", "ThreePair", "ThreeWithTwo", "TwoTrips", "Straight", "StraightFlush", "Bomb", "PASS"]
STRENGTH = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'B', 'R']

CARD_RANK = ['2','3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
    'A', 'BJ', 'RJ']

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

CardToNum = {
    'H2':0, 'H3':1, 'H4':2, 'H5':3, 'H6':4, 'H7':5, 'H8':6, 'H9':7, 'HT':8, 'HJ':9, 'HQ':10, 'HK':11, 'HA':12,
    'S2':13, 'S3':14, 'S4':15, 'S5':16, 'S6':17, 'S7':18, 'S8':19, 'S9':20, 'ST':21, 'SJ':22, 'SQ':23, 'SK':24, 'SA':25,
    'C2':26, 'C3':27, 'C4':28, 'C5':29, 'C6':30, 'C7':31, 'C8':32, 'C9':33, 'CT':34, 'CJ':35, 'CQ':36, 'CK':37, 'CA':38,
    'D2':39, 'D3':40, 'D4':41, 'D5':42, 'D6':43, 'D7':44, 'D8':45, 'D9':46, 'DT':47, 'DJ':48, 'DQ':49, 'DK':50, 'DA':51,
    'SB':52, 'HR':53
}

def _format_observation(obs, device):
    """ 修改信息的格式，并迁移到设备上
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    # 将 NumPy 数组（numpy.ndarray）转换为 PyTorch 张量，使得数据可以在 PyTorch 的计算框架中使用
    # 转换后的张量与原始 NumPy 数组 ​​共享底层内存​​，修改其中一方的数据会直接影响另一方，避免数据复制的性能开销
    x_batch = torch.from_numpy(obs['x_batch'].astype(np.int8)).to(device)
    z_batch = torch.from_numpy(obs['z_batch'].astype(np.int8)).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'].astype(np.int8))
    z = torch.from_numpy(obs['z'])
    # 转换后的obs只包含合法动作数量的输入数据，以及对应的动作特征
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           }
    return position, obs, x_no_action, z

def card2num(list_cards):      # 将字符串转换成数字
    res = []   
    if list_cards == None:
        return res
    if list_cards == -1:
        return [-1] 
    for ele in list_cards:
        if ele in CardToNum:
            res.append(CardToNum[ele])
    return res

def card2array(list_cards):
    "将卡牌列表转为一维包含54个元素的向量"
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)
    
    if list_cards == [-1]:  # 直接添加的(只有一个-1)直接返回负值
        return -1*np.ones(54, dtype=np.int8)
    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    
    for card, num_times in counter.items():
        if card == -1:     # 序列动作里有一个负的
            continue
        if 0 <= card < 52:
            matrix[card // 13, card % 13] = num_times
        elif card == 52:
            jokers[0] = num_times
        elif card == 53:
            jokers[1] = num_times
    return np.concatenate((matrix.flatten('F'), jokers))

def _get_one_hot_array(num_left_cards, max_num_cards, flag):
    if flag == 0:     # 级数的情况
        one_hot = np.zeros(max_num_cards)
        one_hot[num_left_cards - 1] = 1
    else:
        one_hot = np.zeros(max_num_cards+1)    # 剩余的牌（0-1阵格式）
        one_hot[num_left_cards] = 1
    return one_hot

def action_vector(id, action):
    if action[0] == 'PASS':
        return torch.from_numpy(-1*np.ones(83, dtype=np.int8))
    
    vec = np.zeros(83, dtype=np.int8) 
    vec[id] = 1
    vec[4 + TYPPES.index(action[0])] = 1
    vec[14+ STRENGTH.index(action[1])] = 1
    vec[29:] = card2array(card2num(action[2]))
    
    return torch.from_numpy(vec)

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
    cardre_value_s2v = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11,
                       "Q": 12, "K": 13}
    for i in temp:
        cardre[cardre_value_s2v[i[-1]]] +=1
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
            else:

                if minnum == onenum:
                    if mintwonum >= twonum:
                        if zeronum >= onenum + twonum:
                            st = []
                            st.append(10)
                else:
                    if zeronum >= onenum + twonum:
                        st = []
                        st.append(10)

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
    return newcards, bomb_info


class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        state, player_id = self.env.reset()
        obs = self.state_to_obs(self.parse(state, player_id), player_id)
        initial_position, initial_obs, x_no_action, z = _format_observation(obs, self.device)
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool) # 初始化为true

        return initial_position, initial_obs, state, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            )
        
    def step(self, action):
        next_state, next_player_id, done = self.env.step(action)
        obs = self.state_to_obs(self.parse(next_state, next_player_id), next_player_id)
        reward = 0.0 if not done else self.get_reward(self.env.game.temp_result)

        # self.episode_return += reward
        # episode_return = self.episode_return 
        episode_return = reward

        # if done:
        #     # 内部已经自动开启下一小局了
        #     self.episode_return = torch.zeros(1, 1)

        position, obs, x_no_action, z = _format_observation(obs, self.device)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        
        return position, obs, next_state, dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            )

    def close(self):
        self.env.close()

    def is_over(self):
        return self.env.is_over()
    
    def get_reward(self, order):
        cur_pos = self.env.game.round.current_player
        team = [cur_pos, (cur_pos + 2) % 4]
        rewards = {"1100": 3, "1010": 2, "1001": 1, "0110": -1, "0101": -2, "0011": -3}
        res = ""
        for i in order:
            if i in team:
                res += '1'
            else:
                res += '0'
        
        return rewards[res]

    def parse(self, state, player_id):
        assert type(state) == dict
        msg = {}
        rank_list = state['rank_list']
        play_team = state['play_team']
        msg['curPos'] = player_id
        msg['indexRange'] = len(state["actions"]) - 1
        msg['curAction'] = state['greaterAction']
        msg['curRank'] = CARD_RANK[rank_list[play_team]] # str类型
        msg['remain_cards'] = state['remain_cards']
        msg['pass_num'] = state['pass_num'][player_id]
        msg['my_pass_num'] = state['my_pass_num'][player_id]
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

    def proc_universal(self, handCards, cur_rank):
        res = np.zeros(12, dtype=np.int8)

        if handCards[(cur_rank-1)*4] == 0: # 手牌中不包含通配牌，返回全零数组
            return res

        res[0] = 1 # 存在通配牌
        rock_flag = 0
        for i in range(4):
            # 遍历每个花色
            left, right = 0, 5
            # 滑动窗口​​：以窗口大小5（right=5）在花色内滑动，检测连续牌组是否可用通配补全（最多缺1张）。
            temp = [handCards[i + j*4] if i+j*4 != (cur_rank-1)*4 else 0 for j in range(5)] 
            while right <= 12:
                zero_num = temp.count(0)
                if zero_num <= 1: # 允许用1张通配补缺
                    rock_flag = 1 # 存在顺子（如5连顺）或连对（如3连对）
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
        # num_count 记录每个点数（如A~K）的牌数，排除通配
        for i in range(4):
            for j in range(13):
                if handCards[i + j*4] != 0 and i + j*4 != (cur_rank-1)*4:
                    num_count[j] += 1
        num_max = max(num_count)
        if num_max >= 6:
            res[2:8] = 1 # 存在6张及以上相同点数的超级炸弹
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
    
    def state_to_obs(self, message, player_id):

        
        num_legal_actions = message['indexRange'] + 1
        legal_actions = [card2num(i[2]) for i in message['actionList']]
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
        
        # 同理，动作向量形状为 (1, 83)
        last_teammate_action = -1*np.ones(83, dtype=np.int8)                  # 队友最后的动作
        if (player_id + 2) % 4 not in self.env.game.round.result and len(message['trace']):
            # 队友没走，且出过牌
            action = []
            for record in message['trace']:
                if record[0] == (player_id+2) % 4:
                    action = record[1]
            if action != []:
                # 队友动作存在（出过牌）
                last_teammate_action = action_vector((player_id+2) % 4, action)
        
        last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)

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
                    ))
        
        obs = {
            'position': player_id,
            'x_batch': x_batch.astype(np.int8),
            'x_no_action': x_no_action.astype(np.float32),
            'z': history_act,
            'z_batch': z_batch.astype(np.int8),
            'legal_actions': my_action_batch.astype(np.int8) 
        }
        return obs
        
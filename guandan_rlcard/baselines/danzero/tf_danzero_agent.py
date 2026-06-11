from guandan_rlcard.game.player import GuandanPlayer as Player
from random import randint
import numpy as np
from functools import reduce
import time
import pickle
import os
import tensorflow as tf
from tensorflow.keras.backend import get_session, set_session
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

RANK = {
    '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8,
    'T':9, 'J':10, 'Q':11, 'K':12, 'A':13
}

CardToNum = {
    'H2':0, 'H3':1, 'H4':2, 'H5':3, 'H6':4, 'H7':5, 'H8':6, 'H9':7, 'HT':8, 'HJ':9, 'HQ':10, 'HK':11, 'HA':12,
    'S2':13, 'S3':14, 'S4':15, 'S5':16, 'S6':17, 'S7':18, 'S8':19, 'S9':20, 'ST':21, 'SJ':22, 'SQ':23, 'SK':24, 'SA':25,
    'C2':26, 'C3':27, 'C4':28, 'C5':29, 'C6':30, 'C7':31, 'C8':32, 'C9':33, 'CT':34, 'CJ':35, 'CQ':36, 'CK':37, 'CA':38,
    'D2':39, 'D3':40, 'D4':41, 'D5':42, 'D6':43, 'D7':44, 'D8':45, 'D9':46, 'DT':47, 'DJ':48, 'DQ':49, 'DK':50, 'DA':51,
    'SB':52, 'HR':53
}

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

def _get_one_hot_array(num_left_cards, max_num_cards, flag):
    if flag == 0:     # 级数的情况
        one_hot = np.zeros(max_num_cards)
        one_hot[num_left_cards - 1] = 1
    else:
        one_hot = np.zeros(max_num_cards+1)    # 剩余的牌（0-1阵格式）
        one_hot[num_left_cards] = 1
    return one_hot

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

def getlist(handcards, rank):
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
        return card_value_s2v[elem[1]]

    def mysort1(elem):
        return card_value_s2v2[elem[1]]

    if sorted_cards["Single"]:
        for singlecard in sorted_cards['Single']:
            single_actionlist.append(['Single', singlecard[-1], [singlecard]])
        single_actionlist.sort(key=mysort)

    if sorted_cards["Pair"]:
        for paircard in sorted_cards['Pair']:
            pair_actionlist.append(['Pair', paircard[0][-1], paircard])
        pair_actionlist.sort(key=mysort)

    if sorted_cards['Trips']:
        for tripcard in sorted_cards['Trips']:
            trips_actionlist.append(['Trips', tripcard[0][-1], tripcard])
        trips_actionlist.sort(key=mysort)

    if sorted_cards['Pair'] and sorted_cards['Trips']:
        for tripcard in sorted_cards['Trips']:
            for paircard in sorted_cards['Pair']:
                threetwo_actionlist.append(['ThreeWithTwo', tripcard[0][-1], tripcard + paircard])
        threetwo_actionlist.sort(key=mysort)

    
    if len(sorted_cards['Pair']) >= 3:
        for i in range(len(pair_actionlist) - 2):
            if card_value_s2v[pair_actionlist[i][1]] == card_value_s2v[pair_actionlist[i + 1][1]] - 1 and \
                    card_value_s2v[pair_actionlist[i + 1][1]] == card_value_s2v[pair_actionlist[i + 2][1]] - 1:
                action2 = pair_actionlist[i][-1] + pair_actionlist[i + 1][-1] + pair_actionlist[i + 2][-1]
                threepair_actionlist.append(['ThreePair', action2[0][-1], action2])
        threepair_actionlist.sort(key=mysort1)

    
    if len(sorted_cards['Trips']) >= 2:
        for i in range(len(trips_actionlist) - 1):
            if card_value_s2v[trips_actionlist[i][1]] == card_value_s2v[trips_actionlist[i + 1][1]] - 1:
                action3 = trips_actionlist[i][-1] + trips_actionlist[i + 1][-1]
                twotrips_actionlist.append(['TwoTrips', action3[0][-1], action3])
        twotrips_actionlist.sort(key=mysort1)

    if 'Straight' in sorted_cards.keys() and sorted_cards['Straight']:
        for straightcard in sorted_cards['Straight']:
            straight_actionlist.append(['Straight', straightcard[0][-1], straightcard])
        straight_actionlist.sort(key=mysort1)

    return single_actionlist + pair_actionlist + trips_actionlist + threepair_actionlist + threetwo_actionlist + twotrips_actionlist + straight_actionlist

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def placeholder(dtype=tf.float32, shape=None):
    return tf.placeholder(dtype=dtype, shape=combined_shape(None, shape))

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class GDModel():
    def __init__(self, observation_space, action_space, config=None, model_id='0', *args, **kwargs):
        with tf.variable_scope(model_id):
            self.x_ph = placeholder(shape=observation_space)

        # 输出张量
        self.values = None
        self.scope = model_id

        # Initialize Tensorflow session
        self.sess = get_session()
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_id = model_id
        self.config = config

        # 2. Build up model
        self.build()

        # Build assignment ops
        self._weight_ph = None
        self._to_assign = None
        self._nodes = None
        self._build_assign()

        # 参数初始化
        self.sess.run(tf.global_variables_initializer())    


    def set_weights(self, weights) -> None:
        feed_dict = {self._weight_ph[var.name]: weight
                     for (var, weight) in zip(tf.trainable_variables(scope=self.scope), weights)}
        self.sess.run(self._nodes, feed_dict=feed_dict)

    def _build_assign(self):
        self._weight_ph, self._to_assign = dict(), dict()
        variables = tf.trainable_variables(self.scope)
        for var in variables:
            self._weight_ph[var.name] = tf.placeholder(var.value().dtype, var.get_shape().as_list())
            self._to_assign[var.name] = var.assign(self._weight_ph[var.name])
        self._nodes = list(self._to_assign.values())

    def forward(self, x_batch):
        return self.sess.run(self.values, feed_dict={self.x_ph: x_batch})

    def build(self) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('v'):
                self.values = mlp(self.x_ph, [512, 512, 512, 512, 512, 1], activation='tanh',
                                            output_activation=None)


class DanzeroAgent(Player):
    ''' Danzero agent.
    '''
    def __init__(self, player_id, np_random):
        super().__init__(player_id, np_random)
        self.use_raw = True
        self.init_time = time.time()
        
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        
        self.mypos = player_id
        self.history_action = {0: [], 1: [], 2: [], 3:[]}
        self.action_seq = []
        self.action_order = [] # 记录出牌顺序(4个智能体是一样的)
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.other_left_hands = [2 for _ in range(54)]
        self.flag = 0
        self.over = []
        self.rank = {'self_rank': 1, 'oppo_rank': 1}
        self.tongji = {3: 0, 2: 0, 1: 0, -1: 0, -2: 0, -3: 0, 'all': 0}

        # 模型初始化
        self.model  = GDModel((567, ), (5, 216))
        with open(os.environ.get('GUANDAN_DANZERO_CKPT') or os.path.join(os.path.dirname(__file__), 'q_network.ckpt'), 'rb') as f:
            new_weights = pickle.load(f)
        self.model.set_weights(new_weights)

    def step(self, state):
        if not state['actions']:
            return []
        message = self.parse(state)
        
        observation = self.prepare(message)
        
        output = self.model.forward(observation['x_batch'])
        act_index = np.argmax(output)
        
        final = state['actions'][act_index]
        
        return final
    
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


    def prepare(self, message):
        num_legal_actions = message['indexRange'] + 1
        legal_actions = [card2num(i[2]) for i in message['actionList']]
        my_handcards = card2array(card2num(message['handCards']))   # 自己的手牌,54维
        print('my_handcards', my_handcards)
        
        my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

        universal_card_flag = self.proc_universal(my_handcards, RANK[message['curRank']])     # 万能牌的标志位, 12维
        # print('universal_card_flag', universal_card_flag)
        universal_card_flag_batch = np.repeat(universal_card_flag[np.newaxis, :],
                                   num_legal_actions, axis=0)

        other_hands = []       # 其余所有玩家手上剩余的牌，54维
        for i in range(54): 
            if self.other_left_hands[i] == 1:
                other_hands.append(i)
            elif self.other_left_hands[i] == 2:
                other_hands.append(i)
                other_hands.append(i)
        # print(self.mypos, "other handcards: ", other_hands)
        other_handcards = card2array(other_hands)      
        # print('other_handcards', other_handcards)
        other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

        last_action = []         # 最新的动作，54维
        if len(self.action_seq) > 0:
            last_action = card2array(self.action_seq[-1])
        else:
            last_action = card2array([-1])
        # print(last_action)
        last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)
        
        last_teammate_action = []               # 队友最后的动作， 54维
        if len(self.history_action[(self.mypos + 2) % 4]) > 0 and (self.mypos + 2) % 4 not in self.over:
            last_teammate_action = card2array(self.history_action[(self.mypos + 2) % 4][-1])
        else:
            last_teammate_action = card2array([-1])
        # print(last_teammate_action)
        last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)

        my_action_batch = np.zeros(my_handcards_batch.shape)     # 合法动作，54维
        for j, action in enumerate(legal_actions):
            my_action_batch[j, :] = card2array(action)

        down_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 1) % 4], 27, 1)   # 下家剩余的牌数， 28维
        
        # print(down_num_cards_left)
        down_num_cards_left_batch = np.repeat(down_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        teammate_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 2) % 4], 27, 1)   # 对家剩余的牌数
        
        # print(teammate_num_cards_left)
        teammate_num_cards_left_batch = np.repeat(teammate_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        up_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 3) % 4], 27, 1)   # 上家剩余的牌数
        
        # print(up_num_cards_left)
        up_num_cards_left_batch = np.repeat(up_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 1) % 4]) > 0:
            down_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 1) % 4]))    # 下家打过的牌， 54维
        else:
            down_played_cards = card2array([])
        
        # print(down_played_cards)
        down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 2) % 4]) > 0:
            teammate_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 2) % 4]))    # 对家打过的牌
        else:
            teammate_played_cards = card2array([])
        # print(teammate_played_cards)
        teammate_played_cards_batch = np.repeat(teammate_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 3) % 4]) > 0:
            up_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 3) % 4]))    # 上家打过的牌
        else:
            up_played_cards = card2array([])
        # print(up_played_cards)
        up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], num_legal_actions, axis=0)
 
        self_rank = _get_one_hot_array(RANK[message['selfRank']], 13, 0)         # 己方当前的级牌，13维
        # print(self_rank)
        self_rank_batch = np.repeat(self_rank[np.newaxis, :], num_legal_actions, axis=0)

        oppo_rank = _get_one_hot_array(RANK[message['oppoRank']], 13, 0)         # 敌方当前的级牌
        # print(oppo_rank)

        oppo_rank_batch = np.repeat(oppo_rank[np.newaxis, :], num_legal_actions, axis=0)

        cur_rank = _get_one_hot_array(RANK[message['curRank']], 13, 0)         # 当前的级牌
        # print(cur_rank)

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
                            cur_rank
                            ))

        obs = {
            'x_batch': x_batch.astype(np.float32),
            'legal_actions': legal_actions,
            'x_no_action': x_no_action.astype(np.float32),
          }
        return obs


    def reset(self):
        super().reset()
        self.history_action = {0: [], 1: [], 2: [], 3:[]}
        self.action_seq = []
        self.other_left_hands = [2 for _ in range(54)]
        self.flag = 0
        self.action_order = []
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.over = []
        # reward = self.get_reward(message)
        # self.tongji[reward] += 1
        self.tongji['all'] += 1
    
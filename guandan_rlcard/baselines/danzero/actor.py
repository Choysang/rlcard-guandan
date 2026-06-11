def _debug_print(*args, **kwargs):
    # Debug output from the original research code, silenced for
    # the open-source release.
    pass


import os
import numpy as np
import pickle
import torch
import io
from .model import MLPQNetwork

ActionNumber = 1

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class Action():
    def __init__(self) -> None:
        # 模型初始化，拉取DQN网络参数
        self.model_q = MLPQNetwork(567)
        with open(os.environ.get('GUANDAN_DANZERO_CKPT') or os.path.join(os.path.dirname(__file__), 'q_network.ckpt'), 'rb') as f:
            tf_weights = pickle.load(f)
        self.model_q.load_tf_weights(tf_weights)


    def step(self, state) -> int:
        states = state['x_batch']
        # legal_index = np.ones(ActionNumber)
        # state_no_action = state['x_no_action']
        
        indexs = self.model_q.get_max_n_index(states, ActionNumber)
        # _debug_print(indexs)
        # dqn_states = np.asarray(states[indexs])
        # top_actions = dqn_states[:, -54:].flatten()
        
        return indexs[0]

    def step_ppo(self, states) -> int:

        indexs , q_list = self.model_q.get_max_n_index_q(states, ActionNumber)

        return indexs , q_list


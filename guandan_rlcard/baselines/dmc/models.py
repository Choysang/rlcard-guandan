"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn

class PlayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(83, 54, batch_first=True)
        self.dense1 = nn.Linear(655 + 54, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            # training 时返回值，从而和目标做loss
            return dict(values=x)
        else:
            # epsilon-greedy 策略
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0] # x.shape[0]为随机整数上限，(1,)是随机张量的形状
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['1'] = PlayerModel
model_dict['2'] = PlayerModel
model_dict['3'] = PlayerModel
model_dict['0'] = PlayerModel

class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['1'] = PlayerModel().to(torch.device(device))
        self.models['2'] = PlayerModel().to(torch.device(device))
        self.models['3'] = PlayerModel().to(torch.device(device))
        self.models['0'] = PlayerModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        # 模块的所有参数（parameters）和缓冲区（如批归一化的均值和方差）会被移动到共享内存中。共享内存允许不同进程直接访问同一块物理内存，无需通过进程间通信（IPC）复制数据
        # 如果参数或缓冲区已经是 CUDA 张量（存储在 GPU 显存中），此方法不会执行任何操作，因为 CUDA 内存本身支持多进程访问
        self.models['1'].share_memory()
        self.models['2'].share_memory()
        self.models['3'].share_memory()
        self.models['0'].share_memory()

    def eval(self):
        self.models['1'].eval()
        self.models['2'].eval()
        self.models['3'].eval()
        self.models['0'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models

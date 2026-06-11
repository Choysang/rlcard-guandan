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
        self.lstm = nn.LSTM(216, 128, batch_first=True)
        self.dense1 = nn.Linear(817+ 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 256)
        self.li = nn.Linear(256, 1)


    def forward(self, z, x , return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        # print(lstm_out.shape)
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
        x = self.li(x)
        return dict(values=x)

            # return dict(action=action)

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

        self.base_model = PlayerModel().to(torch.device(device))
        
        self.models['1'] = self.base_model
        self.models['2'] = self.base_model
        self.models['3'] = self.base_model
        self.models['0'] = self.base_model

    def forward(self, position, z, x , training=False, flags=None):
        # model = self.models[position]
        return self.base_model.forward(z, x, training, flags)

    def share_memory(self):
        # 模块的所有参数（parameters）和缓冲区（如批归一化的均值和方差）会被移动到共享内存中。共享内存允许不同进程直接访问同一块物理内存，无需通过进程间通信（IPC）复制数据
        # 如果参数或缓冲区已经是 CUDA 张量（存储在 GPU 显存中），此方法不会执行任何操作，因为 CUDA 内存本身支持多进程访问
        # self.models['1'].share_memory()
        # self.models['2'].share_memory()
        # self.models['3'].share_memory()
        # self.models['0'].share_memory()
        # self.models['1']
        # self.models['2']
        # self.models['3']
        # self.models['0']
        self.base_model.share_memory()

    def eval(self):
        # self.models['1'].eval()
        # self.models['2'].eval()
        # self.models['3'].eval()
        # self.models['0'].eval()
        self.base_model.eval()

    def parameters(self, position):
        # return self.models[position].parameters()
        return self.base_model.parameters()

    def get_model(self, position):
        # return self.models[position]
        return self.base_model

    def get_models(self):
        return self.models

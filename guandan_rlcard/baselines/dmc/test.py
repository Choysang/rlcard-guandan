import sys, os
# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取父目录的父目录（上两级目录）
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

import torch
from .arguments import parser
from .models import Model
from .utils import create_optimizers
from .eval import run_evaluation
from compete_list import eval_agent_list
from .tracking import Tracking
import wandb

RANK = {
    '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8,
    'T':9, 'J':10, 'Q':11, 'K':12, 'A':13
}

TYPPES = ["Single", "Pair", "Trips", "ThreePair", "ThreeWithTwo", "TwoTrips", "Straight", "StraightFlush", "Bomb", "PASS"]
STRENGTH = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'SB', 'HR']

hands = ['H2', 'H2', 'D2', 'D2', 'S2', 'C3', 'S5', 'H6', 'H8', 'HQ', 'CQ', 'HK', 'DK', 'HA', 'HA', 'HR']


training_device = 'cpu'
device_iterator = ['cpu']

models = {}
for device in device_iterator:
    model = Model(device=device)
    model.share_memory()
    model.eval()
    models[device] = model

checkpoint_states = torch.load(
    '/home/yangrongxin/workspace/GuanDan/rlcard/baselines/dmc/douzero_checkpoints/douzero/model.tar', map_location=("cuda:"+str(training_device) if training_device != "cpu" else "cpu")
)


wandb.init(project='guandan_RL_test_eval', name='dmc_vs_rules')

# Learner model for training
learner_model = Model(device=training_device)

flags = parser.parse_args()
# Create optimizers
optimizers = create_optimizers(flags, learner_model)

for k in ['0', '1', '2', '3']:
    learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
    optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
    for device in device_iterator:
        models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
        
# 和每个规则打50局测胜率
win_episode_rates = []
win_game_rates = []

for step in range(5):
    for id, agent in enumerate(eval_agent_list):
        print(agent.name)
        win1, win2 = run_evaluation(agent, 10, learner_model.models['0'])
        win_episode_rates.append([win1, step, agent.name])
        win_game_rates.append([win2, step, agent.name])
        wandb.log({
            f"eval/ep_winrate_{agent.name}": win1,
            f"eval/game_winrate_{agent.name}": win2,
        }, step=step)
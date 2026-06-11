import numpy as np
from guandan_rlcard.envs.guandan_env import GuandanEnv
# from baselines.dmc.dmc_agent import DMCAgent
from guandan_rlcard.baselines.danzero_plus.ppo_agent import PPOAgent

def run_evaluation(agent, num_games, model):
    config = {'allow_step_back': True}
    env = GuandanEnv(config)
    
    random = np.random.RandomState()

    # Set agents
    env.set_agents([PPOAgent(0, random, model=model), agent(1, random), PPOAgent(2, random, model=model), agent(3, random)])
    # env.set_agents([agent(0, random) agent(1, random), agent(2, random), agent(3, random)])

    # Generate data from the environment
    gwins = []
    episode_count, game_count = 0, 0
    win0 = 0
    win1 = 0
    print(num_games)
    for x in range(num_games):
        trajectories, gwins, winner_team = env.run(x, is_training=True)  
        game_count = gwins[0] + gwins[1]
        episode_count += 1
        if winner_team == 0:
            win0 += 1
        else:
            win1 += 1

    # 大局胜率分布
    print(f'当前打了{episode_count}局掼蛋, 包括 {game_count} 个小局')
    print(f'0队大局胜率 {win0/episode_count}, 小局胜率 {gwins[0]/game_count}')
    print(f'1队大局胜率 {win1/episode_count}，小局胜率 {gwins[1]/game_count}')
    
    return win0/episode_count, gwins[0]/game_count

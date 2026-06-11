"""DanZero+ baseline: PPO on top of the DanZero framework
(arXiv:2312.02561), adapted interface (requires torch).

``ppo_agent.PPOAgent`` is the inference agent (give it a trained model;
fresh agents play with random weights), ``ppo.py``/``train.py`` run
training and ``ppo_agent_imperfect`` is the imperfect-information
variant. Builds on ``guandan_rlcard.baselines.danzero``.
"""

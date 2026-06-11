"""PPO self-play baseline (requires torch).

Modules:
    ppo_agent - PPOGuandanAgent, PPOMemory and RewardShaper
    models    - LSTM policy/value networks and the state encoder
    train     - standalone self-play training entry point
    evaluate_guandan_ppo - evaluation against the rule baselines

Nothing is imported eagerly here to keep torch optional::

    from guandan_rlcard.baselines.ppo.ppo_agent import PPOGuandanAgent
"""

"""PerfectDan: the author's own PPO self-play algorithm.

Decoupled LSTM policy/value networks over a rich state encoder, trained
by self-play in this environment; in our experiments it reaches high win
rates against the rule baselines and DanZero. Modules: ``ppo_agent``
(PPOGuandanAgent, PPOMemory, RewardShaper), ``models``, ``train``,
``evaluate_guandan_ppo``. Requires torch; nothing imports eagerly::

    from guandan_rlcard.baselines.perfectdan.ppo_agent import PPOGuandanAgent

Pretrained weights are distributed via the GitHub Releases page (too
large for the repository); place them under ``checkpoints/`` as the
evaluate script expects.
"""

"""Deep Monte-Carlo (DanZero-style) training framework (requires torch).

DouZero/DanZero-style distributed DMC training adapted to this
environment: ``dmc.py``/``train.py`` run self-play training,
``dmc_agent.DMCAgent`` is the inference agent (pass a trained model or a
checkpoint via the eval scripts; a fresh agent plays with random
weights). ``curriculum_train_guandan.py`` adds curriculum opponents.
"""

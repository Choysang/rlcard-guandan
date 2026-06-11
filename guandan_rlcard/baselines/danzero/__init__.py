"""DanZero baseline (paper algorithm, adapted interface).

DanZero masters Guandan with distributed Deep Monte-Carlo self-play
(arXiv:2210.17087). This package wraps the released value network for
inference in this environment: ``DanzeroAgent`` loads the pretrained
``q_network.ckpt`` shipped next to this file (override with the
``GUANDAN_DANZERO_CKPT`` environment variable). Requires torch;
``tf_danzero_agent`` is the original TensorFlow variant (optional).

    from guandan_rlcard.baselines.danzero.danzero_agent import DanzeroAgent
"""

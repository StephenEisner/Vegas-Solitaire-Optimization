"""
Reinforcement Learning solvers for Vegas Solitaire.

This module contains RL algorithms:
- TD Learning (Temporal Difference)
- Q-Learning
"""

from optimization.solvers.rl.td_learning import TDLearner, TDPolicy, train_td_policy
from optimization.solvers.rl.q_learning import QLearner, QPolicy, train_q_policy

__all__ = [
    'TDLearner',
    'TDPolicy',
    'train_td_policy',
    'QLearner',
    'QPolicy',
    'train_q_policy',
]

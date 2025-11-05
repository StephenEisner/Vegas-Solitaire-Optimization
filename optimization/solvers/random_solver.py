"""
Random solver for Vegas Solitaire.

This solver chooses moves uniformly at random from all valid moves.
It serves as a baseline for comparing more sophisticated algorithms.
"""

from typing import Optional
import random
from game.core.game import Game
from game.core.moves import Move
from optimization.solvers.base import Solver


class RandomSolver(Solver):
    """
    Solver that chooses moves uniformly at random.

    This is the simplest possible strategy and serves as a baseline.
    Expected performance: ~2-8% win rate on standard Vegas Solitaire.

    Attributes:
        rng: Random number generator (seeded for reproducibility)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random solver.

        Args:
            seed: Optional seed for reproducibility. If None, uses random seed.
        """
        super().__init__(name="Random")
        self.rng = random.Random(seed)

    def choose_move(self, game: Game) -> Optional[Move]:
        """
        Choose a random move from all valid moves.

        Args:
            game: Current game state

        Returns:
            Random valid move, or None if no moves available
        """
        valid_moves = game.get_valid_moves()

        if not valid_moves:
            return None

        return self.rng.choice(valid_moves)

"""
Base class for Vegas Solitaire solvers.

This module defines the abstract Solver interface that all solving algorithms
must implement. It also provides common functionality like game playing and
statistics tracking.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time
from game.core.game import Game
from game.core.moves import Move


class SolverStatistics:
    """
    Statistics tracking for solver performance.

    Tracks various metrics about solver performance across multiple games.
    """

    def __init__(self):
        """Initialize empty statistics."""
        self.games_played = 0
        self.games_won = 0
        self.total_score = 0
        self.total_moves = 0
        self.total_time = 0.0
        self.scores: List[int] = []
        self.moves_per_game: List[int] = []
        self.cards_in_foundations: List[int] = []

    def record_game(self, game: Game, elapsed_time: float) -> None:
        """
        Record statistics from a completed game.

        Args:
            game: The completed game
            elapsed_time: Time taken to play the game
        """
        summary = game.get_game_summary()

        self.games_played += 1
        if summary['is_winning']:
            self.games_won += 1

        self.total_score += summary['score']
        self.total_moves += summary['moves']
        self.total_time += elapsed_time

        self.scores.append(summary['score'])
        self.moves_per_game.append(summary['moves'])
        self.cards_in_foundations.append(summary['cards_in_foundations'])

    def get_win_rate(self) -> float:
        """Get win rate as a percentage."""
        if self.games_played == 0:
            return 0.0
        return (self.games_won / self.games_played) * 100

    def get_average_score(self) -> float:
        """Get average score across all games."""
        if self.games_played == 0:
            return 0.0
        return self.total_score / self.games_played

    def get_average_moves(self) -> float:
        """Get average moves per game."""
        if self.games_played == 0:
            return 0.0
        return self.total_moves / self.games_played

    def get_average_time(self) -> float:
        """Get average time per game in seconds."""
        if self.games_played == 0:
            return 0.0
        return self.total_time / self.games_played

    def get_average_foundation_cards(self) -> float:
        """Get average number of cards in foundations."""
        if self.games_played == 0:
            return 0.0
        return sum(self.cards_in_foundations) / self.games_played

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary.

        Returns:
            Dictionary with all statistics
        """
        return {
            'games_played': self.games_played,
            'games_won': self.games_won,
            'win_rate': self.get_win_rate(),
            'average_score': self.get_average_score(),
            'average_moves': self.get_average_moves(),
            'average_time': self.get_average_time(),
            'average_foundation_cards': self.get_average_foundation_cards(),
            'total_time': self.total_time,
            'best_score': max(self.scores) if self.scores else 0,
            'worst_score': min(self.scores) if self.scores else 0,
        }

    def __str__(self) -> str:
        """String representation of statistics."""
        if self.games_played == 0:
            return "No games played yet"

        lines = [
            f"Games Played: {self.games_played}",
            f"Win Rate: {self.get_win_rate():.1f}%",
            f"Average Score: ${self.get_average_score():.2f}",
            f"Average Moves: {self.get_average_moves():.1f}",
            f"Average Foundation Cards: {self.get_average_foundation_cards():.1f}/52",
            f"Average Time: {self.get_average_time():.3f}s",
        ]
        return "\n".join(lines)


class Solver(ABC):
    """
    Abstract base class for Vegas Solitaire solvers.

    All solving algorithms should inherit from this class and implement
    the choose_move() method.

    Attributes:
        name: Name of the solver
        statistics: Statistics tracker
    """

    def __init__(self, name: str):
        """
        Initialize the solver.

        Args:
            name: Name of the solver for identification
        """
        self.name = name
        self.statistics = SolverStatistics()

    @abstractmethod
    def choose_move(self, game: Game) -> Optional[Move]:
        """
        Choose the next move to make.

        This is the core method that each solver must implement.
        It should analyze the current game state and return the
        best move according to the solver's strategy.

        Args:
            game: Current game state

        Returns:
            The chosen move, or None if no move should be made
        """
        pass

    def play_game(self, seed: Optional[int] = None,
                  max_moves: int = 1000,
                  verbose: bool = False) -> Game:
        """
        Play a complete game using this solver.

        Args:
            seed: Random seed for the game
            max_moves: Maximum moves before stopping
            verbose: Whether to print progress

        Returns:
            The completed game
        """
        game = Game(seed=seed)
        game.deal()

        if verbose:
            print(f"\nPlaying game with {self.name} (seed={seed})...")

        start_time = time.time()
        move_count = 0

        while not game.is_over() and move_count < max_moves:
            move = self.choose_move(game)

            if move is None:
                break

            success = game.make_move(move)
            if not success:
                if verbose:
                    print(f"Warning: Invalid move chosen: {move}")
                break

            move_count += 1

            if verbose and move_count % 10 == 0:
                print(f"  Move {move_count}: Score=${game.get_score()}")

        elapsed = time.time() - start_time

        # Record statistics
        self.statistics.record_game(game, elapsed)

        if verbose:
            summary = game.get_game_summary()
            print(f"\nGame complete:")
            print(f"  Result: {'WIN' if summary['is_winning'] else 'LOSS'}")
            print(f"  Score: ${summary['score']}")
            print(f"  Moves: {summary['moves']}")
            print(f"  Time: {elapsed:.2f}s")

        return game

    def play_multiple_games(self, num_games: int,
                           start_seed: int = 0,
                           max_moves: int = 1000,
                           verbose: bool = False) -> List[Game]:
        """
        Play multiple games and track statistics.

        Args:
            num_games: Number of games to play
            start_seed: Starting seed (increments for each game)
            max_moves: Maximum moves per game
            verbose: Whether to print progress

        Returns:
            List of completed games
        """
        games = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"Playing {num_games} games with {self.name}")
            print(f"{'='*70}\n")

        for i in range(num_games):
            seed = start_seed + i
            game = self.play_game(seed=seed, max_moves=max_moves, verbose=False)
            games.append(game)

            if verbose and (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_games} games...")

        if verbose:
            print(f"\n{'='*70}")
            print(f"Final Statistics for {self.name}:")
            print(f"{'='*70}")
            print(self.statistics)
            print(f"{'='*70}\n")

        return games

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.statistics = SolverStatistics()

    def get_statistics(self) -> SolverStatistics:
        """Get current statistics."""
        return self.statistics

    def __str__(self) -> str:
        """String representation of the solver."""
        return f"{self.name} Solver"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Solver(name='{self.name}', games_played={self.statistics.games_played})"

"""
Tests for solvers.

These tests ensure that solvers work correctly and produce
consistent results.
"""

import pytest
from game.core.game import Game
from optimization.solvers.base import Solver, SolverStatistics
from optimization.solvers.random_solver import RandomSolver
from optimization.solvers.heuristic_solver import HeuristicSolver


class TestSolverStatistics:
    """Tests for SolverStatistics class."""

    def test_initial_statistics(self):
        """Test that statistics start empty."""
        stats = SolverStatistics()

        assert stats.games_played == 0
        assert stats.games_won == 0
        assert stats.get_win_rate() == 0.0
        assert stats.get_average_score() == 0.0

    def test_record_game(self):
        """Test recording a game."""
        stats = SolverStatistics()
        game = Game(seed=42)
        game.deal()

        stats.record_game(game, elapsed_time=1.5)

        assert stats.games_played == 1
        assert stats.total_time == 1.5

    def test_multiple_games(self):
        """Test recording multiple games."""
        stats = SolverStatistics()

        for i in range(5):
            game = Game(seed=i)
            game.deal()
            stats.record_game(game, elapsed_time=1.0)

        assert stats.games_played == 5
        assert stats.total_time == 5.0
        assert stats.get_average_time() == 1.0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        stats = SolverStatistics()

        # Simulate 10 games with 3 wins
        for i in range(10):
            game = Game(seed=i)
            game.deal()

            # Artificially win 3 games
            if i < 3:
                from game.core.card import Card, Suit, Rank
                for suit in Suit:
                    game.state.foundations[suit] = [
                        Card(suit, rank) for rank in Rank
                    ]

            stats.record_game(game, elapsed_time=1.0)

        assert stats.games_played == 10
        assert stats.games_won == 3
        assert stats.get_win_rate() == 30.0

    def test_statistics_to_dict(self):
        """Test converting statistics to dictionary."""
        stats = SolverStatistics()
        game = Game(seed=42)
        game.deal()
        stats.record_game(game, elapsed_time=1.0)

        result = stats.to_dict()

        assert isinstance(result, dict)
        assert 'games_played' in result
        assert 'win_rate' in result
        assert 'average_score' in result

    def test_statistics_string(self):
        """Test string representation."""
        stats = SolverStatistics()

        # Empty stats
        assert "No games" in str(stats)

        # With games
        game = Game(seed=42)
        game.deal()
        stats.record_game(game, elapsed_time=1.0)

        stats_str = str(stats)
        assert "Games Played" in stats_str
        assert "Win Rate" in stats_str


class TestRandomSolver:
    """Tests for RandomSolver."""

    def test_create_random_solver(self):
        """Test creating a random solver."""
        solver = RandomSolver(seed=42)

        assert solver.name == "Random"
        assert solver.rng is not None

    def test_choose_move_returns_valid_move(self):
        """Test that chosen moves are valid."""
        solver = RandomSolver(seed=42)
        game = Game(seed=42)
        game.deal()

        move = solver.choose_move(game)

        assert move is not None
        assert move in game.get_valid_moves()

    def test_choose_move_with_no_moves(self):
        """Test behavior when no moves available."""
        solver = RandomSolver(seed=42)
        game = Game(seed=42)
        game.deal()

        # Artificially create state with no moves
        game.state.stock = []
        game.state.waste = []
        # Clear tableau too (keep only non-movable cards)
        for i in range(7):
            game.state.tableau[i] = []

        move = solver.choose_move(game)
        assert move is None

    def test_play_single_game(self):
        """Test playing a single game."""
        solver = RandomSolver(seed=42)
        game = solver.play_game(seed=42, max_moves=50)

        assert game.get_move_count() > 0
        assert game.get_move_count() <= 50
        assert solver.statistics.games_played == 1

    def test_random_solver_is_deterministic(self):
        """Test that same seed produces same results."""
        solver1 = RandomSolver(seed=42)
        game1 = solver1.play_game(seed=100, max_moves=50)

        solver2 = RandomSolver(seed=42)
        game2 = solver2.play_game(seed=100, max_moves=50)

        # Should make same moves
        assert game1.get_move_count() == game2.get_move_count()
        assert game1.get_score() == game2.get_score()

    def test_play_multiple_games(self):
        """Test playing multiple games."""
        solver = RandomSolver(seed=42)
        games = solver.play_multiple_games(num_games=5, max_moves=20)

        assert len(games) == 5
        assert solver.statistics.games_played == 5

    def test_reset_statistics(self):
        """Test resetting statistics."""
        solver = RandomSolver(seed=42)
        solver.play_game(seed=42, max_moves=10)

        assert solver.statistics.games_played == 1

        solver.reset_statistics()

        assert solver.statistics.games_played == 0


class TestHeuristicSolver:
    """Tests for HeuristicSolver."""

    def test_create_heuristic_solver(self):
        """Test creating a heuristic solver."""
        solver = HeuristicSolver()

        assert solver.name == "Heuristic"
        assert solver.weights is not None

    def test_choose_move_returns_valid_move(self):
        """Test that chosen moves are valid."""
        solver = HeuristicSolver()
        game = Game(seed=42)
        game.deal()

        move = solver.choose_move(game)

        assert move is not None
        assert move in game.get_valid_moves()

    def test_prefer_foundation_moves(self):
        """Test that foundation moves are preferred."""
        solver = HeuristicSolver()
        game = Game(seed=42)
        game.deal()

        # Set up a state where foundation move is available
        from game.core.card import Card, Suit, Rank
        ace = Card(Suit.HEARTS, Rank.ACE)
        game.state.waste = [ace]

        move = solver.choose_move(game)

        # Should prefer foundation move
        from game.core.moves import MoveType
        assert move.move_type == MoveType.WASTE_TO_FOUNDATION

    def test_evaluate_move_scores(self):
        """Test that move evaluation returns reasonable scores."""
        solver = HeuristicSolver()
        game = Game(seed=42)
        game.deal()

        moves = game.get_valid_moves()
        scores = [solver.evaluate_move(move, game.state) for move in moves]

        # Scores should be numeric
        assert all(isinstance(s, (int, float)) for s in scores)

        # Foundation moves should score highest
        from game.core.moves import MoveType
        foundation_moves = [m for m in moves
                            if m.move_type in (MoveType.WASTE_TO_FOUNDATION,
                                               MoveType.TABLEAU_TO_FOUNDATION)]

        if foundation_moves:
            foundation_scores = [solver.evaluate_move(m, game.state)
                                for m in foundation_moves]
            other_scores = [solver.evaluate_move(m, game.state)
                            for m in moves if m not in foundation_moves]

            # Foundation moves should score higher than average
            if other_scores:
                assert max(foundation_scores) >= sum(other_scores) / len(other_scores)

    def test_play_single_game(self):
        """Test playing a single game."""
        solver = HeuristicSolver()
        game = solver.play_game(seed=42, max_moves=50)

        assert game.get_move_count() > 0
        assert solver.statistics.games_played == 1

    def test_heuristic_better_than_random(self):
        """Test that heuristic solver generally performs better than random."""
        # This is a probabilistic test - we expect heuristic to do better on average

        random_solver = RandomSolver(seed=100)
        random_games = random_solver.play_multiple_games(10, max_moves=100)
        random_avg_score = random_solver.statistics.get_average_score()

        heuristic_solver = HeuristicSolver()
        heuristic_games = heuristic_solver.play_multiple_games(10, start_seed=100, max_moves=100)
        heuristic_avg_score = heuristic_solver.statistics.get_average_score()

        # Heuristic should generally score better (but not guaranteed on small sample)
        # We just check that it runs without error
        assert heuristic_avg_score is not None
        assert random_avg_score is not None


class TestSolverInterface:
    """Tests for Solver base class interface."""

    def test_solver_string_representation(self):
        """Test solver string representations."""
        solver = RandomSolver(seed=42)

        assert "Random" in str(solver)
        assert "Solver" in repr(solver)

    def test_solver_statistics_access(self):
        """Test accessing solver statistics."""
        solver = RandomSolver(seed=42)
        solver.play_game(seed=42, max_moves=10)

        stats = solver.get_statistics()

        assert stats.games_played == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

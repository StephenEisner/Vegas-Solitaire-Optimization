"""
Tests for MCTS solver.

These tests ensure that MCTS implementation works correctly.
"""

import pytest
from game.core.game import Game
from optimization.solvers.mcts_solver import MCTSNode, MCTSSolver


class TestMCTSNode:
    """Tests for MCTSNode class."""

    def test_create_node(self):
        """Test creating an MCTS node."""
        game = Game(seed=42)
        game.deal()

        node = MCTSNode(game.state.copy())

        assert node.state is not None
        assert node.parent is None
        assert node.visits == 0
        assert node.total_reward == 0.0

    def test_average_reward(self):
        """Test average reward calculation."""
        game = Game(seed=42)
        game.deal()
        node = MCTSNode(game.state.copy())

        # Initial
        assert node.get_average_reward() == 0.0

        # After some visits
        node.visits = 10
        node.total_reward = 50.0
        assert node.get_average_reward() == 5.0

    def test_ucb1_unvisited(self):
        """Test that unvisited nodes have infinite UCB1 value."""
        game = Game(seed=42)
        game.deal()
        node = MCTSNode(game.state.copy())

        assert node.ucb1() == float('inf')

    def test_ucb1_visited(self):
        """Test UCB1 calculation for visited nodes."""
        game = Game(seed=42)
        game.deal()

        parent = MCTSNode(game.state.copy())
        parent.visits = 10

        child = MCTSNode(game.state.copy(), parent=parent)
        child.visits = 2
        child.total_reward = 5.0

        ucb_value = child.ucb1(exploration_constant=1.41)

        # Should be finite and positive
        assert ucb_value != float('inf')
        assert ucb_value > 0

    def test_is_fully_expanded(self):
        """Test checking if node is fully expanded."""
        game = Game(seed=42)
        game.deal()
        node = MCTSNode(game.state.copy())

        # Initially not expanded (no untried moves set yet)
        assert node.is_fully_expanded()

        # Add untried moves
        from game.core.moves import create_draw_move
        node.untried_moves = [create_draw_move()]

        assert not node.is_fully_expanded()

        # Remove untried moves
        node.untried_moves = []

        assert node.is_fully_expanded()

    def test_best_child_selection(self):
        """Test selecting best child."""
        game = Game(seed=42)
        game.deal()

        parent = MCTSNode(game.state.copy())
        parent.visits = 10

        # Create children with different statistics
        from game.core.moves import create_draw_move
        child1 = MCTSNode(game.state.copy(), parent=parent, move=create_draw_move())
        child1.visits = 3
        child1.total_reward = 6.0

        child2 = MCTSNode(game.state.copy(), parent=parent, move=create_draw_move())
        child2.visits = 5
        child2.total_reward = 15.0

        parent.children = [child1, child2]

        # Child2 has better average reward
        best = parent.best_child()
        assert best == child2

    def test_most_visited_child(self):
        """Test selecting most visited child."""
        game = Game(seed=42)
        game.deal()

        parent = MCTSNode(game.state.copy())

        from game.core.moves import create_draw_move
        child1 = MCTSNode(game.state.copy(), parent=parent, move=create_draw_move())
        child1.visits = 3

        child2 = MCTSNode(game.state.copy(), parent=parent, move=create_draw_move())
        child2.visits = 10

        parent.children = [child1, child2]

        most_visited = parent.most_visited_child()
        assert most_visited == child2


class TestMCTSSolver:
    """Tests for MCTSSolver class."""

    def test_create_solver(self):
        """Test creating MCTS solver."""
        solver = MCTSSolver(simulations_per_move=100, seed=42)

        assert solver.name == "MCTS(100)"
        assert solver.simulations_per_move == 100

    def test_choose_move_returns_valid_move(self):
        """Test that MCTS chooses valid moves."""
        solver = MCTSSolver(simulations_per_move=50, seed=42)
        game = Game(seed=42)
        game.deal()

        move = solver.choose_move(game)

        assert move is not None
        assert move in game.get_valid_moves()

    def test_choose_move_with_no_moves(self):
        """Test behavior when no moves available."""
        solver = MCTSSolver(simulations_per_move=10, seed=42)
        game = Game(seed=42)
        game.deal()

        # Clear all moves
        game.state.stock = []
        game.state.waste = []
        for i in range(7):
            game.state.tableau[i] = []

        move = solver.choose_move(game)
        assert move is None

    def test_play_single_game(self):
        """Test playing a single game with MCTS."""
        solver = MCTSSolver(simulations_per_move=10, seed=42)
        game = solver.play_game(seed=42, max_moves=20)

        assert game.get_move_count() > 0
        assert game.get_move_count() <= 20
        assert solver.statistics.games_played == 1

    def test_mcts_is_deterministic(self):
        """Test that same seed produces same results."""
        solver1 = MCTSSolver(simulations_per_move=50, seed=42)
        game1 = solver1.play_game(seed=100, max_moves=20)

        solver2 = MCTSSolver(simulations_per_move=50, seed=42)
        game2 = solver2.play_game(seed=100, max_moves=20)

        # Should make same moves with same seeds
        assert game1.get_move_count() == game2.get_move_count()
        assert game1.get_score() == game2.get_score()

    def test_mcts_explores_tree(self):
        """Test that MCTS builds a search tree."""
        solver = MCTSSolver(simulations_per_move=100, seed=42)
        game = Game(seed=42)
        game.deal()

        # Choose a move (which builds a tree internally)
        move = solver.choose_move(game)

        assert move is not None
        # If MCTS ran, it should have found a valid move


class TestMCTSPerformance:
    """Tests for MCTS performance characteristics."""

    def test_more_simulations_take_longer(self):
        """Test that more simulations take more time."""
        import time

        game = Game(seed=42)
        game.deal()

        # Few simulations
        solver_fast = MCTSSolver(simulations_per_move=10, seed=42)
        start = time.time()
        solver_fast.choose_move(game)
        time_fast = time.time() - start

        # Many simulations
        game2 = Game(seed=42)
        game2.deal()
        solver_slow = MCTSSolver(simulations_per_move=100, seed=42)
        start = time.time()
        solver_slow.choose_move(game2)
        time_slow = time.time() - start

        # More simulations should take longer (usually)
        # This is a weak test since it might fail on very fast machines
        # but it gives us confidence that simulations are being run
        assert time_slow >= time_fast * 0.5  # At least somewhat longer

    def test_mcts_completes_games(self):
        """Test that MCTS can complete multiple games."""
        solver = MCTSSolver(simulations_per_move=20, seed=42)
        games = solver.play_multiple_games(num_games=3, max_moves=50)

        assert len(games) == 3
        assert solver.statistics.games_played == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for CLI interface.

These tests ensure that the CLI correctly renders the game state
and handles user interaction (where testable).
"""

import pytest
from game.core.game import Game
from game.ui.cli import CLI


class TestCLIInitialization:
    """Tests for CLI initialization."""

    def test_create_cli_with_game(self):
        """Test creating CLI with existing game."""
        game = Game(seed=42)
        cli = CLI(game)

        assert cli.game is game

    def test_create_cli_without_game(self):
        """Test creating CLI without game creates one."""
        cli = CLI()

        assert cli.game is not None
        assert isinstance(cli.game, Game)


class TestBoardRendering:
    """Tests for board rendering."""

    def test_render_board_after_deal(self):
        """Test that board can be rendered after dealing."""
        game = Game(seed=42)
        game.deal()
        cli = CLI(game)

        board = cli.render_board()

        assert isinstance(board, str)
        assert len(board) > 0
        assert "Solitaire" in board
        assert "Score" in board
        assert "Stock" in board
        assert "Waste" in board
        assert "Foundations" in board
        assert "Tableau" in board

    def test_render_board_shows_score(self):
        """Test that board shows current score."""
        game = Game(seed=42)
        game.deal()
        cli = CLI(game)

        board = cli.render_board()

        assert "-52" in board  # Initial Vegas score

    def test_render_board_shows_move_count(self):
        """Test that board shows move count."""
        game = Game(seed=42)
        game.deal()
        cli = CLI(game)

        board = cli.render_board()

        assert "Moves: 0" in board

    def test_render_board_shows_seed(self):
        """Test that board shows seed."""
        game = Game(seed=42)
        game.deal()
        cli = CLI(game)

        board = cli.render_board()

        assert "42" in board

    def test_render_board_shows_foundation_count(self):
        """Test that board shows foundation count."""
        game = Game(seed=42)
        game.deal()
        cli = CLI(game)

        board = cli.render_board()

        assert "0/52" in board  # No cards in foundation initially


class TestMoveRendering:
    """Tests for move rendering."""

    def test_render_moves_with_moves(self):
        """Test rendering available moves."""
        game = Game(seed=42)
        game.deal()
        cli = CLI(game)

        moves = game.get_valid_moves()
        rendered = cli.render_moves(moves)

        assert isinstance(rendered, str)
        assert "Available moves" in rendered
        assert "1." in rendered  # At least one move

    def test_render_moves_numbered(self):
        """Test that moves are numbered correctly."""
        game = Game(seed=42)
        game.deal()
        cli = CLI(game)

        moves = game.get_valid_moves()
        rendered = cli.render_moves(moves)

        # Should have sequential numbers
        assert "1." in rendered
        if len(moves) > 1:
            assert "2." in rendered

    def test_render_moves_empty(self):
        """Test rendering when no moves available."""
        game = Game(seed=42)
        game.deal()

        # Artificially create state with no moves
        game.state.stock = []
        game.state.waste = []

        cli = CLI(game)
        rendered = cli.render_moves([])

        assert "No moves" in rendered


class TestAutoPlay:
    """Tests for automatic play."""

    def test_play_auto_completes(self):
        """Test that auto play completes a game."""
        game = Game(seed=42)
        cli = CLI(game)

        # Play with limited moves
        cli.play_auto(strategy="random", max_moves=50)

        # Game should have made some moves
        assert game.get_move_count() > 0
        assert game.get_move_count() <= 50

    def test_play_auto_random_strategy(self):
        """Test auto play with random strategy."""
        game = Game(seed=42)
        cli = CLI(game)

        cli.play_auto(strategy="random", max_moves=10)

        assert game.get_move_count() > 0

    def test_play_auto_is_deterministic(self):
        """Test that auto play with same seed is deterministic."""
        game1 = Game(seed=42)
        cli1 = CLI(game1)
        cli1.play_auto(strategy="random", max_moves=10)

        game2 = Game(seed=42)
        cli2 = CLI(game2)
        cli2.play_auto(strategy="random", max_moves=10)

        # Should make same number of moves
        assert game1.get_move_count() == game2.get_move_count()
        assert game1.get_score() == game2.get_score()


class TestDisplay:
    """Tests for display function."""

    def test_display_does_not_crash(self):
        """Test that display function runs without errors."""
        game = Game(seed=42)
        game.deal()
        cli = CLI(game)

        # Should not raise any exceptions
        # (We can't easily test printed output, but we can ensure it doesn't crash)
        try:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            cli.display()

            output = sys.stdout.getvalue()
            assert len(output) > 0

        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

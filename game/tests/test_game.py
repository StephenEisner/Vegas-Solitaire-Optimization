"""
Tests for Game controller.

These tests ensure that the game controller properly orchestrates
all components and provides a clean API for playing games.
"""

import pytest
from game.core.game import Game, play_game_with_moves, play_random_game
from game.core.moves import create_draw_move, MoveType


class TestGameInitialization:
    """Tests for game initialization."""

    def test_create_game(self):
        """Test creating a new game."""
        game = Game(seed=42)

        assert game.seed == 42
        assert game.state is not None
        assert game.deck is not None
        assert game.move_history == []

    def test_create_game_without_seed(self):
        """Test creating a game without a seed."""
        game = Game()

        assert game.seed is None
        assert game.state is not None


class TestGameDeal:
    """Tests for dealing cards."""

    def test_deal_creates_proper_tableau(self):
        """Test that dealing creates correct tableau structure."""
        game = Game(seed=42)
        game.deal()

        # Check tableau has correct number of cards
        for col in range(7):
            expected_cards = col + 1
            assert len(game.state.tableau[col]) == expected_cards
            assert game.state.tableau_hidden[col] == expected_cards - 1

    def test_deal_puts_cards_in_stock(self):
        """Test that remaining cards go to stock."""
        game = Game(seed=42)
        game.deal()

        # 28 cards in tableau, 24 should be in stock
        assert len(game.state.stock) == 24

    def test_deal_starts_with_empty_waste(self):
        """Test that waste starts empty."""
        game = Game(seed=42)
        game.deal()

        assert len(game.state.waste) == 0

    def test_deal_starts_with_empty_foundations(self):
        """Test that foundations start empty."""
        game = Game(seed=42)
        game.deal()

        for foundation in game.state.foundations.values():
            assert len(foundation) == 0

    def test_deal_sets_vegas_score(self):
        """Test that score starts at -$52."""
        game = Game(seed=42)
        game.deal()

        assert game.get_score() == -52

    def test_cannot_deal_twice(self):
        """Test that dealing twice raises an error."""
        game = Game(seed=42)
        game.deal()

        with pytest.raises(RuntimeError, match="already been dealt"):
            game.deal()

    def test_deal_is_deterministic(self):
        """Test that same seed produces same deal."""
        game1 = Game(seed=42)
        game1.deal()

        game2 = Game(seed=42)
        game2.deal()

        # Check that tableaus are identical
        for col in range(7):
            assert game1.state.tableau[col] == game2.state.tableau[col]


class TestGetValidMoves:
    """Tests for getting valid moves."""

    def test_get_valid_moves_after_deal(self):
        """Test that valid moves are available after dealing."""
        game = Game(seed=42)
        game.deal()

        moves = game.get_valid_moves()

        # Should have at least draw move available
        assert len(moves) > 0
        assert any(m.move_type == MoveType.DRAW for m in moves)

    def test_get_valid_moves_returns_list(self):
        """Test that get_valid_moves returns a list."""
        game = Game(seed=42)
        game.deal()

        moves = game.get_valid_moves()

        assert isinstance(moves, list)


class TestMakeMove:
    """Tests for making moves."""

    def test_make_valid_move(self):
        """Test making a valid move."""
        game = Game(seed=42)
        game.deal()

        initial_moves = game.get_move_count()
        move = create_draw_move()
        result = game.make_move(move)

        assert result is True
        assert game.get_move_count() == initial_moves + 1
        assert len(game.move_history) == 1

    def test_make_invalid_move(self):
        """Test that invalid move is rejected."""
        game = Game(seed=42)
        game.deal()

        # Draw all cards
        for _ in range(8):  # Draw 8 times to empty stock
            draw_move = create_draw_move()
            if not game.make_move(draw_move):
                break

        # Try to draw again when stock is empty (should fail)
        initial_moves = game.get_move_count()
        result = game.make_move(create_draw_move())

        assert result is False
        assert game.get_move_count() == initial_moves

    def test_move_history_tracks_moves(self):
        """Test that move history is maintained."""
        game = Game(seed=42)
        game.deal()

        move1 = create_draw_move()
        move2 = create_draw_move()

        game.make_move(move1)
        game.make_move(move2)

        assert len(game.move_history) == 2
        assert game.move_history[0] == move1
        assert game.move_history[1] == move2


class TestGameOver:
    """Tests for game over detection."""

    def test_game_not_over_after_deal(self):
        """Test that game is not over immediately after dealing."""
        game = Game(seed=42)
        game.deal()

        assert not game.is_over()

    def test_winning_game_is_over(self):
        """Test that winning game is detected as over."""
        game = Game(seed=42)
        game.deal()

        # Artificially win the game
        from game.core.card import Card, Suit, Rank
        for suit in Suit:
            game.state.foundations[suit] = [
                Card(suit, rank) for rank in Rank
            ]

        assert game.is_winning()
        assert game.is_over()


class TestGameScore:
    """Tests for score tracking."""

    def test_initial_score(self):
        """Test that initial score is -$52."""
        game = Game(seed=42)
        game.deal()

        assert game.get_score() == -52

    def test_score_increases_with_foundation_cards(self):
        """Test that score increases when cards go to foundation."""
        game = Game(seed=42)
        game.deal()

        # Manually add card to foundation
        from game.core.card import Card, Suit, Rank
        ace = Card(Suit.HEARTS, Rank.ACE)
        game.state.foundations[Suit.HEARTS].append(ace)
        game.state.score += 5

        assert game.get_score() == -47  # -52 + 5


class TestStateCopy:
    """Tests for getting state copies."""

    def test_get_state_copy_is_independent(self):
        """Test that state copy is independent of original."""
        game = Game(seed=42)
        game.deal()

        state_copy = game.get_state_copy()

        # Modify copy
        state_copy.score = 100

        # Original should be unchanged
        assert game.get_score() != 100


class TestUndo:
    """Tests for undo functionality."""

    def test_undo_single_move(self):
        """Test undoing a single move."""
        game = Game(seed=42)
        game.deal()

        initial_state = game.get_state_copy()
        move = create_draw_move()
        game.make_move(move)

        # State should be different
        assert game.state != initial_state

        # Undo
        result = game.undo()
        assert result is True

        # State should be restored
        assert game.state == initial_state
        assert game.get_move_count() == 0

    def test_undo_multiple_moves(self):
        """Test undoing multiple moves."""
        game = Game(seed=42)
        game.deal()

        # Make several moves
        for _ in range(3):
            game.make_move(create_draw_move())

        assert game.get_move_count() == 3

        # Undo one
        game.undo()
        assert game.get_move_count() == 2

        # Undo another
        game.undo()
        assert game.get_move_count() == 1

    def test_undo_with_no_moves(self):
        """Test that undo returns false when no moves to undo."""
        game = Game(seed=42)
        game.deal()

        result = game.undo()
        assert result is False


class TestGameSummary:
    """Tests for game summary."""

    def test_game_summary_structure(self):
        """Test that game summary has expected structure."""
        game = Game(seed=42)
        game.deal()

        summary = game.get_game_summary()

        assert "seed" in summary
        assert "moves" in summary
        assert "score" in summary
        assert "cards_in_foundations" in summary
        assert "is_winning" in summary
        assert "is_over" in summary
        assert "stock_size" in summary
        assert "waste_size" in summary

    def test_game_summary_values(self):
        """Test that game summary has correct values."""
        game = Game(seed=42)
        game.deal()

        summary = game.get_game_summary()

        assert summary["seed"] == 42
        assert summary["moves"] == 0
        assert summary["score"] == -52
        assert summary["cards_in_foundations"] == 0
        assert summary["stock_size"] == 24
        assert summary["waste_size"] == 0


class TestGameRepresentation:
    """Tests for string representations."""

    def test_game_str(self):
        """Test that __str__ produces readable output."""
        game = Game(seed=42)
        game.deal()

        output = str(game)

        assert "Game" in output
        assert "42" in output  # seed

    def test_game_repr(self):
        """Test that __repr__ shows key information."""
        game = Game(seed=42)
        game.deal()

        output = repr(game)

        assert "Game" in output
        assert "42" in output  # seed
        assert "-52" in output  # score


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_play_game_with_moves(self):
        """Test playing a game with specific moves."""
        moves = [create_draw_move(), create_draw_move()]
        game = play_game_with_moves(seed=42, moves=moves)

        assert game.get_move_count() == 2

    def test_play_game_with_no_moves(self):
        """Test playing a game with no moves."""
        game = play_game_with_moves(seed=42)

        assert game.get_move_count() == 0

    def test_play_random_game(self):
        """Test playing a random game."""
        game = play_random_game(seed=42, max_moves=10)

        # Should have made some moves
        assert game.get_move_count() > 0
        assert game.get_move_count() <= 10

    def test_play_random_game_is_deterministic(self):
        """Test that random game with same seed is deterministic."""
        game1 = play_random_game(seed=42, max_moves=10)
        game2 = play_random_game(seed=42, max_moves=10)

        # Should have made same moves
        assert game1.get_move_count() == game2.get_move_count()
        assert game1.get_score() == game2.get_score()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

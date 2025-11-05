"""
Tests for rules engine.

These tests ensure that move validation, generation, and application
work correctly for all move types.
"""

import pytest
from game.core.rules import (
    can_move_to_foundation, can_move_to_tableau,
    is_valid_sequence, get_valid_moves, apply_move, is_valid_move
)
from game.core.state import GameState
from game.core.card import Card, Suit, Rank
from game.core.moves import (
    MoveType, create_draw_move, create_recycle_move,
    create_waste_to_foundation_move, create_waste_to_tableau_move,
    create_tableau_to_foundation_move, create_tableau_to_tableau_move
)


class TestFoundationValidation:
    """Tests for foundation move validation."""

    def test_ace_to_empty_foundation(self):
        """Test that Aces can start empty foundations."""
        state = GameState()
        ace = Card(Suit.HEARTS, Rank.ACE)

        assert can_move_to_foundation(ace, state)

    def test_two_to_empty_foundation_fails(self):
        """Test that non-Aces cannot start foundations."""
        state = GameState()
        two = Card(Suit.HEARTS, Rank.TWO)

        assert not can_move_to_foundation(two, state)

    def test_correct_sequence_on_foundation(self):
        """Test that cards must follow correct rank sequence."""
        state = GameState()
        state.foundations[Suit.HEARTS] = [Card(Suit.HEARTS, Rank.ACE)]

        two = Card(Suit.HEARTS, Rank.TWO)
        three = Card(Suit.HEARTS, Rank.THREE)

        assert can_move_to_foundation(two, state)
        assert not can_move_to_foundation(three, state)

    def test_wrong_suit_foundation_fails(self):
        """Test that suits must match on foundations."""
        state = GameState()
        state.foundations[Suit.HEARTS] = [Card(Suit.HEARTS, Rank.ACE)]

        spades_two = Card(Suit.SPADES, Rank.TWO)

        assert not can_move_to_foundation(spades_two, state)


class TestTableauValidation:
    """Tests for tableau move validation."""

    def test_king_to_empty_tableau(self):
        """Test that only Kings can go on empty tableau."""
        king = Card(Suit.HEARTS, Rank.KING)
        queen = Card(Suit.HEARTS, Rank.QUEEN)

        assert can_move_to_tableau(king, [], 0)
        assert not can_move_to_tableau(queen, [], 0)

    def test_alternating_colors_tableau(self):
        """Test that tableau requires alternating colors."""
        column = [Card(Suit.HEARTS, Rank.KING)]  # Red King

        black_queen = Card(Suit.SPADES, Rank.QUEEN)
        red_queen = Card(Suit.DIAMONDS, Rank.QUEEN)

        assert can_move_to_tableau(black_queen, column, 0)
        assert not can_move_to_tableau(red_queen, column, 0)

    def test_descending_rank_tableau(self):
        """Test that tableau requires descending ranks."""
        column = [Card(Suit.HEARTS, Rank.KING)]

        queen = Card(Suit.SPADES, Rank.QUEEN)
        jack = Card(Suit.SPADES, Rank.JACK)

        assert can_move_to_tableau(queen, column, 0)
        assert not can_move_to_tableau(jack, column, 0)

    def test_hidden_cards_not_available(self):
        """Test that hidden cards are not considered for stacking."""
        # Column has hidden red King and visible black Queen
        column = [
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.SPADES, Rank.QUEEN)
        ]
        hidden_count = 1

        # Red Jack should go on black Queen, not red King
        red_jack = Card(Suit.HEARTS, Rank.JACK)
        assert can_move_to_tableau(red_jack, column, hidden_count)

        # Black Jack should not go on black Queen
        black_jack = Card(Suit.SPADES, Rank.JACK)
        assert not can_move_to_tableau(black_jack, column, hidden_count)


class TestSequenceValidation:
    """Tests for sequence validation."""

    def test_single_card_sequence(self):
        """Test that single card is valid sequence."""
        cards = [Card(Suit.HEARTS, Rank.KING)]
        assert is_valid_sequence(cards)

    def test_valid_alternating_sequence(self):
        """Test valid alternating color descending sequence."""
        cards = [
            Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.DIAMONDS, Rank.THREE)
        ]
        assert is_valid_sequence(cards)

    def test_invalid_same_color_sequence(self):
        """Test that same color sequence is invalid."""
        cards = [
            Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.DIAMONDS, Rank.FOUR)  # Both red
        ]
        assert not is_valid_sequence(cards)

    def test_invalid_wrong_rank_sequence(self):
        """Test that wrong rank order is invalid."""
        cards = [
            Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.SPADES, Rank.TWO)  # Skip rank 4 and 3
        ]
        assert not is_valid_sequence(cards)


class TestMoveGeneration:
    """Tests for move generation."""

    def test_draw_move_available_with_stock(self):
        """Test that draw move is available when stock has cards."""
        state = GameState()
        state.stock = [Card(Suit.HEARTS, Rank.ACE)]

        moves = get_valid_moves(state)
        draw_moves = [m for m in moves if m.move_type == MoveType.DRAW]

        assert len(draw_moves) == 1

    def test_recycle_move_when_stock_empty(self):
        """Test that recycle move is available when stock is empty."""
        state = GameState()
        state.stock = []
        state.waste = [Card(Suit.HEARTS, Rank.ACE)]

        moves = get_valid_moves(state)
        recycle_moves = [m for m in moves if m.move_type == MoveType.RECYCLE]

        assert len(recycle_moves) == 1

    def test_waste_to_foundation_generated(self):
        """Test that waste to foundation moves are generated."""
        state = GameState()
        state.waste = [Card(Suit.HEARTS, Rank.ACE)]

        moves = get_valid_moves(state)
        waste_to_found = [m for m in moves
                          if m.move_type == MoveType.WASTE_TO_FOUNDATION]

        assert len(waste_to_found) == 1

    def test_tableau_to_foundation_generated(self):
        """Test that tableau to foundation moves are generated."""
        state = GameState()
        state.tableau[0] = [Card(Suit.HEARTS, Rank.ACE)]
        state.tableau_hidden[0] = 0

        moves = get_valid_moves(state)
        tab_to_found = [m for m in moves
                        if m.move_type == MoveType.TABLEAU_TO_FOUNDATION]

        assert len(tab_to_found) == 1

    def test_waste_to_tableau_generated(self):
        """Test that waste to tableau moves are generated."""
        state = GameState()
        state.waste = [Card(Suit.HEARTS, Rank.KING)]

        moves = get_valid_moves(state)
        waste_to_tab = [m for m in moves
                        if m.move_type == MoveType.WASTE_TO_TABLEAU]

        # King can go on any empty tableau column (7 options)
        assert len(waste_to_tab) == 7

    def test_tableau_to_tableau_generated(self):
        """Test that tableau to tableau moves are generated."""
        state = GameState()
        state.tableau[0] = [Card(Suit.HEARTS, Rank.KING)]
        state.tableau[1] = [Card(Suit.SPADES, Rank.QUEEN)]
        state.tableau_hidden[0] = 0
        state.tableau_hidden[1] = 0

        moves = get_valid_moves(state)
        tab_to_tab = [m for m in moves
                      if m.move_type == MoveType.TABLEAU_TO_TABLEAU]

        # Queen can go on King
        assert any(m.source == 1 and m.destination == 0 for m in tab_to_tab)


class TestApplyMove:
    """Tests for applying moves."""

    def test_apply_draw_move(self):
        """Test applying a draw move."""
        state = GameState()
        state.stock = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.DIAMONDS, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR)
        ]

        move = create_draw_move()
        new_state = apply_move(state, move)

        assert len(new_state.stock) == 1
        assert len(new_state.waste) == 3
        assert new_state.move_count == 1

    def test_apply_recycle_move(self):
        """Test applying a recycle move."""
        state = GameState()
        state.stock = []
        state.waste = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.DIAMONDS, Rank.THREE)
        ]

        move = create_recycle_move()
        new_state = apply_move(state, move)

        assert len(new_state.stock) == 3
        assert len(new_state.waste) == 0
        # Stock should be reversed order of waste
        assert new_state.stock[0] == Card(Suit.DIAMONDS, Rank.THREE)

    def test_apply_waste_to_foundation(self):
        """Test applying waste to foundation move."""
        state = GameState()
        ace = Card(Suit.HEARTS, Rank.ACE)
        state.waste = [ace]
        initial_score = state.score

        move = create_waste_to_foundation_move(ace)
        new_state = apply_move(state, move)

        assert len(new_state.waste) == 0
        assert len(new_state.foundations[Suit.HEARTS]) == 1
        assert new_state.score == initial_score + 5

    def test_apply_waste_to_tableau(self):
        """Test applying waste to tableau move."""
        state = GameState()
        king = Card(Suit.HEARTS, Rank.KING)
        state.waste = [king]

        move = create_waste_to_tableau_move(king, 0)
        new_state = apply_move(state, move)

        assert len(new_state.waste) == 0
        assert len(new_state.tableau[0]) == 1
        assert new_state.tableau[0][0] == king

    def test_apply_tableau_to_foundation(self):
        """Test applying tableau to foundation move."""
        state = GameState()
        ace = Card(Suit.HEARTS, Rank.ACE)
        state.tableau[0] = [ace]
        state.tableau_hidden[0] = 0
        initial_score = state.score

        move = create_tableau_to_foundation_move(0, ace)
        new_state = apply_move(state, move)

        assert len(new_state.tableau[0]) == 0
        assert len(new_state.foundations[Suit.HEARTS]) == 1
        assert new_state.score == initial_score + 5

    def test_apply_tableau_to_foundation_reveals_card(self):
        """Test that moving from tableau reveals hidden card."""
        state = GameState()
        state.tableau[0] = [
            Card(Suit.HEARTS, Rank.KING),  # Hidden
            Card(Suit.SPADES, Rank.ACE)    # Visible
        ]
        state.tableau_hidden[0] = 1

        move = create_tableau_to_foundation_move(0, Card(Suit.SPADES, Rank.ACE))
        new_state = apply_move(state, move)

        # Hidden count should decrease
        assert new_state.tableau_hidden[0] == 0
        assert len(new_state.tableau[0]) == 1

    def test_apply_tableau_to_tableau_single(self):
        """Test applying single card tableau to tableau move."""
        state = GameState()
        state.tableau[0] = [Card(Suit.HEARTS, Rank.KING)]
        state.tableau[1] = [Card(Suit.SPADES, Rank.QUEEN)]
        state.tableau_hidden[0] = 0
        state.tableau_hidden[1] = 0

        move = create_tableau_to_tableau_move(1, 0, Card(Suit.SPADES, Rank.QUEEN), 1)
        new_state = apply_move(state, move)

        assert len(new_state.tableau[1]) == 0
        assert len(new_state.tableau[0]) == 2
        assert new_state.tableau[0][1].rank == Rank.QUEEN

    def test_apply_tableau_to_tableau_sequence(self):
        """Test applying multi-card sequence move."""
        state = GameState()
        state.tableau[0] = [Card(Suit.HEARTS, Rank.KING)]
        state.tableau[1] = [
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK),
            Card(Suit.SPADES, Rank.TEN)
        ]
        state.tableau_hidden[0] = 0
        state.tableau_hidden[1] = 0

        # Move 3-card sequence
        move = create_tableau_to_tableau_move(1, 0, Card(Suit.SPADES, Rank.QUEEN), 3)
        new_state = apply_move(state, move)

        assert len(new_state.tableau[1]) == 0
        assert len(new_state.tableau[0]) == 4
        assert new_state.tableau[0][-1].rank == Rank.TEN

    def test_apply_move_creates_independent_state(self):
        """Test that applying a move doesn't modify original state."""
        state = GameState()
        state.stock = [Card(Suit.HEARTS, Rank.ACE)]

        move = create_draw_move()
        new_state = apply_move(state, move)

        # Original state should be unchanged
        assert len(state.stock) == 1
        assert len(state.waste) == 0
        assert state.move_count == 0


class TestMoveValidation:
    """Tests for is_valid_move function."""

    def test_valid_draw_move(self):
        """Test validating a draw move."""
        state = GameState()
        state.stock = [Card(Suit.HEARTS, Rank.ACE)]

        move = create_draw_move()
        assert is_valid_move(state, move)

    def test_invalid_draw_move_empty_stock(self):
        """Test that draw is invalid when stock is empty."""
        state = GameState()
        state.stock = []

        move = create_draw_move()
        assert not is_valid_move(state, move)

    def test_valid_recycle_move(self):
        """Test validating a recycle move."""
        state = GameState()
        state.stock = []
        state.waste = [Card(Suit.HEARTS, Rank.ACE)]

        move = create_recycle_move()
        assert is_valid_move(state, move)

    def test_invalid_recycle_with_stock(self):
        """Test that recycle is invalid when stock has cards."""
        state = GameState()
        state.stock = [Card(Suit.HEARTS, Rank.ACE)]
        state.waste = [Card(Suit.SPADES, Rank.TWO)]

        move = create_recycle_move()
        assert not is_valid_move(state, move)

    def test_valid_foundation_move(self):
        """Test validating foundation move."""
        state = GameState()
        ace = Card(Suit.HEARTS, Rank.ACE)
        state.waste = [ace]

        move = create_waste_to_foundation_move(ace)
        assert is_valid_move(state, move)

    def test_invalid_foundation_move_wrong_rank(self):
        """Test that wrong rank to foundation is invalid."""
        state = GameState()
        two = Card(Suit.HEARTS, Rank.TWO)
        state.waste = [two]

        move = create_waste_to_foundation_move(two)
        assert not is_valid_move(state, move)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

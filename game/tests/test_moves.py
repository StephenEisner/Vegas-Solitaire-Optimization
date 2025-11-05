"""
Tests for Move representation.

These tests ensure that moves can be created, displayed, and
queried correctly.
"""

import pytest
from game.core.moves import (
    Move, MoveType,
    create_draw_move, create_recycle_move,
    create_waste_to_foundation_move, create_waste_to_tableau_move,
    create_tableau_to_foundation_move, create_tableau_to_tableau_move
)
from game.core.card import Card, Suit, Rank


class TestMoveType:
    """Tests for MoveType enum."""

    def test_move_type_string(self):
        """Test that move types have readable string representations."""
        assert "Tableau To Foundation" in str(MoveType.TABLEAU_TO_FOUNDATION)
        assert "Draw" in str(MoveType.DRAW)


class TestMoveCreation:
    """Tests for basic move creation."""

    def test_create_simple_move(self):
        """Test creating a simple move."""
        move = Move(MoveType.DRAW)
        assert move.move_type == MoveType.DRAW
        assert move.source is None
        assert move.destination is None
        assert move.card_count == 1
        assert move.card is None

    def test_create_move_with_card(self):
        """Test creating a move with a card."""
        card = Card(Suit.HEARTS, Rank.ACE)
        move = Move(MoveType.WASTE_TO_FOUNDATION, card=card)

        assert move.move_type == MoveType.WASTE_TO_FOUNDATION
        assert move.card == card

    def test_create_move_with_positions(self):
        """Test creating a move with source and destination."""
        card = Card(Suit.HEARTS, Rank.KING)
        move = Move(
            MoveType.TABLEAU_TO_TABLEAU,
            source=2,
            destination=5,
            card=card
        )

        assert move.source == 2
        assert move.destination == 5

    def test_create_move_with_sequence(self):
        """Test creating a move with multiple cards."""
        card = Card(Suit.HEARTS, Rank.FIVE)
        move = Move(
            MoveType.TABLEAU_TO_TABLEAU,
            source=2,
            destination=5,
            card_count=3,
            card=card
        )

        assert move.card_count == 3


class TestMoveImmutability:
    """Tests for move immutability."""

    def test_moves_are_immutable(self):
        """Test that moves are frozen and cannot be modified."""
        move = Move(MoveType.DRAW)

        with pytest.raises(AttributeError):
            move.move_type = MoveType.RECYCLE  # type: ignore

    def test_moves_are_hashable(self):
        """Test that moves can be used in sets and as dict keys."""
        move1 = Move(MoveType.DRAW)
        move2 = Move(MoveType.DRAW)
        move3 = Move(MoveType.RECYCLE)

        move_set = {move1, move2, move3}
        assert len(move_set) == 2  # move1 and move2 are the same


class TestMoveEquality:
    """Tests for move equality."""

    def test_identical_moves_are_equal(self):
        """Test that identical moves are equal."""
        card = Card(Suit.HEARTS, Rank.ACE)

        move1 = Move(MoveType.WASTE_TO_FOUNDATION, card=card)
        move2 = Move(MoveType.WASTE_TO_FOUNDATION, card=card)

        assert move1 == move2

    def test_different_moves_not_equal(self):
        """Test that different moves are not equal."""
        card1 = Card(Suit.HEARTS, Rank.ACE)
        card2 = Card(Suit.SPADES, Rank.ACE)

        move1 = Move(MoveType.WASTE_TO_FOUNDATION, card=card1)
        move2 = Move(MoveType.WASTE_TO_FOUNDATION, card=card2)

        assert move1 != move2


class TestMoveStringRepresentation:
    """Tests for move string representations."""

    def test_draw_move_string(self):
        """Test string representation of draw move."""
        move = Move(MoveType.DRAW)
        assert "Draw" in str(move)
        assert "stock" in str(move)

    def test_recycle_move_string(self):
        """Test string representation of recycle move."""
        move = Move(MoveType.RECYCLE)
        assert "Recycle" in str(move)

    def test_waste_to_foundation_string(self):
        """Test string representation of waste to foundation."""
        card = Card(Suit.HEARTS, Rank.ACE)
        move = Move(MoveType.WASTE_TO_FOUNDATION, card=card)

        move_str = str(move)
        assert "waste" in move_str
        assert "foundation" in move_str
        assert "A♥" in move_str

    def test_waste_to_tableau_string(self):
        """Test string representation of waste to tableau."""
        card = Card(Suit.HEARTS, Rank.KING)
        move = Move(MoveType.WASTE_TO_TABLEAU, destination=3, card=card)

        move_str = str(move)
        assert "waste" in move_str
        assert "3" in move_str
        assert "K♥" in move_str

    def test_tableau_to_foundation_string(self):
        """Test string representation of tableau to foundation."""
        card = Card(Suit.HEARTS, Rank.ACE)
        move = Move(MoveType.TABLEAU_TO_FOUNDATION, source=2, card=card)

        move_str = str(move)
        assert "column 2" in move_str or "2" in move_str
        assert "foundation" in move_str
        assert "A♥" in move_str

    def test_tableau_to_tableau_single_card_string(self):
        """Test string representation of single card tableau move."""
        card = Card(Suit.HEARTS, Rank.QUEEN)
        move = Move(
            MoveType.TABLEAU_TO_TABLEAU,
            source=1,
            destination=4,
            card=card
        )

        move_str = str(move)
        assert "1" in move_str
        assert "4" in move_str
        assert "Q♥" in move_str

    def test_tableau_to_tableau_sequence_string(self):
        """Test string representation of multi-card tableau move."""
        card = Card(Suit.HEARTS, Rank.FIVE)
        move = Move(
            MoveType.TABLEAU_TO_TABLEAU,
            source=1,
            destination=4,
            card_count=3,
            card=card
        )

        move_str = str(move)
        assert "1" in move_str
        assert "4" in move_str
        assert "3" in move_str  # card count
        assert "5♥" in move_str

    def test_move_repr(self):
        """Test detailed repr for debugging."""
        card = Card(Suit.HEARTS, Rank.ACE)
        move = Move(
            MoveType.TABLEAU_TO_FOUNDATION,
            source=2,
            card=card
        )

        repr_str = repr(move)
        assert "Move" in repr_str
        assert "TABLEAU_TO_FOUNDATION" in repr_str
        assert "2" in repr_str


class TestMoveHelperFunctions:
    """Tests for move helper/factory functions."""

    def test_create_draw_move(self):
        """Test draw move factory function."""
        move = create_draw_move()
        assert move.move_type == MoveType.DRAW

    def test_create_recycle_move(self):
        """Test recycle move factory function."""
        move = create_recycle_move()
        assert move.move_type == MoveType.RECYCLE

    def test_create_waste_to_foundation_move(self):
        """Test waste to foundation factory function."""
        card = Card(Suit.HEARTS, Rank.ACE)
        move = create_waste_to_foundation_move(card)

        assert move.move_type == MoveType.WASTE_TO_FOUNDATION
        assert move.card == card

    def test_create_waste_to_tableau_move(self):
        """Test waste to tableau factory function."""
        card = Card(Suit.HEARTS, Rank.KING)
        move = create_waste_to_tableau_move(card, 3)

        assert move.move_type == MoveType.WASTE_TO_TABLEAU
        assert move.destination == 3
        assert move.card == card

    def test_create_tableau_to_foundation_move(self):
        """Test tableau to foundation factory function."""
        card = Card(Suit.HEARTS, Rank.ACE)
        move = create_tableau_to_foundation_move(2, card)

        assert move.move_type == MoveType.TABLEAU_TO_FOUNDATION
        assert move.source == 2
        assert move.card == card

    def test_create_tableau_to_tableau_move(self):
        """Test tableau to tableau factory function."""
        card = Card(Suit.HEARTS, Rank.QUEEN)
        move = create_tableau_to_tableau_move(1, 4, card, 2)

        assert move.move_type == MoveType.TABLEAU_TO_TABLEAU
        assert move.source == 1
        assert move.destination == 4
        assert move.card_count == 2
        assert move.card == card


class TestMoveQueries:
    """Tests for move query methods."""

    def test_is_draw_move(self):
        """Test is_draw_move query."""
        assert create_draw_move().is_draw_move()
        assert create_recycle_move().is_draw_move()

        card = Card(Suit.HEARTS, Rank.ACE)
        assert not create_waste_to_foundation_move(card).is_draw_move()

    def test_is_foundation_move(self):
        """Test is_foundation_move query."""
        card = Card(Suit.HEARTS, Rank.ACE)

        assert create_waste_to_foundation_move(card).is_foundation_move()
        assert create_tableau_to_foundation_move(0, card).is_foundation_move()

        assert not create_draw_move().is_foundation_move()
        assert not create_waste_to_tableau_move(card, 0).is_foundation_move()

    def test_is_tableau_move(self):
        """Test is_tableau_move query."""
        card = Card(Suit.HEARTS, Rank.KING)

        assert create_tableau_to_foundation_move(0, card).is_tableau_move()
        assert create_tableau_to_tableau_move(0, 1, card).is_tableau_move()
        assert create_waste_to_tableau_move(card, 0).is_tableau_move()

        assert not create_draw_move().is_tableau_move()
        assert not create_recycle_move().is_tableau_move()

    def test_reveals_card(self):
        """Test reveals_card hint."""
        card = Card(Suit.HEARTS, Rank.ACE)

        # Tableau moves might reveal cards
        assert create_tableau_to_foundation_move(0, card).reveals_card()
        assert create_tableau_to_tableau_move(0, 1, card).reveals_card()

        # Other moves don't reveal cards
        assert not create_draw_move().reveals_card()
        assert not create_waste_to_foundation_move(card).reveals_card()

    def test_empties_column(self):
        """Test empties_column hint."""
        card = Card(Suit.HEARTS, Rank.ACE)

        # Tableau moves might empty columns
        assert create_tableau_to_foundation_move(0, card).empties_column()
        assert create_tableau_to_tableau_move(0, 1, card).empties_column()

        # Other moves don't empty columns
        assert not create_draw_move().empties_column()
        assert not create_waste_to_tableau_move(card, 0).empties_column()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

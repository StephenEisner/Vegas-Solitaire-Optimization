"""
Tests for GameState class.

These tests ensure that game state operations work correctly including
copying, hashing, and state queries.
"""

import pytest
from game.core.state import GameState
from game.core.card import Card, Suit, Rank


class TestGameStateInitialization:
    """Tests for GameState initialization."""

    def test_default_state(self):
        """Test that default state is properly initialized."""
        state = GameState()

        assert state.stock == []
        assert state.waste == []
        assert len(state.tableau) == 7
        assert all(col == [] for col in state.tableau)
        assert state.tableau_hidden == [0] * 7
        assert len(state.foundations) == 4
        assert all(cards == [] for cards in state.foundations.values())
        assert state.score == -52  # Vegas entry cost
        assert state.move_count == 0

    def test_state_with_cards(self):
        """Test creating state with cards."""
        state = GameState()
        state.stock = [Card(Suit.HEARTS, Rank.ACE)]
        state.waste = [Card(Suit.SPADES, Rank.TWO)]
        state.tableau[0] = [Card(Suit.DIAMONDS, Rank.KING)]

        assert len(state.stock) == 1
        assert len(state.waste) == 1
        assert len(state.tableau[0]) == 1


class TestGameStateCopy:
    """Tests for state copying."""

    def test_copy_is_independent(self):
        """Test that copied states are independent."""
        state1 = GameState()
        state1.waste.append(Card(Suit.HEARTS, Rank.ACE))
        state1.score = 10
        state1.move_count = 5

        state2 = state1.copy()

        # Modify state2
        state2.waste.append(Card(Suit.SPADES, Rank.TWO))
        state2.score = 20
        state2.move_count = 10

        # state1 should be unchanged
        assert len(state1.waste) == 1
        assert state1.score == 10
        assert state1.move_count == 5

        # state2 should have changes
        assert len(state2.waste) == 2
        assert state2.score == 20
        assert state2.move_count == 10

    def test_copy_tableau_independence(self):
        """Test that tableau columns are independently copied."""
        state1 = GameState()
        state1.tableau[0] = [Card(Suit.HEARTS, Rank.ACE)]
        state1.tableau_hidden[0] = 1

        state2 = state1.copy()

        # Modify state2's tableau
        state2.tableau[0].append(Card(Suit.SPADES, Rank.TWO))
        state2.tableau_hidden[0] = 2

        # state1 should be unchanged
        assert len(state1.tableau[0]) == 1
        assert state1.tableau_hidden[0] == 1

    def test_copy_foundations_independence(self):
        """Test that foundations are independently copied."""
        state1 = GameState()
        state1.foundations[Suit.HEARTS] = [Card(Suit.HEARTS, Rank.ACE)]

        state2 = state1.copy()

        # Modify state2's foundations
        state2.foundations[Suit.HEARTS].append(Card(Suit.HEARTS, Rank.TWO))

        # state1 should be unchanged
        assert len(state1.foundations[Suit.HEARTS]) == 1
        assert len(state2.foundations[Suit.HEARTS]) == 2


class TestGameStateHashing:
    """Tests for state hashing and equality."""

    def test_identical_states_are_equal(self):
        """Test that identical states are considered equal."""
        state1 = GameState()
        state1.waste = [Card(Suit.HEARTS, Rank.ACE)]
        state1.tableau[0] = [Card(Suit.SPADES, Rank.KING)]

        state2 = GameState()
        state2.waste = [Card(Suit.HEARTS, Rank.ACE)]
        state2.tableau[0] = [Card(Suit.SPADES, Rank.KING)]

        assert state1 == state2
        assert hash(state1) == hash(state2)

    def test_different_states_not_equal(self):
        """Test that different states are not equal."""
        state1 = GameState()
        state1.waste = [Card(Suit.HEARTS, Rank.ACE)]

        state2 = GameState()
        state2.waste = [Card(Suit.SPADES, Rank.ACE)]

        assert state1 != state2

    def test_score_not_in_equality(self):
        """Test that score doesn't affect equality."""
        state1 = GameState()
        state1.score = 10

        state2 = GameState()
        state2.score = 20

        assert state1 == state2

    def test_move_count_not_in_equality(self):
        """Test that move count doesn't affect equality."""
        state1 = GameState()
        state1.move_count = 5

        state2 = GameState()
        state2.move_count = 10

        assert state1 == state2

    def test_states_can_be_used_in_set(self):
        """Test that states can be added to sets (hashable)."""
        state1 = GameState()
        state1.waste = [Card(Suit.HEARTS, Rank.ACE)]

        state2 = GameState()
        state2.waste = [Card(Suit.HEARTS, Rank.ACE)]

        state3 = GameState()
        state3.waste = [Card(Suit.SPADES, Rank.ACE)]

        state_set = {state1, state2, state3}

        # state1 and state2 are the same, so set should have 2 elements
        assert len(state_set) == 2

    def test_states_can_be_dict_keys(self):
        """Test that states can be used as dictionary keys."""
        state1 = GameState()
        state1.waste = [Card(Suit.HEARTS, Rank.ACE)]

        state2 = GameState()
        state2.waste = [Card(Suit.HEARTS, Rank.ACE)]

        state_dict = {state1: "value1"}
        state_dict[state2] = "value2"

        # state1 and state2 are the same, so only one key exists
        assert len(state_dict) == 1
        assert state_dict[state1] == "value2"


class TestWinningCondition:
    """Tests for winning state detection."""

    def test_empty_state_not_winning(self):
        """Test that an empty state is not winning."""
        state = GameState()
        assert not state.is_winning()

    def test_full_foundations_is_winning(self):
        """Test that a state with all cards in foundations is winning."""
        state = GameState()

        # Fill all foundations
        for suit in Suit:
            state.foundations[suit] = [
                Card(suit, rank) for rank in Rank
            ]

        assert state.is_winning()
        assert state.get_foundation_count() == 52

    def test_partial_foundations_not_winning(self):
        """Test that partial foundations are not winning."""
        state = GameState()

        # Fill only one foundation
        state.foundations[Suit.HEARTS] = [
            Card(Suit.HEARTS, rank) for rank in Rank
        ]

        assert not state.is_winning()
        assert state.get_foundation_count() == 13


class TestFoundationCount:
    """Tests for foundation card counting."""

    def test_empty_foundations(self):
        """Test counting with empty foundations."""
        state = GameState()
        assert state.get_foundation_count() == 0

    def test_some_foundation_cards(self):
        """Test counting with some cards in foundations."""
        state = GameState()
        state.foundations[Suit.HEARTS] = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO)
        ]
        state.foundations[Suit.SPADES] = [
            Card(Suit.SPADES, Rank.ACE)
        ]

        assert state.get_foundation_count() == 3


class TestTableauQueries:
    """Tests for tableau query methods."""

    def test_get_visible_cards(self):
        """Test getting visible cards from tableau."""
        state = GameState()
        state.tableau[0] = [
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.DIAMONDS, Rank.JACK)
        ]
        state.tableau_hidden[0] = 1  # First card is hidden

        visible = state.get_tableau_visible_cards(0)

        assert len(visible) == 2
        assert visible[0].rank == Rank.QUEEN
        assert visible[1].rank == Rank.JACK

    def test_get_visible_cards_all_visible(self):
        """Test getting visible cards when all are visible."""
        state = GameState()
        state.tableau[0] = [
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.SPADES, Rank.QUEEN)
        ]
        state.tableau_hidden[0] = 0  # All visible

        visible = state.get_tableau_visible_cards(0)

        assert len(visible) == 2

    def test_get_visible_cards_all_hidden(self):
        """Test getting visible cards when all are hidden."""
        state = GameState()
        state.tableau[0] = [
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.SPADES, Rank.QUEEN)
        ]
        state.tableau_hidden[0] = 2  # All hidden

        visible = state.get_tableau_visible_cards(0)

        assert len(visible) == 0

    def test_get_visible_cards_invalid_column(self):
        """Test that invalid column raises error."""
        state = GameState()

        with pytest.raises(ValueError):
            state.get_tableau_visible_cards(-1)

        with pytest.raises(ValueError):
            state.get_tableau_visible_cards(7)

    def test_is_tableau_column_empty(self):
        """Test checking if tableau column is empty."""
        state = GameState()

        assert state.is_tableau_column_empty(0)

        state.tableau[0] = [Card(Suit.HEARTS, Rank.ACE)]

        assert not state.is_tableau_column_empty(0)

    def test_get_top_tableau_card(self):
        """Test getting top card from tableau."""
        state = GameState()
        state.tableau[0] = [
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.SPADES, Rank.QUEEN)
        ]
        state.tableau_hidden[0] = 1

        top = state.get_top_tableau_card(0)

        assert top is not None
        assert top.rank == Rank.QUEEN

    def test_get_top_tableau_card_empty(self):
        """Test getting top card from empty tableau."""
        state = GameState()
        top = state.get_top_tableau_card(0)

        assert top is None


class TestWasteQueries:
    """Tests for waste pile query methods."""

    def test_get_top_waste_card(self):
        """Test getting top card from waste."""
        state = GameState()
        state.waste = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.DIAMONDS, Rank.THREE)
        ]

        top = state.get_top_waste_card()

        assert top is not None
        assert top.rank == Rank.THREE

    def test_get_top_waste_card_empty(self):
        """Test getting top card from empty waste."""
        state = GameState()
        top = state.get_top_waste_card()

        assert top is None


class TestFoundationQueries:
    """Tests for foundation query methods."""

    def test_get_foundation_top(self):
        """Test getting top card from foundation."""
        state = GameState()
        state.foundations[Suit.HEARTS] = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO)
        ]

        top = state.get_foundation_top(Suit.HEARTS)

        assert top is not None
        assert top.rank == Rank.TWO

    def test_get_foundation_top_empty(self):
        """Test getting top card from empty foundation."""
        state = GameState()
        top = state.get_foundation_top(Suit.HEARTS)

        assert top is None


class TestStateRepresentation:
    """Tests for state string representations."""

    def test_str_representation(self):
        """Test that __str__ produces readable output."""
        state = GameState()
        state.score = 10
        state.move_count = 5

        output = str(state)

        assert "Score: $10" in output
        assert "Moves: 5" in output
        assert "Stock:" in output
        assert "Waste:" in output
        assert "Foundations:" in output
        assert "Tableau:" in output

    def test_repr_representation(self):
        """Test that __repr__ shows key information."""
        state = GameState()
        state.score = 10
        state.move_count = 5

        output = repr(state)

        assert "GameState" in output
        assert "10" in output  # score
        assert "5" in output   # moves


class TestMLVector:
    """Tests for ML vector representation."""

    def test_to_vector_not_yet_implemented(self):
        """Test that to_vector is stubbed for Phase 4."""
        state = GameState()

        with pytest.raises(NotImplementedError):
            state.to_vector()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

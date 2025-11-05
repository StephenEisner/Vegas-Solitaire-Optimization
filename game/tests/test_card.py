"""
Tests for Card and Deck classes.

These tests ensure that card operations, deck shuffling, and card stacking
logic work correctly.
"""

import pytest
from game.core.card import Card, Deck, Suit, Rank


class TestSuit:
    """Tests for Suit enum."""

    def test_suit_string_representation(self):
        """Test that suits display with correct symbols."""
        assert str(Suit.SPADES) == "♠"
        assert str(Suit.HEARTS) == "♥"
        assert str(Suit.DIAMONDS) == "♦"
        assert str(Suit.CLUBS) == "♣"

    def test_suit_colors(self):
        """Test that suits have correct colors."""
        assert Suit.SPADES.color == "black"
        assert Suit.CLUBS.color == "black"
        assert Suit.HEARTS.color == "red"
        assert Suit.DIAMONDS.color == "red"


class TestRank:
    """Tests for Rank enum."""

    def test_rank_string_representation(self):
        """Test that ranks display correctly."""
        assert str(Rank.ACE) == "A"
        assert str(Rank.TWO) == "2"
        assert str(Rank.JACK) == "J"
        assert str(Rank.QUEEN) == "Q"
        assert str(Rank.KING) == "K"

    def test_rank_numeric_values(self):
        """Test that ranks have correct numeric values."""
        assert Rank.ACE.numeric_value == 1
        assert Rank.TWO.numeric_value == 2
        assert Rank.TEN.numeric_value == 10
        assert Rank.JACK.numeric_value == 11
        assert Rank.QUEEN.numeric_value == 12
        assert Rank.KING.numeric_value == 13


class TestCard:
    """Tests for Card class."""

    def test_card_creation(self):
        """Test that cards can be created."""
        card = Card(Suit.HEARTS, Rank.ACE)
        assert card.suit == Suit.HEARTS
        assert card.rank == Rank.ACE

    def test_card_color(self):
        """Test that cards return correct color."""
        assert Card(Suit.HEARTS, Rank.ACE).color == "red"
        assert Card(Suit.DIAMONDS, Rank.KING).color == "red"
        assert Card(Suit.SPADES, Rank.ACE).color == "black"
        assert Card(Suit.CLUBS, Rank.QUEEN).color == "black"

    def test_card_value(self):
        """Test that cards return correct numeric value."""
        assert Card(Suit.HEARTS, Rank.ACE).value == 1
        assert Card(Suit.SPADES, Rank.FIVE).value == 5
        assert Card(Suit.DIAMONDS, Rank.KING).value == 13

    def test_card_string_representation(self):
        """Test that cards display correctly."""
        assert str(Card(Suit.HEARTS, Rank.ACE)) == "A♥"
        assert str(Card(Suit.SPADES, Rank.KING)) == "K♠"
        assert str(Card(Suit.DIAMONDS, Rank.TEN)) == "10♦"

    def test_card_repr(self):
        """Test that card repr is detailed."""
        card = Card(Suit.HEARTS, Rank.ACE)
        assert "HEARTS" in repr(card)
        assert "ACE" in repr(card)

    def test_card_immutability(self):
        """Test that cards are immutable (frozen dataclass)."""
        card = Card(Suit.HEARTS, Rank.ACE)
        with pytest.raises(AttributeError):
            card.suit = Suit.SPADES  # type: ignore

    def test_card_equality(self):
        """Test that identical cards are equal."""
        card1 = Card(Suit.HEARTS, Rank.ACE)
        card2 = Card(Suit.HEARTS, Rank.ACE)
        card3 = Card(Suit.SPADES, Rank.ACE)

        assert card1 == card2
        assert card1 != card3

    def test_card_hashable(self):
        """Test that cards can be used in sets and as dict keys."""
        card1 = Card(Suit.HEARTS, Rank.ACE)
        card2 = Card(Suit.HEARTS, Rank.ACE)
        card3 = Card(Suit.SPADES, Rank.ACE)

        card_set = {card1, card2, card3}
        assert len(card_set) == 2  # card1 and card2 are the same


class TestCardStacking:
    """Tests for tableau stacking rules."""

    def test_kings_only_to_empty_tableau(self):
        """Test that only Kings can go on empty tableau piles."""
        king = Card(Suit.HEARTS, Rank.KING)
        queen = Card(Suit.HEARTS, Rank.QUEEN)
        ace = Card(Suit.SPADES, Rank.ACE)

        assert king.can_stack_on(None)
        assert not queen.can_stack_on(None)
        assert not ace.can_stack_on(None)

    def test_alternating_colors(self):
        """Test that cards must alternate colors in tableau."""
        red_king = Card(Suit.HEARTS, Rank.KING)
        black_queen = Card(Suit.SPADES, Rank.QUEEN)
        red_queen = Card(Suit.DIAMONDS, Rank.QUEEN)

        # Black queen on red king should work
        assert black_queen.can_stack_on(red_king)

        # Red queen on red king should not work
        assert not red_queen.can_stack_on(red_king)

    def test_descending_rank(self):
        """Test that cards must be one rank lower."""
        king = Card(Suit.HEARTS, Rank.KING)
        queen = Card(Suit.SPADES, Rank.QUEEN)
        jack = Card(Suit.HEARTS, Rank.JACK)
        ten = Card(Suit.SPADES, Rank.TEN)

        # Queen on King works
        assert queen.can_stack_on(king)

        # Jack on King doesn't work (not one lower)
        assert not jack.can_stack_on(king)

        # Jack on Queen works
        assert jack.can_stack_on(queen)

        # Ten on Queen doesn't work
        assert not ten.can_stack_on(queen)

    def test_ace_can_stack_on_two(self):
        """Test that Aces can be placed on Twos (descending sequence)."""
        ace = Card(Suit.HEARTS, Rank.ACE)
        two = Card(Suit.SPADES, Rank.TWO)

        # Ace (value 1) can go on Two (value 2) in descending order
        assert ace.can_stack_on(two)

    def test_nothing_can_stack_on_ace(self):
        """Test that no card can be placed on an Ace in tableau."""
        ace = Card(Suit.HEARTS, Rank.ACE)
        two = Card(Suit.SPADES, Rank.TWO)
        king = Card(Suit.SPADES, Rank.KING)

        # Nothing can go on Ace (it's the lowest)
        assert not two.can_stack_on(ace)
        assert not king.can_stack_on(ace)

    def test_complete_valid_sequence(self):
        """Test a complete valid stacking sequence."""
        cards = [
            Card(Suit.HEARTS, Rank.KING),    # Red King (base)
            Card(Suit.SPADES, Rank.QUEEN),   # Black Queen
            Card(Suit.DIAMONDS, Rank.JACK),  # Red Jack
            Card(Suit.CLUBS, Rank.TEN),      # Black Ten
        ]

        # Each card should be able to stack on the previous one
        assert cards[1].can_stack_on(cards[0])
        assert cards[2].can_stack_on(cards[1])
        assert cards[3].can_stack_on(cards[2])


class TestDeck:
    """Tests for Deck class."""

    def test_deck_initialization(self):
        """Test that a new deck has 52 cards."""
        deck = Deck()
        assert len(deck) == 52
        assert len(deck.cards) == 52

    def test_deck_has_all_cards(self):
        """Test that deck contains all 52 unique cards."""
        deck = Deck()
        card_set = set(deck.cards)
        assert len(card_set) == 52

        # Check that all suits and ranks are present
        suits_present = {card.suit for card in deck.cards}
        ranks_present = {card.rank for card in deck.cards}

        assert len(suits_present) == 4
        assert len(ranks_present) == 13

    def test_deck_shuffle_is_deterministic(self):
        """Test that shuffling with same seed produces same order."""
        deck1 = Deck(seed=42)
        deck1.shuffle()

        deck2 = Deck(seed=42)
        deck2.shuffle()

        assert deck1.cards == deck2.cards

    def test_deck_shuffle_changes_order(self):
        """Test that shuffling actually changes card order."""
        deck1 = Deck(seed=42)
        original_order = deck1.cards.copy()

        deck1.shuffle()

        # Very unlikely to have same order after shuffle
        assert deck1.cards != original_order

    def test_deck_different_seeds_produce_different_shuffles(self):
        """Test that different seeds produce different orders."""
        deck1 = Deck(seed=42)
        deck1.shuffle()

        deck2 = Deck(seed=123)
        deck2.shuffle()

        assert deck1.cards != deck2.cards

    def test_draw_single_card(self):
        """Test drawing a single card."""
        deck = Deck(seed=42)
        initial_count = len(deck)
        top_card = deck.cards[0]

        drawn = deck.draw_one()

        assert drawn == top_card
        assert len(deck) == initial_count - 1

    def test_draw_multiple_cards(self):
        """Test drawing multiple cards at once."""
        deck = Deck(seed=42)
        initial_count = len(deck)
        top_three = deck.cards[:3]

        drawn = deck.draw(3)

        assert drawn == top_three
        assert len(deck) == initial_count - 3

    def test_draw_all_cards(self):
        """Test that we can draw all cards from deck."""
        deck = Deck(seed=42)
        all_cards = deck.draw(52)

        assert len(all_cards) == 52
        assert len(deck) == 0
        assert deck.is_empty()

    def test_draw_too_many_cards_raises_error(self):
        """Test that drawing more cards than available raises error."""
        deck = Deck(seed=42)
        deck.draw(50)  # Draw 50, leaving 2

        with pytest.raises(ValueError, match="Cannot draw"):
            deck.draw(3)  # Try to draw 3 when only 2 remain

    def test_draw_from_empty_deck_raises_error(self):
        """Test that drawing from empty deck raises error."""
        deck = Deck(seed=42)
        deck.draw(52)  # Draw all cards

        with pytest.raises(ValueError, match="empty"):
            deck.draw_one()

    def test_deck_is_empty(self):
        """Test the is_empty method."""
        deck = Deck(seed=42)
        assert not deck.is_empty()

        deck.draw(52)
        assert deck.is_empty()

    def test_deck_string_representation(self):
        """Test deck string representation."""
        deck = Deck(seed=42)
        assert "52" in str(deck)

        deck.draw(10)
        assert "42" in str(deck)

    def test_deck_repr(self):
        """Test deck repr for debugging."""
        deck = Deck(seed=42)
        assert "52" in repr(deck)
        assert "42" in repr(deck)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

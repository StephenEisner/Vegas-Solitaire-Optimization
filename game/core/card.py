"""
Card and Deck classes for Vegas Solitaire.

This module implements the basic card and deck structures used throughout the game.
Cards are immutable and decks support seeded shuffling for reproducibility.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import random


class Suit(Enum):
    """Card suits with their symbols."""
    SPADES = ("♠", "black")
    HEARTS = ("♥", "red")
    DIAMONDS = ("♦", "red")
    CLUBS = ("♣", "black")

    def __str__(self) -> str:
        return self.value[0]

    @property
    def color(self) -> str:
        """Returns the color of the suit (red or black)."""
        return self.value[1]


class Rank(Enum):
    """Card ranks with their numeric values."""
    ACE = (1, "A")
    TWO = (2, "2")
    THREE = (3, "3")
    FOUR = (4, "4")
    FIVE = (5, "5")
    SIX = (6, "6")
    SEVEN = (7, "7")
    EIGHT = (8, "8")
    NINE = (9, "9")
    TEN = (10, "10")
    JACK = (11, "J")
    QUEEN = (12, "Q")
    KING = (13, "K")

    def __str__(self) -> str:
        return self.value[1]

    @property
    def numeric_value(self) -> int:
        """Returns the numeric value of the rank (1-13)."""
        return self.value[0]


@dataclass(frozen=True)
class Card:
    """
    Immutable card representation.

    Attributes:
        suit: The card's suit (Spades, Hearts, Diamonds, or Clubs)
        rank: The card's rank (Ace through King)
    """
    suit: Suit
    rank: Rank

    @property
    def color(self) -> str:
        """Returns the color of the card (red or black)."""
        return self.suit.color

    @property
    def value(self) -> int:
        """Returns the numeric value of the card (1-13)."""
        return self.rank.numeric_value

    def __str__(self) -> str:
        """String representation for display (e.g., 'A♠', 'K♥')."""
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"Card({self.suit.name}, {self.rank.name})"

    def can_stack_on(self, other: Optional['Card']) -> bool:
        """
        Check if this card can be placed on another card in the tableau.

        Rules:
        - Must be one rank lower
        - Must be opposite color
        - Kings can go on empty piles (other=None)

        Args:
            other: The card to stack on, or None for empty pile

        Returns:
            True if the card can be placed on the other card
        """
        if other is None:
            # Only Kings can go on empty tableau piles
            return self.rank == Rank.KING

        # Must be opposite color and one rank lower
        return (self.color != other.color and
                self.value == other.value - 1)


class Deck:
    """
    Standard 52-card deck with shuffling support.

    The deck uses seeded random generation for reproducible games,
    which is essential for benchmarking and debugging.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize a new deck.

        Args:
            seed: Optional random seed for reproducible shuffling
        """
        self.cards: List[Card] = []
        self.seed = seed
        self._rng = random.Random(seed)
        self._initialize_deck()

    def _initialize_deck(self) -> None:
        """Create a standard 52-card deck in standard order."""
        self.cards = [
            Card(suit, rank)
            for suit in Suit
            for rank in Rank
        ]

    def shuffle(self) -> None:
        """Shuffle the deck using the seeded RNG."""
        self._rng.shuffle(self.cards)

    def draw(self, n: int = 1) -> List[Card]:
        """
        Draw n cards from the top of the deck.

        Args:
            n: Number of cards to draw

        Returns:
            List of drawn cards

        Raises:
            ValueError: If trying to draw more cards than available
        """
        if n > len(self.cards):
            raise ValueError(
                f"Cannot draw {n} cards, only {len(self.cards)} remaining"
            )

        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def draw_one(self) -> Card:
        """
        Draw a single card from the deck.

        Returns:
            The drawn card

        Raises:
            ValueError: If the deck is empty
        """
        if not self.cards:
            raise ValueError("Cannot draw from empty deck")
        return self.draw(1)[0]

    def is_empty(self) -> bool:
        """Check if the deck is empty."""
        return len(self.cards) == 0

    def __len__(self) -> int:
        """Return the number of cards remaining in the deck."""
        return len(self.cards)

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"Deck(cards={len(self.cards)}, seed={self.seed})"

    def __str__(self) -> str:
        """String representation showing card count."""
        return f"Deck with {len(self.cards)} cards"

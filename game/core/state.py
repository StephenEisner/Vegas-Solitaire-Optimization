"""
Game state representation for Vegas Solitaire.

This module defines the complete game state including all card positions,
scores, and move history. States are designed to be copyable and hashable
for use in search algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import copy
from game.core.card import Card, Suit


@dataclass
class GameState:
    """
    Complete representation of a Vegas Solitaire game state.

    Attributes:
        stock: Cards remaining in the draw pile
        waste: Cards that have been drawn (top 3 visible in Vegas rules)
        tableau: 7 columns of cards on the playing field
        tableau_hidden: Number of face-down cards in each tableau column
        foundations: Four piles building from Ace to King by suit
        score: Current score (Vegas: start at -$52, +$5 per foundation card)
        move_count: Number of moves made so far
    """
    stock: List[Card] = field(default_factory=list)
    waste: List[Card] = field(default_factory=list)
    tableau: List[List[Card]] = field(default_factory=lambda: [[] for _ in range(7)])
    tableau_hidden: List[int] = field(default_factory=lambda: [0] * 7)
    foundations: Dict[Suit, List[Card]] = field(default_factory=lambda: {suit: [] for suit in Suit})
    score: int = -52  # Vegas entry cost
    move_count: int = 0

    def copy(self) -> 'GameState':
        """
        Create a deep copy of the game state.

        This is essential for search algorithms that need to explore
        different move sequences without affecting the original state.

        Returns:
            A completely independent copy of this state
        """
        return GameState(
            stock=self.stock.copy(),
            waste=self.waste.copy(),
            tableau=[col.copy() for col in self.tableau],
            tableau_hidden=self.tableau_hidden.copy(),
            foundations={suit: cards.copy() for suit, cards in self.foundations.items()},
            score=self.score,
            move_count=self.move_count
        )

    def __hash__(self) -> int:
        """
        Generate hash for state caching in search algorithms.

        We hash only the visible game state, not the move count or
        internal history. This allows transposition table usage.

        Returns:
            Integer hash of the game state
        """
        # Convert lists to tuples for hashing
        stock_tuple = tuple(self.stock)
        waste_tuple = tuple(self.waste)
        tableau_tuple = tuple(tuple(col) for col in self.tableau)
        tableau_hidden_tuple = tuple(self.tableau_hidden)
        # Sort foundations by suit name for consistent hashing
        foundations_tuple = tuple(
            (suit, tuple(cards)) for suit, cards in sorted(self.foundations.items(), key=lambda x: x[0].name)
        )

        return hash((
            stock_tuple,
            waste_tuple,
            tableau_tuple,
            tableau_hidden_tuple,
            foundations_tuple
        ))

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another state.

        Two states are equal if all their cards are in the same positions.
        Score and move count are not considered for equality.

        Args:
            other: Another object to compare with

        Returns:
            True if states are equal
        """
        if not isinstance(other, GameState):
            return False

        return (
            self.stock == other.stock and
            self.waste == other.waste and
            self.tableau == other.tableau and
            self.tableau_hidden == other.tableau_hidden and
            self.foundations == other.foundations
        )

    def is_winning(self) -> bool:
        """
        Check if this is a winning state.

        A state is winning if all 52 cards are in the foundations.

        Returns:
            True if the game is won
        """
        return self.get_foundation_count() == 52

    def get_foundation_count(self) -> int:
        """
        Count total number of cards in all foundations.

        Returns:
            Number of cards in foundations (0-52)
        """
        return sum(len(cards) for cards in self.foundations.values())

    def get_tableau_visible_cards(self, column: int) -> List[Card]:
        """
        Get only the visible (face-up) cards in a tableau column.

        Args:
            column: Tableau column index (0-6)

        Returns:
            List of visible cards in the column (bottom to top)
        """
        if not 0 <= column < 7:
            raise ValueError(f"Column must be 0-6, got {column}")

        hidden_count = self.tableau_hidden[column]
        return self.tableau[column][hidden_count:]

    def is_tableau_column_empty(self, column: int) -> bool:
        """
        Check if a tableau column is empty.

        Args:
            column: Tableau column index (0-6)

        Returns:
            True if the column has no cards
        """
        if not 0 <= column < 7:
            raise ValueError(f"Column must be 0-6, got {column}")

        return len(self.tableau[column]) == 0

    def get_top_waste_card(self) -> Optional[Card]:
        """
        Get the top card from the waste pile (most recently drawn).

        Returns:
            The top waste card, or None if waste is empty
        """
        return self.waste[-1] if self.waste else None

    def get_top_tableau_card(self, column: int) -> Optional[Card]:
        """
        Get the top visible card from a tableau column.

        Args:
            column: Tableau column index (0-6)

        Returns:
            The top card, or None if column is empty
        """
        if not 0 <= column < 7:
            raise ValueError(f"Column must be 0-6, got {column}")

        visible = self.get_tableau_visible_cards(column)
        return visible[-1] if visible else None

    def get_foundation_top(self, suit: Suit) -> Optional[Card]:
        """
        Get the top card of a foundation pile.

        Args:
            suit: The suit of the foundation

        Returns:
            The top card, or None if foundation is empty
        """
        cards = self.foundations.get(suit, [])
        return cards[-1] if cards else None

    def to_vector(self) -> List[float]:
        """
        Convert state to fixed-size vector for ML models.

        This is a stub that will be implemented in Phase 4 when we
        add machine learning components.

        Returns:
            Vector representation of the state
        """
        # TODO: Implement in Phase 4 (ML)
        # Will encode:
        # - Card positions (one-hot or embedding)
        # - Visibility masks
        # - Foundation states
        # - Additional features (empty columns, etc.)
        raise NotImplementedError("ML state encoding will be implemented in Phase 4")

    def __str__(self) -> str:
        """
        Human-readable string representation of the game state.

        Returns:
            Multi-line string showing all game elements
        """
        lines = []

        # Header with score and moves
        lines.append(f"Score: ${self.score}  |  Moves: {self.move_count}")
        lines.append("=" * 60)

        # Stock and waste
        stock_str = f"Stock: [{len(self.stock)}]"
        waste_cards = " ".join(str(card) for card in self.waste[-3:]) if self.waste else "empty"
        lines.append(f"{stock_str}  |  Waste: {waste_cards}")
        lines.append("")

        # Foundations
        foundation_strs = []
        for suit in Suit:
            cards = self.foundations[suit]
            if cards:
                top = cards[-1].rank
                foundation_strs.append(f"{suit}[{top}({len(cards)})]")
            else:
                foundation_strs.append(f"{suit}[--]")
        lines.append("Foundations: " + "  ".join(foundation_strs))
        lines.append("")

        # Tableau
        lines.append("Tableau:")

        # Column headers
        lines.append("  " + "    ".join(str(i) for i in range(7)))

        # Find max column height
        max_height = max(len(col) for col in self.tableau) if any(self.tableau) else 0

        # Print each row
        for row in range(max_height):
            row_str = []
            for col in range(7):
                if row < len(self.tableau[col]):
                    card = self.tableau[col][row]
                    if row < self.tableau_hidden[col]:
                        row_str.append("[▓▓]")  # Hidden card
                    else:
                        card_str = str(card)
                        # Pad to 4 characters for alignment
                        row_str.append(f"{card_str:>4}")
                else:
                    row_str.append("    ")
            lines.append("  " + " ".join(row_str))

        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Detailed representation for debugging.

        Returns:
            String with key state information
        """
        return (
            f"GameState(stock={len(self.stock)}, waste={len(self.waste)}, "
            f"foundations={self.get_foundation_count()}, score={self.score}, "
            f"moves={self.move_count}, winning={self.is_winning()})"
        )

"""
Move representation for Vegas Solitaire.

This module defines all possible move types and the Move dataclass
that encapsulates all information needed to execute and display moves.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from game.core.card import Card


class MoveType(Enum):
    """
    All possible move types in Vegas Solitaire.

    Each move type corresponds to a specific game action:
    - TABLEAU_TO_FOUNDATION: Move card(s) from tableau to foundation
    - TABLEAU_TO_TABLEAU: Move sequence from one tableau column to another
    - WASTE_TO_FOUNDATION: Move top waste card to foundation
    - WASTE_TO_TABLEAU: Move top waste card to tableau
    - FOUNDATION_TO_TABLEAU: Move card from foundation back to tableau
    - DRAW: Draw 3 cards from stock to waste
    - RECYCLE: When stock is empty, flip waste pile back to stock
    """
    TABLEAU_TO_FOUNDATION = auto()
    TABLEAU_TO_TABLEAU = auto()
    WASTE_TO_FOUNDATION = auto()
    WASTE_TO_TABLEAU = auto()
    FOUNDATION_TO_TABLEAU = auto()
    DRAW = auto()
    RECYCLE = auto()

    def __str__(self) -> str:
        """Human-readable move type name."""
        return self.name.replace("_", " ").title()


@dataclass(frozen=True)
class Move:
    """
    Immutable representation of a game move.

    A move contains all information needed to:
    1. Execute the move on a game state
    2. Display the move to a user
    3. Validate the move

    Attributes:
        move_type: The type of move being made
        source: Source location (tableau column index, or None)
        destination: Destination location (tableau column or foundation)
        card_count: Number of cards being moved (for sequences)
        card: The card being moved (for display and validation)

    Examples:
        # Draw from stock
        Move(MoveType.DRAW, None, None, 0, None)

        # Move Ace from waste to foundation
        Move(MoveType.WASTE_TO_FOUNDATION, None, None, 1,
             Card(Suit.HEARTS, Rank.ACE))

        # Move 3-card sequence from column 2 to column 5
        Move(MoveType.TABLEAU_TO_TABLEAU, 2, 5, 3,
             Card(Suit.HEARTS, Rank.FIVE))
    """
    move_type: MoveType
    source: Optional[int] = None
    destination: Optional[int] = None
    card_count: int = 1
    card: Optional[Card] = None

    def __str__(self) -> str:
        """
        Human-readable string representation of the move.

        Returns:
            String describing the move in plain English
        """
        if self.move_type == MoveType.DRAW:
            return "Draw from stock"

        elif self.move_type == MoveType.RECYCLE:
            return "Recycle waste to stock"

        elif self.move_type == MoveType.WASTE_TO_FOUNDATION:
            card_str = str(self.card) if self.card else "card"
            return f"Move {card_str} from waste to foundation"

        elif self.move_type == MoveType.WASTE_TO_TABLEAU:
            card_str = str(self.card) if self.card else "card"
            return f"Move {card_str} from waste to column {self.destination}"

        elif self.move_type == MoveType.TABLEAU_TO_FOUNDATION:
            card_str = str(self.card) if self.card else "card"
            return f"Move {card_str} from column {self.source} to foundation"

        elif self.move_type == MoveType.TABLEAU_TO_TABLEAU:
            card_str = str(self.card) if self.card else "card"
            if self.card_count > 1:
                return f"Move {self.card_count} cards ({card_str}+) from column {self.source} to column {self.destination}"
            else:
                return f"Move {card_str} from column {self.source} to column {self.destination}"

        return f"Unknown move: {self.move_type}"

    def __repr__(self) -> str:
        """
        Detailed representation for debugging.

        Returns:
            String with all move attributes
        """
        card_repr = repr(self.card) if self.card else "None"
        return (
            f"Move(type={self.move_type.name}, "
            f"source={self.source}, dest={self.destination}, "
            f"count={self.card_count}, card={card_repr})"
        )

    def is_draw_move(self) -> bool:
        """Check if this is a draw or recycle move."""
        return self.move_type in (MoveType.DRAW, MoveType.RECYCLE)

    def is_foundation_move(self) -> bool:
        """Check if this move adds to a foundation."""
        return self.move_type in (
            MoveType.TABLEAU_TO_FOUNDATION,
            MoveType.WASTE_TO_FOUNDATION
        )

    def is_tableau_move(self) -> bool:
        """Check if this move involves tableau manipulation."""
        return self.move_type in (
            MoveType.TABLEAU_TO_TABLEAU,
            MoveType.WASTE_TO_TABLEAU,
            MoveType.TABLEAU_TO_FOUNDATION
        )

    def reveals_card(self) -> bool:
        """
        Check if this move might reveal a hidden card.

        This is useful for heuristic evaluation.

        Returns:
            True if the move removes the last visible card from a tableau column
        """
        # Moving from tableau might reveal a card
        if self.move_type in (MoveType.TABLEAU_TO_FOUNDATION,
                               MoveType.TABLEAU_TO_TABLEAU):
            # Note: Actual revelation depends on state, this is a hint
            return True
        return False

    def empties_column(self) -> bool:
        """
        Check if this move might empty a tableau column.

        Returns:
            True if the move could empty a column
        """
        # Moving from tableau might empty the column
        if self.move_type in (MoveType.TABLEAU_TO_FOUNDATION,
                               MoveType.TABLEAU_TO_TABLEAU):
            # Note: Actual emptying depends on state, this is a hint
            return True
        return False


def create_draw_move() -> Move:
    """
    Create a draw move (draw 3 cards from stock to waste).

    Returns:
        Move representing a draw action
    """
    return Move(MoveType.DRAW)


def create_recycle_move() -> Move:
    """
    Create a recycle move (flip waste back to stock when stock is empty).

    Returns:
        Move representing a recycle action
    """
    return Move(MoveType.RECYCLE)


def create_waste_to_foundation_move(card: Card) -> Move:
    """
    Create a move from waste to foundation.

    Args:
        card: The card being moved

    Returns:
        Move representing waste to foundation
    """
    return Move(MoveType.WASTE_TO_FOUNDATION, card=card)


def create_waste_to_tableau_move(card: Card, destination: int) -> Move:
    """
    Create a move from waste to tableau.

    Args:
        card: The card being moved
        destination: Target tableau column (0-6)

    Returns:
        Move representing waste to tableau
    """
    return Move(MoveType.WASTE_TO_TABLEAU, destination=destination, card=card)


def create_tableau_to_foundation_move(source: int, card: Card) -> Move:
    """
    Create a move from tableau to foundation.

    Args:
        source: Source tableau column (0-6)
        card: The card being moved

    Returns:
        Move representing tableau to foundation
    """
    return Move(MoveType.TABLEAU_TO_FOUNDATION, source=source, card=card)


def create_tableau_to_tableau_move(source: int, destination: int,
                                     card: Card, card_count: int = 1) -> Move:
    """
    Create a move from tableau to tableau.

    Args:
        source: Source tableau column (0-6)
        destination: Destination tableau column (0-6)
        card: The bottom card of the sequence being moved
        card_count: Number of cards in the sequence

    Returns:
        Move representing tableau to tableau
    """
    return Move(
        MoveType.TABLEAU_TO_TABLEAU,
        source=source,
        destination=destination,
        card_count=card_count,
        card=card
    )


def create_foundation_to_tableau_move(suit, destination: int, card: Card) -> Move:
    """
    Create a move from foundation to tableau.

    Args:
        suit: The suit of the foundation pile
        destination: Destination tableau column (0-6)
        card: The card being moved

    Returns:
        Move representing foundation to tableau
    """
    return Move(
        MoveType.FOUNDATION_TO_TABLEAU,
        destination=destination,
        card=card
    )

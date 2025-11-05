"""
Rules engine for Vegas Solitaire.

This module implements move validation, move generation, and move application.
These functions are called intensively during search, so efficiency is critical.
"""

from typing import List, Optional
from game.core.card import Card, Suit, Rank
from game.core.state import GameState
from game.core.moves import (
    Move, MoveType,
    create_draw_move, create_recycle_move,
    create_waste_to_foundation_move, create_waste_to_tableau_move,
    create_tableau_to_foundation_move, create_tableau_to_tableau_move
)


def can_move_to_foundation(card: Card, state: GameState) -> bool:
    """
    Check if a card can be moved to its foundation pile.

    Foundation rules:
    - Aces can start empty foundations
    - Other cards must be one rank higher than current foundation top
    - Must match suit

    Args:
        card: The card to check
        state: Current game state

    Returns:
        True if the card can be moved to foundation
    """
    foundation = state.foundations[card.suit]

    if not foundation:
        # Empty foundation - only Aces allowed
        return card.rank == Rank.ACE

    # Foundation has cards - must be next rank in sequence
    top_card = foundation[-1]
    return card.value == top_card.value + 1


def can_move_to_tableau(card: Card, tableau_column: List[Card],
                        hidden_count: int) -> bool:
    """
    Check if a card can be moved to a tableau column.

    Tableau rules:
    - Kings can go on empty columns
    - Other cards must be:
      - One rank lower than the current top card
      - Opposite color from the current top card

    Args:
        card: The card to check
        tableau_column: The target tableau column
        hidden_count: Number of hidden cards in the column

    Returns:
        True if the card can be moved to the tableau column
    """
    visible_cards = tableau_column[hidden_count:]

    if not visible_cards:
        # Empty column - only Kings allowed
        return card.rank == Rank.KING

    # Must stack on top visible card
    top_card = visible_cards[-1]
    return card.can_stack_on(top_card)


def is_valid_sequence(cards: List[Card]) -> bool:
    """
    Check if a list of cards forms a valid descending alternating sequence.

    Args:
        cards: List of cards to check (bottom to top)

    Returns:
        True if the sequence is valid for tableau
    """
    if len(cards) <= 1:
        return True

    for i in range(len(cards) - 1):
        if not cards[i + 1].can_stack_on(cards[i]):
            return False

    return True


def get_valid_moves(state: GameState) -> List[Move]:
    """
    Generate all valid moves from the current state.

    This is a critical function called millions of times during search.
    Order of move generation can affect search performance.

    Args:
        state: Current game state

    Returns:
        List of all valid moves
    """
    moves: List[Move] = []

    # 1. Draw/Recycle moves (always consider stock first)
    if state.stock:
        moves.append(create_draw_move())
    elif state.waste:
        # Can recycle if stock is empty but waste has cards
        moves.append(create_recycle_move())

    # 2. Foundation moves from waste (high priority - progress toward win)
    if state.waste:
        top_waste = state.waste[-1]
        if can_move_to_foundation(top_waste, state):
            moves.append(create_waste_to_foundation_move(top_waste))

    # 3. Foundation moves from tableau (high priority)
    for col_idx in range(7):
        visible = state.get_tableau_visible_cards(col_idx)
        if visible:
            top_card = visible[-1]
            if can_move_to_foundation(top_card, state):
                moves.append(create_tableau_to_foundation_move(col_idx, top_card))

    # 4. Tableau moves from waste (medium priority)
    if state.waste:
        top_waste = state.waste[-1]
        for col_idx in range(7):
            if can_move_to_tableau(top_waste, state.tableau[col_idx],
                                   state.tableau_hidden[col_idx]):
                moves.append(create_waste_to_tableau_move(top_waste, col_idx))

    # 5. Tableau to tableau moves (most complex - can move sequences)
    for source_col in range(7):
        visible = state.get_tableau_visible_cards(source_col)
        if not visible:
            continue

        # Try moving sequences of different lengths
        # Start from single card at top, then try longer sequences
        for seq_start in range(len(visible)):
            sequence = visible[seq_start:]
            if not is_valid_sequence(sequence):
                continue

            bottom_card = sequence[0]
            seq_length = len(sequence)

            # Try moving this sequence to each other column
            for dest_col in range(7):
                if dest_col == source_col:
                    continue

                if can_move_to_tableau(bottom_card, state.tableau[dest_col],
                                       state.tableau_hidden[dest_col]):
                    moves.append(create_tableau_to_tableau_move(
                        source_col, dest_col, bottom_card, seq_length
                    ))

    return moves


def apply_move(state: GameState, move: Move) -> GameState:
    """
    Apply a move to a state, creating a new state.

    This function does NOT validate the move - it assumes the move is valid.
    Use get_valid_moves() or validation functions before calling this.

    Args:
        state: Current game state
        move: Move to apply

    Returns:
        New game state after the move
    """
    # Create a copy of the state
    new_state = state.copy()
    new_state.move_count += 1

    if move.move_type == MoveType.DRAW:
        # Draw 3 cards from stock to waste
        cards_to_draw = min(3, len(new_state.stock))
        drawn_cards = new_state.stock[:cards_to_draw]
        new_state.stock = new_state.stock[cards_to_draw:]
        new_state.waste.extend(drawn_cards)

    elif move.move_type == MoveType.RECYCLE:
        # Flip waste back to stock
        new_state.stock = new_state.waste[::-1]  # Reverse order
        new_state.waste = []
        new_state.passes_through_deck += 1

    elif move.move_type == MoveType.WASTE_TO_FOUNDATION:
        # Move top waste card to foundation
        card = new_state.waste.pop()
        new_state.foundations[card.suit].append(card)
        new_state.score += 5  # Vegas scoring: $5 per foundation card

    elif move.move_type == MoveType.WASTE_TO_TABLEAU:
        # Move top waste card to tableau
        card = new_state.waste.pop()
        new_state.tableau[move.destination].append(card)

    elif move.move_type == MoveType.TABLEAU_TO_FOUNDATION:
        # Move top tableau card to foundation
        card = new_state.tableau[move.source].pop()
        new_state.foundations[card.suit].append(card)
        new_state.score += 5  # Vegas scoring

        # Reveal hidden card if column is not empty
        if new_state.tableau[move.source] and new_state.tableau_hidden[move.source] > 0:
            new_state.tableau_hidden[move.source] -= 1

    elif move.move_type == MoveType.TABLEAU_TO_TABLEAU:
        # Move sequence from one tableau column to another
        source_col = new_state.tableau[move.source]
        hidden_count = new_state.tableau_hidden[move.source]
        visible_cards = source_col[hidden_count:]

        # Find the sequence to move (last card_count cards)
        sequence = visible_cards[-move.card_count:]

        # Remove from source
        new_state.tableau[move.source] = source_col[:-move.card_count]

        # Add to destination
        new_state.tableau[move.destination].extend(sequence)

        # Reveal hidden card if source column is not empty
        if new_state.tableau[move.source] and new_state.tableau_hidden[move.source] > 0:
            new_state.tableau_hidden[move.source] -= 1

    return new_state


def is_valid_move(state: GameState, move: Move) -> bool:
    """
    Check if a move is valid in the current state.

    Args:
        state: Current game state
        move: Move to validate

    Returns:
        True if the move is valid
    """
    if move.move_type == MoveType.DRAW:
        return len(state.stock) > 0

    elif move.move_type == MoveType.RECYCLE:
        # Can only recycle if stock is empty, waste has cards, and haven't passed through deck 3 times yet
        return len(state.stock) == 0 and len(state.waste) > 0 and state.passes_through_deck < 3

    elif move.move_type == MoveType.WASTE_TO_FOUNDATION:
        if not state.waste:
            return False
        return can_move_to_foundation(state.waste[-1], state)

    elif move.move_type == MoveType.WASTE_TO_TABLEAU:
        if not state.waste or move.destination is None:
            return False
        return can_move_to_tableau(
            state.waste[-1],
            state.tableau[move.destination],
            state.tableau_hidden[move.destination]
        )

    elif move.move_type == MoveType.TABLEAU_TO_FOUNDATION:
        if move.source is None:
            return False
        visible = state.get_tableau_visible_cards(move.source)
        if not visible:
            return False
        return can_move_to_foundation(visible[-1], state)

    elif move.move_type == MoveType.TABLEAU_TO_TABLEAU:
        if move.source is None or move.destination is None:
            return False
        if move.source == move.destination:
            return False

        visible = state.get_tableau_visible_cards(move.source)
        if not visible or len(visible) < move.card_count:
            return False

        sequence = visible[-move.card_count:]
        if not is_valid_sequence(sequence):
            return False

        return can_move_to_tableau(
            sequence[0],
            state.tableau[move.destination],
            state.tableau_hidden[move.destination]
        )

    return False

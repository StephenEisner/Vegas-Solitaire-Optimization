"""
Heuristic solver for Vegas Solitaire.

This solver uses hand-crafted heuristics to evaluate and choose moves.
It represents a "smart human" strategy based on common solitaire wisdom.
"""

from typing import Optional, List, Tuple
from game.core.game import Game
from game.core.moves import Move, MoveType
from game.core.state import GameState
from optimization.solvers.base import Solver


class HeuristicSolver(Solver):
    """
    Solver that uses heuristics to evaluate moves.

    Heuristics used (in priority order):
    1. Foundation moves (always good)
    2. Revealing hidden cards (valuable)
    3. Creating empty columns (very valuable)
    4. Moving from waste to tableau (clears waste)
    5. Drawing cards (explores options)
    6. Tableau to tableau (reorganization)

    Expected performance: ~15-25% win rate
    """

    def __init__(self):
        """Initialize the heuristic solver."""
        super().__init__(name="Heuristic")

        # Heuristic weights (tuned through experimentation)
        self.weights = {
            'foundation_move': 100,           # Always prefer foundation
            'reveals_hidden_card': 50,        # Very valuable
            'empties_column': 80,             # Creates space for Kings
            'creates_sequence': 10,           # Building sequences is good
            'moves_from_waste': 15,           # Clear waste pile
            'tableau_to_empty_column': 60,    # Use empty columns wisely
            'sequence_length': 3,             # Longer sequences = better
            'draw_move': 5,                   # Neutral, but necessary
        }

    def choose_move(self, game: Game) -> Optional[Move]:
        """
        Choose the best move based on heuristic evaluation.

        Args:
            game: Current game state

        Returns:
            Best move according to heuristics, or None if no moves
        """
        valid_moves = game.get_valid_moves()

        if not valid_moves:
            return None

        # Score each move
        scored_moves: List[Tuple[Move, float]] = []
        for move in valid_moves:
            score = self.evaluate_move(move, game.state)
            scored_moves.append((move, score))

        # Sort by score (descending)
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        # Return best move
        return scored_moves[0][0]

    def evaluate_move(self, move: Move, state: GameState) -> float:
        """
        Evaluate a move using heuristics.

        Args:
            move: Move to evaluate
            state: Current game state

        Returns:
            Score for the move (higher is better)
        """
        score = 0.0

        # 1. Foundation moves are almost always good
        if move.is_foundation_move():
            score += self.weights['foundation_move']

            # Extra bonus for Aces (start foundations)
            if move.card and move.card.rank.numeric_value == 1:
                score += 20

        # 2. Moves that reveal hidden cards
        if self._reveals_card(move, state):
            score += self.weights['reveals_hidden_card']

        # 3. Moves that empty a column (very valuable)
        if self._empties_column(move, state):
            score += self.weights['empties_column']

        # 4. Moving from waste to tableau clears waste
        if move.move_type == MoveType.WASTE_TO_TABLEAU:
            score += self.weights['moves_from_waste']

            # Bonus for Kings to empty columns
            if move.card and move.card.rank.numeric_value == 13:
                if move.destination is not None:
                    if len(state.tableau[move.destination]) == 0:
                        score += self.weights['tableau_to_empty_column']

        # 5. Tableau to tableau moves
        if move.move_type == MoveType.TABLEAU_TO_TABLEAU:
            # Moving to empty column
            if move.destination is not None:
                if len(state.tableau[move.destination]) == 0:
                    score += self.weights['tableau_to_empty_column']

            # Longer sequences are better
            score += move.card_count * self.weights['sequence_length']

        # 6. Draw moves are neutral but necessary
        if move.move_type == MoveType.DRAW:
            score += self.weights['draw_move']

        # 7. Recycle is last resort
        if move.move_type == MoveType.RECYCLE:
            score -= 10  # Slight penalty

        return score

    def _reveals_card(self, move: Move, state: GameState) -> bool:
        """
        Check if a move will reveal a hidden card.

        Args:
            move: Move to check
            state: Current game state

        Returns:
            True if the move reveals a hidden card
        """
        if move.move_type not in (MoveType.TABLEAU_TO_FOUNDATION,
                                   MoveType.TABLEAU_TO_TABLEAU):
            return False

        if move.source is None:
            return False

        # Check if there are hidden cards in the source column
        hidden_count = state.tableau_hidden[move.source]
        if hidden_count == 0:
            return False

        # Check if we're removing all visible cards
        visible_cards = state.get_tableau_visible_cards(move.source)
        if len(visible_cards) == move.card_count:
            # Removing all visible cards will reveal a hidden one
            return True

        return False

    def _empties_column(self, move: Move, state: GameState) -> bool:
        """
        Check if a move empties a tableau column.

        Args:
            move: Move to check
            state: Current game state

        Returns:
            True if the move empties a column
        """
        if move.move_type not in (MoveType.TABLEAU_TO_FOUNDATION,
                                   MoveType.TABLEAU_TO_TABLEAU):
            return False

        if move.source is None:
            return False

        # Check if we're removing all cards from the column
        column = state.tableau[move.source]
        hidden_count = state.tableau_hidden[move.source]
        visible_cards = column[hidden_count:]

        # Moving all visible cards from a column with no hidden cards empties it
        return (len(visible_cards) == move.card_count and hidden_count == 0)

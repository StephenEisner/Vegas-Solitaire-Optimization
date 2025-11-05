"""
Heuristic solver for Vegas Solitaire.

This solver uses hand-crafted heuristics to evaluate and choose moves.
It represents a "smart human" strategy based on common solitaire wisdom.
"""

from typing import Optional, List, Tuple
from game.core.game import Game
from game.core.moves import Move, MoveType
from game.core.state import GameState
from game.core.rules import apply_move
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

        # Cycle detection
        self.state_history = []
        self.max_history = 10  # Track last N states

        # Improved heuristic weights
        self.weights = {
            # Direct progress (highest priority)
            'foundation_move': 100,           # Always prefer foundation
            'foundation_ace': 20,             # Extra bonus for Aces

            # Information gain
            'reveals_hidden_card': 50,        # Very valuable
            'reveals_from_large_stack': 10,   # Bonus for big stacks

            # Space management
            'empties_column': 80,             # Creates space for Kings
            'king_to_empty': 50,              # Only use empty for Kings
            'non_king_to_empty': -30,         # Penalty for wasting space

            # Sequencing
            'sequence_length': 3,             # Longer sequences = better (only when revealing!)
            'creates_sequence': 10,           # Building sequences is good

            # Waste management
            'moves_from_waste': 15,           # Clear waste pile
            'waste_blocks_foundation': 25,    # Higher if waste card blocks progress

            # Movement types
            'draw_move': 10,                  # Encourage drawing to see new cards
            'recycle_penalty': -10,           # Avoid recycling if possible

            # Penalties
            'tableau_shuffle_no_reveal': -40, # Heavy penalty for non-progressive moves
            'foundation_to_tableau': -50,     # Usually bad to move back
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

        # Filter out moves that create cycles
        non_cycling_moves = []
        for move, score in scored_moves:
            # Simulate the move
            next_state = apply_move(game.state, move)

            # Check if this state was recently seen
            if not self._is_recent_state(next_state):
                non_cycling_moves.append((move, score))

        # Update state history with current state
        self._add_to_history(game.state)

        # If all moves cycle, take the best one anyway (break the cycle)
        if not non_cycling_moves:
            # Take best move but penalize it
            return scored_moves[0][0]

        # Return best non-cycling move
        return non_cycling_moves[0][0]

    def evaluate_move(self, move: Move, state: GameState) -> float:
        """
        Evaluate a move using improved heuristics.

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
                score += self.weights['foundation_ace']

        # 2. Foundation to Tableau (usually bad)
        if move.move_type == MoveType.FOUNDATION_TO_TABLEAU:
            score += self.weights['foundation_to_tableau']

        # 3. Moves that reveal hidden cards
        if self._reveals_card(move, state):
            score += self.weights['reveals_hidden_card']

            # Bonus for revealing from large stacks
            if move.source is not None:
                hidden_count = state.tableau_hidden[move.source]
                if hidden_count > 3:
                    score += self.weights['reveals_from_large_stack']

        # 4. Moves that empty a column (very valuable)
        if self._empties_column(move, state):
            score += self.weights['empties_column']

        # 5. Moving from waste to tableau
        if move.move_type == MoveType.WASTE_TO_TABLEAU:
            score += self.weights['moves_from_waste']

            # Check if move is to empty column
            if move.destination is not None and len(state.tableau[move.destination]) == 0:
                # Only Kings should go to empty columns
                if move.card and move.card.rank.numeric_value == 13:
                    score += self.weights['king_to_empty']
                else:
                    score += self.weights['non_king_to_empty']

        # 6. Tableau to tableau moves
        if move.move_type == MoveType.TABLEAU_TO_TABLEAU:
            # Check if moving to empty column
            if move.destination is not None and len(state.tableau[move.destination]) == 0:
                # Only Kings should go to empty columns
                if move.card and move.card.rank.numeric_value == 13:
                    score += self.weights['king_to_empty']
                else:
                    score += self.weights['non_king_to_empty']
            else:
                # Regular tableau move
                reveals = self._reveals_card(move, state)

                if reveals:
                    # Only give sequence bonus if revealing a card
                    score += move.card_count * self.weights['sequence_length']
                else:
                    # Heavy penalty for moving cards without revealing
                    # Bigger penalty for moving more cards (wasteful shuffling)
                    score += self.weights['tableau_shuffle_no_reveal']
                    score -= move.card_count * 5  # Strong penalty per card moved without purpose

        # 7. Draw moves are neutral but necessary
        if move.move_type == MoveType.DRAW:
            score += self.weights['draw_move']

        # 8. Recycle penalty
        if move.move_type == MoveType.RECYCLE:
            score += self.weights['recycle_penalty']

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

    def _is_recent_state(self, state: GameState) -> bool:
        """
        Check if a state was recently seen (cycle detection).

        Args:
            state: State to check

        Returns:
            True if state was in recent history
        """
        state_hash = hash(state)
        return state_hash in self.state_history

    def _add_to_history(self, state: GameState) -> None:
        """
        Add a state to the history.

        Args:
            state: State to add
        """
        state_hash = hash(state)
        self.state_history.append(state_hash)

        # Keep only last N states
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

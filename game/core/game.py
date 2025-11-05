"""
Game controller for Vegas Solitaire.

This module provides the main Game class that orchestrates all game components:
cards, state, rules, and move history. It provides a clean API for playing games.
"""

from typing import List, Optional
from game.core.card import Deck, Card
from game.core.state import GameState
from game.core.moves import Move
from game.core.rules import get_valid_moves, apply_move, is_valid_move


class Game:
    """
    Main game controller for Vegas Solitaire.

    This class manages the complete game lifecycle including dealing,
    move making, and game state tracking.

    Attributes:
        state: Current game state
        deck: The deck used for this game
        move_history: List of all moves made so far
        seed: Random seed for reproducibility
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize a new game.

        Args:
            seed: Optional random seed for reproducible games
        """
        self.seed = seed
        self.deck = Deck(seed=seed)
        self.state = GameState()
        self.move_history: List[Move] = []
        self._dealt = False

    def deal(self) -> None:
        """
        Deal cards to start a new game.

        Vegas Solitaire setup:
        - Shuffle the deck
        - Deal to tableau: 1 card to column 0, 2 to column 1, ..., 7 to column 6
        - First card in each column is face up, rest are face down
        - Remaining 24 cards go to stock
        - Waste starts empty
        - Foundations start empty
        - Score starts at -$52 (Vegas entry cost)

        Raises:
            RuntimeError: If the game has already been dealt
        """
        if self._dealt:
            raise RuntimeError("Game has already been dealt")

        self.deck.shuffle()

        # Deal to tableau
        for col in range(7):
            num_cards = col + 1
            cards = self.deck.draw(num_cards)
            self.state.tableau[col] = cards
            # All but the last card are hidden
            self.state.tableau_hidden[col] = num_cards - 1

        # Remaining cards go to stock
        self.state.stock = self.deck.cards.copy()
        self.deck.cards = []  # Deck is now empty

        self._dealt = True

    def get_valid_moves(self) -> List[Move]:
        """
        Get all valid moves from the current state.

        Returns:
            List of valid moves
        """
        return get_valid_moves(self.state)

    def make_move(self, move: Move) -> bool:
        """
        Make a move if it's valid.

        This validates the move, applies it, and records it in history.

        Args:
            move: The move to make

        Returns:
            True if the move was valid and applied, False otherwise
        """
        if not is_valid_move(self.state, move):
            return False

        # Apply the move
        self.state = apply_move(self.state, move)
        self.move_history.append(move)

        return True

    def is_over(self) -> bool:
        """
        Check if the game is over.

        A game is over when:
        1. All cards are in foundations (win), OR
        2. No valid moves are available (loss)

        Returns:
            True if the game is over
        """
        if self.state.is_winning():
            return True

        # Check if any non-draw/recycle moves are available
        # (Draw/recycle can cycle forever, so we need other moves)
        valid_moves = self.get_valid_moves()
        non_cycle_moves = [
            m for m in valid_moves
            if not m.is_draw_move()
        ]

        return len(non_cycle_moves) == 0

    def is_winning(self) -> bool:
        """
        Check if the game is in a winning state.

        Returns:
            True if all 52 cards are in foundations
        """
        return self.state.is_winning()

    def get_score(self) -> int:
        """
        Get the current score.

        Vegas scoring:
        - Start at -$52 (entry cost)
        - Earn $5 for each card in foundations

        Returns:
            Current score in dollars
        """
        return self.state.score

    def get_move_count(self) -> int:
        """
        Get the number of moves made so far.

        Returns:
            Number of moves
        """
        return self.state.move_count

    def get_state_copy(self) -> GameState:
        """
        Get a copy of the current game state.

        This is useful for search algorithms that need to explore
        different move sequences without affecting the actual game.

        Returns:
            Deep copy of current state
        """
        return self.state.copy()

    def undo(self) -> bool:
        """
        Undo the last move (for debugging/testing).

        Note: This requires replaying the entire game from the beginning,
        so it's O(n) where n is the number of moves. Not suitable for
        performance-critical code.

        Returns:
            True if undo was successful, False if no moves to undo
        """
        if not self.move_history:
            return False

        # Remove last move
        self.move_history.pop()

        # Replay game from start
        self._replay_from_beginning()

        return True

    def _replay_from_beginning(self) -> None:
        """
        Replay the game from the beginning using the move history.

        This is used by the undo() method.
        """
        # Save history and seed
        history = self.move_history.copy()
        seed = self.seed

        # Reset game
        self.__init__(seed=seed)
        self.deal()

        # Replay moves
        for move in history:
            success = self.make_move(move)
            if not success:
                raise RuntimeError(f"Failed to replay move: {move}")

    def get_game_summary(self) -> dict:
        """
        Get a summary of the current game state.

        Returns:
            Dictionary with game statistics
        """
        return {
            "seed": self.seed,
            "moves": self.get_move_count(),
            "score": self.get_score(),
            "cards_in_foundations": self.state.get_foundation_count(),
            "is_winning": self.is_winning(),
            "is_over": self.is_over(),
            "stock_size": len(self.state.stock),
            "waste_size": len(self.state.waste),
        }

    def __str__(self) -> str:
        """
        String representation showing game state.

        Returns:
            Multi-line string with game information
        """
        lines = []
        lines.append(f"Game (seed={self.seed})")
        lines.append(str(self.state))

        if self.is_winning():
            lines.append("\n🎉 YOU WIN! 🎉")
        elif self.is_over():
            lines.append("\n❌ Game Over - No more moves available")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Detailed representation for debugging.

        Returns:
            String with game information
        """
        return (
            f"Game(seed={self.seed}, moves={self.get_move_count()}, "
            f"score={self.get_score()}, "
            f"foundations={self.state.get_foundation_count()}/52)"
        )


def play_game_with_moves(seed: Optional[int] = None,
                         moves: Optional[List[Move]] = None) -> Game:
    """
    Helper function to create a game and play a sequence of moves.

    This is useful for testing and debugging specific game scenarios.

    Args:
        seed: Random seed for the game
        moves: List of moves to play (optional)

    Returns:
        Game object after all moves have been played

    Example:
        >>> from game.core.moves import create_draw_move
        >>> game = play_game_with_moves(seed=42, moves=[create_draw_move()])
        >>> print(game.get_move_count())
        1
    """
    game = Game(seed=seed)
    game.deal()

    if moves:
        for move in moves:
            if not game.make_move(move):
                print(f"Warning: Invalid move skipped: {move}")
                break

    return game


def play_random_game(seed: Optional[int] = None,
                     max_moves: int = 1000) -> Game:
    """
    Play a complete game making random moves.

    This is useful for testing and as a baseline for optimization.

    Args:
        seed: Random seed for the game
        max_moves: Maximum number of moves before stopping

    Returns:
        Game object after playing

    Example:
        >>> game = play_random_game(seed=42)
        >>> print(game.get_game_summary())
    """
    import random

    game = Game(seed=seed)
    game.deal()

    rng = random.Random(seed)
    move_count = 0

    while not game.is_over() and move_count < max_moves:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break

        # Choose a random move
        move = rng.choice(valid_moves)
        game.make_move(move)
        move_count += 1

    return game

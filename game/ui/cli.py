"""
Command-line interface for Vegas Solitaire.

This module provides a text-based playable interface for the game,
useful for testing, debugging, and human play.
"""

import sys
from typing import Optional, List
from game.core.game import Game
from game.core.moves import Move, MoveType


class CLI:
    """
    Command-line interface for Vegas Solitaire.

    Provides a text-based UI with ASCII art card display and
    interactive move selection.
    """

    def __init__(self, game: Optional[Game] = None):
        """
        Initialize the CLI.

        Args:
            game: Optional game instance. If None, creates a new game.
        """
        self.game = game if game else Game()

    def render_board(self) -> str:
        """
        Render the game board as ASCII art.

        Returns:
            Multi-line string representing the board
        """
        state = self.game.state
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(f"  Vegas Solitaire (Seed: {self.game.seed})")
        lines.append(f"  Score: ${state.score}  |  Moves: {state.move_count}  |  "
                    f"Foundation: {state.get_foundation_count()}/52")
        lines.append("=" * 70)
        lines.append("")

        # Stock and Waste
        stock_str = f"[{len(state.stock)}]" if state.stock else "[empty]"
        if state.waste:
            # Show last 3 cards in waste
            waste_cards = state.waste[-3:]
            waste_str = " ".join(str(card) for card in waste_cards)
        else:
            waste_str = "[empty]"

        lines.append(f"  Stock: {stock_str}    Waste: {waste_str}")
        lines.append("")

        # Foundations
        foundation_parts = []
        for suit in ["♠", "♥", "♦", "♣"]:
            from game.core.card import Suit
            suit_enum = {
                "♠": Suit.SPADES,
                "♥": Suit.HEARTS,
                "♦": Suit.DIAMONDS,
                "♣": Suit.CLUBS
            }[suit]

            foundation = state.foundations[suit_enum]
            if foundation:
                top_card = foundation[-1]
                foundation_parts.append(f"{suit}[{top_card.rank}]")
            else:
                foundation_parts.append(f"{suit}[--]")

        lines.append(f"  Foundations: {' '.join(foundation_parts)}")
        lines.append("")

        # Tableau
        lines.append("  Tableau:")
        lines.append("     " + "    ".join(f"[{i}]" for i in range(7)))

        # Find max height
        max_height = max((len(col) for col in state.tableau), default=0)

        # Render each row
        for row in range(max_height):
            row_parts = []
            for col in range(7):
                if row < len(state.tableau[col]):
                    card = state.tableau[col][row]
                    if row < state.tableau_hidden[col]:
                        row_parts.append("[▓▓]")  # Hidden card
                    else:
                        card_str = str(card)
                        row_parts.append(f"{card_str:>4}")
                else:
                    row_parts.append("    ")

            lines.append("     " + " ".join(row_parts))

        lines.append("")
        return "\n".join(lines)

    def render_moves(self, moves: List[Move]) -> str:
        """
        Render available moves as a numbered list.

        Args:
            moves: List of available moves

        Returns:
            Multi-line string with numbered moves
        """
        if not moves:
            return "  No moves available!"

        lines = ["  Available moves:"]
        for i, move in enumerate(moves, 1):
            lines.append(f"    {i:2d}. {move}")

        return "\n".join(lines)

    def display(self) -> None:
        """Display the current game state and available moves."""
        print(self.render_board())

        moves = self.game.get_valid_moves()
        print(self.render_moves(moves))
        print()

    def get_move_choice(self, moves: List[Move]) -> Optional[Move]:
        """
        Prompt user to choose a move.

        Args:
            moves: List of available moves

        Returns:
            Chosen move, or None to quit
        """
        if not moves:
            return None

        while True:
            try:
                choice = input("  Choose move (number), 'q' to quit, 'u' to undo: ").strip()

                if choice.lower() == 'q':
                    return None

                if choice.lower() == 'u':
                    if self.game.undo():
                        print("  ✓ Move undone!")
                        return self.get_move_choice(self.game.get_valid_moves())
                    else:
                        print("  ✗ No moves to undo!")
                        continue

                move_num = int(choice)
                if 1 <= move_num <= len(moves):
                    return moves[move_num - 1]
                else:
                    print(f"  ✗ Please choose a number between 1 and {len(moves)}")

            except ValueError:
                print("  ✗ Invalid input. Please enter a number, 'q', or 'u'")
            except KeyboardInterrupt:
                print("\n  Game interrupted!")
                return None

    def play_interactive(self) -> None:
        """
        Play an interactive game with user input.

        The game continues until the user quits or the game is over.
        """
        if not self.game._dealt:
            self.game.deal()

        print("\n" + "=" * 70)
        print("  Welcome to Vegas Solitaire!")
        print("=" * 70)
        print()

        while not self.game.is_over():
            self.display()

            moves = self.game.get_valid_moves()
            chosen_move = self.get_move_choice(moves)

            if chosen_move is None:
                print("\n  Game ended by user.")
                break

            success = self.game.make_move(chosen_move)
            if not success:
                print("  ✗ Move failed (this shouldn't happen!)")
            else:
                print(f"  ✓ Made move: {chosen_move}")
                print()

        # Game over
        self.display()
        summary = self.game.get_game_summary()

        print()
        print("=" * 70)
        if self.game.is_winning():
            print("  🎉 CONGRATULATIONS! YOU WIN! 🎉")
        else:
            print("  ❌ Game Over - No more moves available")

        print()
        print(f"  Final Score: ${summary['score']}")
        print(f"  Total Moves: {summary['moves']}")
        print(f"  Cards in Foundations: {summary['cards_in_foundations']}/52")
        print("=" * 70)
        print()

    def play_auto(self, strategy: str = "random", max_moves: int = 1000) -> None:
        """
        Play a game automatically using a strategy.

        Args:
            strategy: Strategy to use ("random" for now)
            max_moves: Maximum number of moves before stopping
        """
        if not self.game._dealt:
            self.game.deal()

        print(f"\n  Playing game automatically using '{strategy}' strategy...")
        print(f"  Seed: {self.game.seed}")
        print()

        import random
        rng = random.Random(self.game.seed)

        move_count = 0
        while not self.game.is_over() and move_count < max_moves:
            moves = self.game.get_valid_moves()
            if not moves:
                break

            # Choose move based on strategy
            if strategy == "random":
                move = rng.choice(moves)
            else:
                move = moves[0]  # Default to first move

            self.game.make_move(move)
            move_count += 1

        # Show results
        summary = self.game.get_game_summary()

        print("=" * 70)
        if self.game.is_winning():
            print("  🎉 WIN!")
        else:
            print("  ❌ Loss")

        print()
        print(f"  Final Score: ${summary['score']}")
        print(f"  Total Moves: {summary['moves']}")
        print(f"  Cards in Foundations: {summary['cards_in_foundations']}/52")
        print("=" * 70)


def main():
    """
    Main entry point for the CLI.

    Usage:
        python -m game.ui.cli [seed] [--auto]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Vegas Solitaire CLI")
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible games"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Play automatically using random strategy"
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=1000,
        help="Maximum moves for auto play (default: 1000)"
    )

    args = parser.parse_args()

    # Create game
    game = Game(seed=args.seed)
    cli = CLI(game)

    # Play
    if args.auto:
        cli.play_auto(strategy="random", max_moves=args.max_moves)
    else:
        cli.play_interactive()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick demo of Vegas Solitaire game.

This demonstrates the core functionality and shows a sample game.
"""

from game.core.game import Game, play_random_game
from game.ui.cli import CLI


def demo_basic_game():
    """Demo: Create and play a few moves."""
    print("=" * 70)
    print("DEMO 1: Basic Game Setup and Moves")
    print("=" * 70)
    print()

    # Create a game with a specific seed
    game = Game(seed=42)
    game.deal()

    # Show initial state
    cli = CLI(game)
    print(cli.render_board())

    # Get and display available moves
    moves = game.get_valid_moves()
    print(f"Available moves: {len(moves)}")
    print(cli.render_moves(moves[:5]))  # Show first 5
    print()

    # Make a few moves
    print("Making 3 moves...")
    for i in range(3):
        moves = game.get_valid_moves()
        if moves:
            game.make_move(moves[0])
            print(f"  Move {i+1}: {moves[0]}")

    print()
    print(f"After 3 moves:")
    print(f"  Score: ${game.get_score()}")
    print(f"  Move count: {game.get_move_count()}")
    print()


def demo_random_game():
    """Demo: Play a complete random game."""
    print("=" * 70)
    print("DEMO 2: Complete Random Game")
    print("=" * 70)
    print()

    game = play_random_game(seed=123, max_moves=100)
    summary = game.get_game_summary()

    print(f"Completed random game with seed 123:")
    print(f"  Final Score: ${summary['score']}")
    print(f"  Total Moves: {summary['moves']}")
    print(f"  Cards in Foundations: {summary['cards_in_foundations']}/52")
    print(f"  Win: {summary['is_winning']}")
    print(f"  Game Over: {summary['is_over']}")
    print()


def demo_statistics():
    """Demo: Run multiple games and show statistics."""
    print("=" * 70)
    print("DEMO 3: Multiple Game Statistics")
    print("=" * 70)
    print()

    print("Playing 10 random games...")
    scores = []
    wins = 0
    foundation_cards = []

    for seed in range(10):
        game = play_random_game(seed=seed, max_moves=200)
        summary = game.get_game_summary()

        scores.append(summary['score'])
        foundation_cards.append(summary['cards_in_foundations'])
        if summary['is_winning']:
            wins += 1

    print(f"\nResults from 10 games:")
    print(f"  Wins: {wins}/10 ({wins*10}%)")
    print(f"  Average Score: ${sum(scores)/len(scores):.2f}")
    print(f"  Best Score: ${max(scores)}")
    print(f"  Worst Score: ${min(scores)}")
    print(f"  Average Foundation Cards: {sum(foundation_cards)/len(foundation_cards):.1f}/52")
    print()


def demo_game_state():
    """Demo: Show game state details."""
    print("=" * 70)
    print("DEMO 4: Game State Details")
    print("=" * 70)
    print()

    game = Game(seed=999)
    game.deal()

    state = game.state

    print("Game state breakdown:")
    print(f"  Stock: {len(state.stock)} cards")
    print(f"  Waste: {len(state.waste)} cards")
    print(f"  Tableau columns: {[len(col) for col in state.tableau]}")
    print(f"  Hidden cards: {state.tableau_hidden}")
    print(f"  Foundation cards: {state.get_foundation_count()}")
    print(f"  Score: ${state.score}")
    print()

    # Show some tableau cards
    print("Sample tableau cards (column 6):")
    for i, card in enumerate(state.tableau[6]):
        if i < state.tableau_hidden[6]:
            print(f"  {i}: [HIDDEN]")
        else:
            print(f"  {i}: {card}")
    print()


if __name__ == "__main__":
    demo_basic_game()
    demo_random_game()
    demo_statistics()
    demo_game_state()

    print("=" * 70)
    print("All demos completed! ✓")
    print()
    print("Try playing interactively:")
    print("  python -m game.ui.cli")
    print()
    print("Or auto-play with a specific seed:")
    print("  python -m game.ui.cli --seed 42 --auto")
    print("=" * 70)

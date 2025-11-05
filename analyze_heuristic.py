"""Analyze heuristic solver behavior in detail."""

from collections import Counter
from game.core.game import Game
from game.core.moves import MoveType
from optimization.solvers.heuristic_solver import HeuristicSolver


def analyze_heuristic_moves():
    """Analyze what types of moves the heuristic solver makes."""
    print("=" * 60)
    print("Heuristic Solver Move Analysis")
    print("=" * 60)

    game = Game(seed=42)
    game.deal()

    solver = HeuristicSolver()

    moves_made = 0
    max_moves = 200  # Shorter test
    move_type_counts = Counter()
    last_10_moves = []
    foundation_progress = []

    while moves_made < max_moves:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            print(f"\nGame over (no valid moves) after {moves_made} moves")
            break

        move = solver.choose_move(game)
        if not move:
            print(f"\nSolver returned no move after {moves_made} moves")
            break

        if game.make_move(move):
            moves_made += 1
            move_type_counts[move.move_type] += 1
            last_10_moves.append(str(move))
            if len(last_10_moves) > 10:
                last_10_moves.pop(0)

            # Track foundation progress
            if moves_made % 10 == 0:
                foundation_progress.append((moves_made, game.state.foundation_count))
                print(f"  {moves_made:3d} moves: Score ${game.state.score:4d}, Foundation {game.state.foundation_count:2d}/52")
        else:
            print(f"\nInvalid move at {moves_made} moves")
            break

    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"Total moves: {moves_made}")
    print(f"Final score: ${game.state.score}")
    print(f"Foundation cards: {game.state.foundation_count}/52")

    print("\n" + "-" * 60)
    print("Move Type Distribution")
    print("-" * 60)
    for move_type, count in sorted(move_type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / moves_made * 100) if moves_made > 0 else 0
        print(f"  {move_type.name:25s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "-" * 60)
    print("Last 10 Moves")
    print("-" * 60)
    for i, move in enumerate(last_10_moves, 1):
        print(f"  {i}. {move}")

    print("\n" + "-" * 60)
    print("Foundation Progress")
    print("-" * 60)
    for moves, foundation_count in foundation_progress:
        print(f"  After {moves:3d} moves: {foundation_count:2d} cards")

    # Check for repetitive patterns
    if len(last_10_moves) >= 4:
        print("\n" + "-" * 60)
        print("Checking for repetitive patterns...")
        print("-" * 60)
        # Check if moves are cycling
        if last_10_moves[-1] == last_10_moves[-3] and last_10_moves[-2] == last_10_moves[-4]:
            print("  ⚠️  WARNING: Detected alternating pattern in last 4 moves!")
            print(f"    Move A: {last_10_moves[-1]}")
            print(f"    Move B: {last_10_moves[-2]}")


if __name__ == "__main__":
    analyze_heuristic_moves()

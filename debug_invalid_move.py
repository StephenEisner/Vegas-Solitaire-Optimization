"""Debug why heuristic solver hits an invalid move."""

from game.core.game import Game
from game.core.rules import is_valid_move
from optimization.solvers.heuristic_solver import HeuristicSolver


def debug_invalid_move():
    """Run the heuristic solver and catch the invalid move."""
    game = Game(seed=42)
    game.deal()

    solver = HeuristicSolver()

    moves_made = 0
    max_moves = 100

    while moves_made < max_moves:
        valid_moves = game.get_valid_moves()

        if not valid_moves:
            print(f"\n✅ Game ended naturally (no valid moves) after {moves_made} moves")
            break

        print(f"\nMove {moves_made + 1}:")
        print(f"  Valid moves available: {len(valid_moves)}")

        move = solver.choose_move(game)

        if not move:
            print(f"  ❌ Solver returned None")
            break

        print(f"  Solver chose: {move}")

        # Check if move is in valid moves
        if move not in valid_moves:
            print(f"  ⚠️  WARNING: Chosen move NOT in valid moves list!")
            print(f"  Valid moves:")
            for i, vm in enumerate(valid_moves[:10], 1):  # Show first 10
                print(f"    {i}. {vm}")
            if len(valid_moves) > 10:
                print(f"    ... and {len(valid_moves) - 10} more")

        # Check if move is valid according to rules
        if not is_valid_move(game.state, move):
            print(f"  ❌ Move failed validation!")
            print(f"  Current state:")
            print(f"    Foundation: {sum(len(cards) for cards in game.state.foundations.values())} cards")
            print(f"    Stock: {len(game.state.stock)} cards")
            print(f"    Waste: {len(game.state.waste)} cards")
            print(f"    Tableau columns: {[len(col) for col in game.state.tableau]}")
            break

        # Try to make the move
        if game.make_move(move):
            moves_made += 1
            print(f"  ✅ Move succeeded")
        else:
            print(f"  ❌ Move failed in game.make_move()")
            print(f"  This shouldn't happen if move was valid!")
            break

    print(f"\nFinal: {moves_made} moves, score: ${game.state.score}, foundation: {game.state.foundation_count}/52")


if __name__ == "__main__":
    debug_invalid_move()

"""Debug the recycle passes tracking."""

from game.core.game import Game
from game.core.moves import MoveType
from optimization.solvers.heuristic_solver import HeuristicSolver


def debug_recycle():
    """Track recycle counts."""
    game = Game(seed=42)
    game.deal()

    solver = HeuristicSolver()

    moves_made = 0
    recycle_count = 0
    max_moves = 100

    while moves_made < max_moves:
        valid_moves = game.get_valid_moves()

        if not valid_moves:
            print(f"\nGame ended naturally (no valid moves)")
            break

        move = solver.choose_move(game)

        if not move:
            print(f"Solver returned None")
            break

        if move.move_type == MoveType.RECYCLE:
            recycle_count += 1
            print(f"\nMove {moves_made + 1}: RECYCLE #{recycle_count}")
            print(f"  Before recycle:")
            print(f"    passes_through_deck = {game.state.passes_through_deck}")
            print(f"    stock = {len(game.state.stock)} cards")
            print(f"    waste = {len(game.state.waste)} cards")

        if game.make_move(move):
            moves_made += 1

            if move.move_type == MoveType.RECYCLE:
                print(f"  After recycle:")
                print(f"    passes_through_deck = {game.state.passes_through_deck}")
                print(f"    stock = {len(game.state.stock)} cards")
                print(f"    waste = {len(game.state.waste)} cards")
        else:
            print(f"\n❌ Move {moves_made + 1} failed: {move}")
            print(f"   passes_through_deck = {game.state.passes_through_deck}")
            print(f"   stock = {len(game.state.stock)} cards")
            print(f"   waste = {len(game.state.waste)} cards")

            # Check if this is a recycle attempt
            if move.move_type == MoveType.RECYCLE:
                print(f"\n⚠️  Recycle validation should have caught this!")
                print(f"   Validation check: stock==0 ({len(game.state.stock)==0}) and waste>0 ({len(game.state.waste)>0}) and passes<3 ({game.state.passes_through_deck<3})")
            break

    print(f"\nFinal: {moves_made} moves, {recycle_count} recycles, passes_through_deck = {game.state.passes_through_deck}")


if __name__ == "__main__":
    debug_recycle()

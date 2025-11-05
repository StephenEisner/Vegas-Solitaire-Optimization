"""Quick test script to verify solvers are working correctly."""

from game.core.game import Game
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.solvers.mcts_solver import MCTSSolver

def test_heuristic_no_cycling():
    """Test that heuristic solver doesn't get stuck in cycles."""
    print("Testing Heuristic Solver...")
    game = Game(seed=42)
    game.deal()

    solver = HeuristicSolver()
    moves_made = 0
    max_moves = 500

    while moves_made < max_moves:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            print(f"  Game over (no valid moves) after {moves_made} moves")
            break

        move = solver.choose_move(game)
        if not move:
            print(f"  Solver returned no move after {moves_made} moves")
            break

        if game.make_move(move):
            moves_made += 1
            if moves_made % 50 == 0:
                print(f"  Made {moves_made} moves, score: ${game.state.score}")
        else:
            print(f"  Invalid move at {moves_made} moves")
            break

    print(f"  Final: {moves_made} moves, score: ${game.state.score}, foundation: {game.state.foundation_count}/52")
    return moves_made


def test_mcts():
    """Test that MCTS solver makes moves."""
    print("\nTesting MCTS Solver...")
    game = Game(seed=42)
    game.deal()

    solver = MCTSSolver(simulations_per_move=100, seed=42)
    moves_made = 0
    max_moves = 50  # Just test a few moves

    while moves_made < max_moves:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            print(f"  Game over (no valid moves) after {moves_made} moves")
            break

        print(f"  Move {moves_made + 1}: Searching...")
        move = solver.choose_move(game)

        if not move:
            print(f"  Solver returned None after {moves_made} moves")
            print(f"  Valid moves available: {len(valid_moves)}")
            if valid_moves:
                print(f"  First valid move: {valid_moves[0]}")
            break

        print(f"  Chose: {move}")

        if game.make_move(move):
            moves_made += 1
            print(f"  Made move {moves_made}, score: ${game.state.score}")
        else:
            print(f"  Invalid move at {moves_made} moves")
            break

    print(f"  Final: {moves_made} moves, score: ${game.state.score}, foundation: {game.state.foundation_count}/52")
    return moves_made


if __name__ == "__main__":
    print("=" * 60)
    print("Solver Verification Tests")
    print("=" * 60)

    heuristic_moves = test_heuristic_no_cycling()
    mcts_moves = test_mcts()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Heuristic: {heuristic_moves} moves")
    print(f"  MCTS: {mcts_moves} moves")
    print("=" * 60)

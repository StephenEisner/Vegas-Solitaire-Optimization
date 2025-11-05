"""Test to reproduce the watch mode MCTS issue."""

from game.core.game import Game
from optimization.solvers.mcts_solver import MCTSSolver

def create_solver(solver_name):
    """Create a solver instance based on name (same as watch mode)."""
    if solver_name == "MCTS (100 sims)":
        return MCTSSolver(simulations_per_move=100, seed=None)
    else:
        return MCTSSolver(simulations_per_move=1000, seed=None)


def make_solver_move(game, solver_name):
    """
    Make a single move with the selected solver (same as watch mode).

    Returns:
        bool: True if move was made, False if game is over
    """
    valid_moves = game.get_valid_moves()

    if not valid_moves:
        print("No valid moves available")
        return False

    print(f"Valid moves: {len(valid_moves)}")

    solver = create_solver(solver_name)
    print(f"Created solver: {solver}")

    move = solver.choose_move(game)

    if not move:
        print("Solver returned None")
        return False

    print(f"Solver chose: {move}")

    success = game.make_move(move)
    print(f"Move success: {success}")

    return success


def test_watch_mode_pattern():
    """Test the exact pattern used in watch mode."""
    print("=" * 60)
    print("Testing Watch Mode Pattern with MCTS")
    print("=" * 60)

    # Initialize game (same as watch mode)
    seed = 42
    game = Game(seed=seed)
    game.deal()

    solver_name = "MCTS (100 sims)"

    # Try to make a few moves (same as watch mode)
    for i in range(5):
        print(f"\n--- Move {i+1} ---")
        success = make_solver_move(game, solver_name)

        if not success:
            print("Failed to make move!")
            break

        print(f"Score: ${game.state.score}, Foundation: {game.state.foundation_count}/52")


if __name__ == "__main__":
    test_watch_mode_pattern()

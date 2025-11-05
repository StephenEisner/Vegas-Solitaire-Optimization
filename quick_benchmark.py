"""Quick benchmark to compare solvers after improvements."""

from optimization.solvers.random_solver import RandomSolver
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.solvers.mcts_solver import MCTSSolver
from optimization.evaluation.benchmark import Benchmark


def main():
    print("=" * 70)
    print("Quick Benchmark - Comparing Improved Solvers")
    print("=" * 70)

    # Create solvers
    solvers = [
        RandomSolver(seed=42),
        HeuristicSolver(),
        MCTSSolver(simulations_per_move=100, seed=42),
    ]

    # Run small benchmark (10 games)
    num_games = 10
    start_seed = 0
    max_moves = 500

    print(f"\nSettings:")
    print(f"  Games: {num_games}")
    print(f"  Start seed: {start_seed}")
    print(f"  Max moves: {max_moves}")
    print()

    benchmark = Benchmark(test_set_name="quick_test")

    results = []
    for solver in solvers:
        print(f"\nTesting {solver.__class__.__name__}...")
        print("-" * 70)

        result = benchmark.evaluate_solver(
            solver=solver,
            num_games=num_games,
            start_seed=start_seed,
            max_moves=max_moves,
            verbose=True
        )

        results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for result in results:
        print(f"\n{result.solver_name}:")
        print(f"  Win Rate:       {result.win_rate:6.1%}")
        print(f"  Avg Score:      ${result.average_score:6.0f}")
        print(f"  Avg Foundation: {result.average_foundation_count:5.1f} / 52")
        print(f"  Avg Moves:      {result.average_moves:6.0f}")
        print(f"  Avg Time:       {result.average_time_ms:6.0f} ms")


if __name__ == "__main__":
    main()

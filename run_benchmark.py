#!/usr/bin/env python3
"""
Benchmark runner for Vegas Solitaire solvers.

This script runs benchmarks on all available solvers and displays results.
"""

import argparse
from optimization.solvers.random_solver import RandomSolver
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.evaluation.benchmark import Benchmark


def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark Vegas Solitaire Solvers")
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games per solver (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting seed (default: 0)"
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=1000,
        help="Maximum moves per game (default: 1000)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON files"
    )

    args = parser.parse_args()

    # Create solvers
    solvers = [
        RandomSolver(seed=42),
        HeuristicSolver(),
    ]

    # Create benchmark
    benchmark = Benchmark(test_set_name=f"standard_{args.games}_games")

    # Run comparison
    results = benchmark.compare_solvers(
        solvers=solvers,
        num_games=args.games,
        start_seed=args.seed,
        max_moves=args.max_moves,
        verbose=True
    )

    # Save if requested
    if args.save:
        benchmark.save_results()
        print("\nResults saved to benchmark_results/")


if __name__ == "__main__":
    main()

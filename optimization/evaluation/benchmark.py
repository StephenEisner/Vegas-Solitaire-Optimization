"""
Benchmark framework for comparing solvers.

This module provides tools for systematically evaluating and comparing
different solving strategies.
"""

from typing import List, Dict, Any, Optional
import json
import time
from pathlib import Path
from optimization.solvers.base import Solver, SolverStatistics


class BenchmarkResults:
    """
    Results from a benchmark run.

    Stores detailed results for one or more solvers on a test set.
    """

    def __init__(self, solver_name: str, test_set_name: str):
        """
        Initialize benchmark results.

        Args:
            solver_name: Name of the solver
            test_set_name: Name of the test set used
        """
        self.solver_name = solver_name
        self.test_set_name = test_set_name
        self.statistics: Optional[SolverStatistics] = None
        self.timestamp = time.time()

    def set_statistics(self, stats: SolverStatistics) -> None:
        """Set the statistics for this benchmark."""
        self.statistics = stats

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to dictionary.

        Returns:
            Dictionary with all results
        """
        result = {
            'solver_name': self.solver_name,
            'test_set': self.test_set_name,
            'timestamp': self.timestamp,
        }

        if self.statistics:
            result['statistics'] = self.statistics.to_dict()

        return result

    def to_json(self, filepath: str) -> None:
        """
        Save results to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'BenchmarkResults':
        """
        Load results from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            BenchmarkResults object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        results = cls(data['solver_name'], data['test_set'])
        results.timestamp = data['timestamp']

        # Reconstruct statistics if present
        if 'statistics' in data:
            stats = SolverStatistics()
            stats.games_played = data['statistics']['games_played']
            stats.games_won = data['statistics']['games_won']
            stats.total_score = data['statistics']['average_score'] * stats.games_played
            stats.total_moves = data['statistics']['average_moves'] * stats.games_played
            stats.total_time = data['statistics']['total_time']
            results.statistics = stats

        return results


class Benchmark:
    """
    Benchmark framework for evaluating solvers.

    Provides methods to run standardized tests on solvers and
    compare their performance.
    """

    def __init__(self, test_set_name: str = "default"):
        """
        Initialize benchmark.

        Args:
            test_set_name: Name for this test set
        """
        self.test_set_name = test_set_name
        self.results: Dict[str, BenchmarkResults] = {}

    def evaluate_solver(self, solver: Solver,
                       num_games: int = 100,
                       start_seed: int = 0,
                       max_moves: int = 1000,
                       verbose: bool = False) -> BenchmarkResults:
        """
        Evaluate a solver on a test set.

        Args:
            solver: Solver to evaluate
            num_games: Number of games to play
            start_seed: Starting seed for games
            max_moves: Maximum moves per game
            verbose: Whether to print progress

        Returns:
            BenchmarkResults with evaluation results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Benchmarking: {solver.name}")
            print(f"Test Set: {self.test_set_name}")
            print(f"Games: {num_games}")
            print(f"{'='*70}\n")

        # Reset solver statistics
        solver.reset_statistics()

        # Play games
        start_time = time.time()
        solver.play_multiple_games(
            num_games=num_games,
            start_seed=start_seed,
            max_moves=max_moves,
            verbose=verbose
        )
        total_time = time.time() - start_time

        if verbose:
            print(f"\nBenchmark completed in {total_time:.2f}s")
            print(f"Average time per game: {total_time/num_games:.3f}s\n")

        # Store results
        results = BenchmarkResults(solver.name, self.test_set_name)
        results.set_statistics(solver.get_statistics())

        self.results[solver.name] = results

        return results

    def compare_solvers(self, solvers: List[Solver],
                       num_games: int = 100,
                       start_seed: int = 0,
                       max_moves: int = 1000,
                       verbose: bool = False) -> Dict[str, BenchmarkResults]:
        """
        Compare multiple solvers on the same test set.

        Args:
            solvers: List of solvers to compare
            num_games: Number of games per solver
            start_seed: Starting seed (same for all solvers)
            max_moves: Maximum moves per game
            verbose: Whether to print progress

        Returns:
            Dictionary mapping solver names to results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"SOLVER COMPARISON")
            print(f"Test Set: {self.test_set_name}")
            print(f"Solvers: {len(solvers)}")
            print(f"Games per solver: {num_games}")
            print(f"{'='*70}\n")

        results = {}

        for i, solver in enumerate(solvers, 1):
            if verbose:
                print(f"\n[{i}/{len(solvers)}] Evaluating {solver.name}...")

            result = self.evaluate_solver(
                solver,
                num_games=num_games,
                start_seed=start_seed,
                max_moves=max_moves,
                verbose=False
            )

            results[solver.name] = result

            if verbose:
                print(f"\n{solver.name} Results:")
                print(f"  Win Rate: {result.statistics.get_win_rate():.1f}%")
                print(f"  Avg Score: ${result.statistics.get_average_score():.2f}")
                print(f"  Avg Moves: {result.statistics.get_average_moves():.1f}")

        if verbose:
            self.print_comparison_table(results)

        return results

    def print_comparison_table(self, results: Optional[Dict[str, BenchmarkResults]] = None) -> None:
        """
        Print a formatted comparison table.

        Args:
            results: Results to display (uses self.results if None)
        """
        if results is None:
            results = self.results

        if not results:
            print("No results to display")
            return

        print(f"\n{'='*70}")
        print(f"COMPARISON TABLE - {self.test_set_name}")
        print(f"{'='*70}")

        # Header
        print(f"{'Solver':<15} {'Win%':>8} {'Avg Score':>12} {'Avg Moves':>11} {'Avg Time':>11}")
        print("-" * 70)

        # Sort by win rate (descending)
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].statistics.get_win_rate() if x[1].statistics else 0,
            reverse=True
        )

        # Rows
        for name, result in sorted_results:
            if result.statistics:
                stats = result.statistics
                print(f"{name:<15} {stats.get_win_rate():>7.1f}% "
                      f"${stats.get_average_score():>10.2f} "
                      f"{stats.get_average_moves():>10.1f} "
                      f"{stats.get_average_time():>10.3f}s")

        print("=" * 70 + "\n")

    def save_results(self, directory: str = "benchmark_results") -> None:
        """
        Save all benchmark results to JSON files.

        Args:
            directory: Directory to save results
        """
        Path(directory).mkdir(parents=True, exist_ok=True)

        for name, result in self.results.items():
            filename = f"{directory}/{name.lower().replace(' ', '_')}_results.json"
            result.to_json(filename)
            print(f"Saved {name} results to {filename}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all benchmark results.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'test_set': self.test_set_name,
            'num_solvers': len(self.results),
            'solvers': {}
        }

        for name, result in self.results.items():
            if result.statistics:
                summary['solvers'][name] = result.statistics.to_dict()

        return summary

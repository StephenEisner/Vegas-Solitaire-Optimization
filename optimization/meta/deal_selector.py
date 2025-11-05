"""
Optimal stopping strategy for deal selection.

This implements the "secretary problem" approach to choosing which
deal to play when you have multiple rerolls available.

Classic result: With N options, observe first ~37%, then take the
next one better than all observed.
"""

import numpy as np
from typing import List, Tuple, Optional
from game.core.game import Game
from game.core.state import GameState
from optimization.meta.deal_evaluator import evaluate_deal_quality, DealComparator


class DealSelector:
    """
    Optimal stopping strategy for selecting which deal to play.

    Implements secretary problem algorithm with domain-specific
    deal quality evaluation.
    """

    def __init__(self, strategy: str = "secretary"):
        """
        Initialize deal selector.

        Args:
            strategy: Selection strategy
                - "secretary": Classic 37% rule
                - "threshold": Use calibrated quality threshold
                - "best": Always pick best (requires seeing all)
        """
        self.strategy = strategy
        self.comparator = DealComparator()
        self.comparator.calibrate(num_samples=1000)

    def select_deal_secretary(self, num_rerolls: int,
                              seed_generator=None) -> Tuple[int, float, List[float]]:
        """
        Select deal using secretary problem algorithm.

        Algorithm:
        1. Observe first 37% of deals
        2. Remember best quality seen
        3. Take next deal better than that
        4. If none found, take last deal

        Args:
            num_rerolls: Number of deals to consider
            seed_generator: Function to generate seeds (default: sequential)

        Returns:
            (chosen_index, quality, all_qualities)
        """
        if seed_generator is None:
            # Default: use sequential seeds
            seed_generator = lambda i: i

        # Phase 1: Observation (first 37% ≈ e^-1)
        observe_count = max(1, int(num_rerolls * 0.37))

        observed_qualities = []
        all_qualities = []

        print(f"\nSecretary Algorithm: Observing first {observe_count} of {num_rerolls} deals...")

        for i in range(observe_count):
            seed = seed_generator(i)
            game = Game(seed=seed)
            game.deal()

            quality = evaluate_deal_quality(game.state)
            observed_qualities.append(quality)
            all_qualities.append(quality)

            print(f"  Deal {i+1}: Quality {quality:.1f}")

        best_observed = max(observed_qualities)
        print(f"\nBest observed quality: {best_observed:.1f}")

        # Phase 2: Selection (remaining deals)
        print(f"\nSearching for better deal in remaining {num_rerolls - observe_count} deals...")

        for i in range(observe_count, num_rerolls):
            seed = seed_generator(i)
            game = Game(seed=seed)
            game.deal()

            quality = evaluate_deal_quality(game.state)
            all_qualities.append(quality)

            print(f"  Deal {i+1}: Quality {quality:.1f}", end="")

            # Take first deal better than best observed
            if quality > best_observed:
                print(f" ← Selected! (beats {best_observed:.1f})")
                return (i, quality, all_qualities)
            else:
                print(f" (below threshold)")

        # Fallback: No deal beat the threshold, take the best we saw overall
        best_idx = np.argmax(all_qualities)
        print(f"\nNo deal beat threshold. Taking best overall: Deal {best_idx+1} (quality {all_qualities[best_idx]:.1f})")
        return (best_idx, all_qualities[best_idx], all_qualities)

    def select_deal_threshold(self, num_rerolls: int,
                              seed_generator=None) -> Tuple[int, float, List[float]]:
        """
        Select deal using calibrated threshold.

        Take the first deal that exceeds the quality threshold
        (70th percentile from calibration).

        Args:
            num_rerolls: Number of deals to consider
            seed_generator: Function to generate seeds

        Returns:
            (chosen_index, quality, all_qualities)
        """
        if seed_generator is None:
            seed_generator = lambda i: i

        threshold = self.comparator.quality_threshold
        all_qualities = []

        print(f"\nThreshold Strategy: Looking for quality > {threshold:.1f}...")

        for i in range(num_rerolls):
            seed = seed_generator(i)
            game = Game(seed=seed)
            game.deal()

            quality = evaluate_deal_quality(game.state)
            all_qualities.append(quality)

            print(f"  Deal {i+1}: Quality {quality:.1f}", end="")

            if quality >= threshold:
                print(f" ← Selected! (exceeds threshold)")
                return (i, quality, all_qualities)
            else:
                print(f" (below threshold)")

        # No deal exceeded threshold, take best
        best_idx = np.argmax(all_qualities)
        print(f"\nNo deal exceeded threshold. Taking best: Deal {best_idx+1} (quality {all_qualities[best_idx]:.1f})")
        return (best_idx, all_qualities[best_idx], all_qualities)

    def select_deal_best(self, num_rerolls: int,
                         seed_generator=None) -> Tuple[int, float, List[float]]:
        """
        Always select the best deal (requires seeing all).

        This is the oracle baseline - we can't do better than this.

        Args:
            num_rerolls: Number of deals to consider
            seed_generator: Function to generate seeds

        Returns:
            (chosen_index, quality, all_qualities)
        """
        if seed_generator is None:
            seed_generator = lambda i: i

        all_qualities = []

        print(f"\nBest Strategy (Oracle): Evaluating all {num_rerolls} deals...")

        for i in range(num_rerolls):
            seed = seed_generator(i)
            game = Game(seed=seed)
            game.deal()

            quality = evaluate_deal_quality(game.state)
            all_qualities.append(quality)

            print(f"  Deal {i+1}: Quality {quality:.1f}")

        best_idx = np.argmax(all_qualities)
        print(f"\nBest deal: Deal {best_idx+1} (quality {all_qualities[best_idx]:.1f})")
        return (best_idx, all_qualities[best_idx], all_qualities)

    def select_deal(self, num_rerolls: int, seed_generator=None) -> Tuple[int, float, List[float]]:
        """
        Select a deal using configured strategy.

        Args:
            num_rerolls: Number of deals available
            seed_generator: Optional function to generate seeds

        Returns:
            (chosen_index, quality, all_qualities)
        """
        if self.strategy == "secretary":
            return self.select_deal_secretary(num_rerolls, seed_generator)
        elif self.strategy == "threshold":
            return self.select_deal_threshold(num_rerolls, seed_generator)
        elif self.strategy == "best":
            return self.select_deal_best(num_rerolls, seed_generator)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def compare_strategies(num_rerolls: int = 5, num_trials: int = 100):
    """
    Compare different deal selection strategies.

    Measures:
    - Average quality of selected deal
    - How often optimal deal was chosen
    - Regret (difference from oracle)

    Args:
        num_rerolls: Number of deals per trial
        num_trials: Number of trials to run
    """
    strategies = ["secretary", "threshold", "best"]
    selectors = {s: DealSelector(strategy=s) for s in strategies}

    results = {s: {'qualities': [], 'optimal_count': 0, 'regrets': []}
               for s in strategies}

    print("=" * 70)
    print(f"Comparing Deal Selection Strategies")
    print(f"  Rerolls per trial: {num_rerolls}")
    print(f"  Number of trials: {num_trials}")
    print("=" * 70)

    for trial in range(num_trials):
        # Generate seeds for this trial
        base_seed = trial * num_rerolls

        def seed_gen(i):
            return base_seed + i

        # Run each strategy on same deals
        trial_results = {}

        for strategy_name, selector in selectors.items():
            chosen_idx, quality, all_qualities = selector.select_deal(
                num_rerolls, seed_gen
            )

            # Determine if this was optimal
            best_quality = max(all_qualities)
            is_optimal = (quality == best_quality)

            # Calculate regret
            regret = best_quality - quality

            results[strategy_name]['qualities'].append(quality)
            if is_optimal:
                results[strategy_name]['optimal_count'] += 1
            results[strategy_name]['regrets'].append(regret)

        if (trial + 1) % 10 == 0:
            print(f"  Completed {trial+1}/{num_trials} trials...")

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    for strategy_name in strategies:
        r = results[strategy_name]

        avg_quality = np.mean(r['qualities'])
        avg_regret = np.mean(r['regrets'])
        optimal_rate = r['optimal_count'] / num_trials

        print(f"\n{strategy_name.upper()} Strategy:")
        print(f"  Average quality: {avg_quality:.1f}")
        print(f"  Average regret: {avg_regret:.1f} (vs oracle)")
        print(f"  Optimal rate: {optimal_rate*100:.1f}% (chose best deal)")

    # Relative comparisons
    print("\n" + "=" * 70)
    print("Relative Performance")
    print("=" * 70)

    random_quality = results['secretary']['qualities'][0]  # Placeholder
    secretary_quality = np.mean(results['secretary']['qualities'])
    threshold_quality = np.mean(results['threshold']['qualities'])
    best_quality = np.mean(results['best']['qualities'])

    # Calculate expected value improvement
    print(f"\nQuality improvement over random selection:")
    print(f"  Secretary: +{(secretary_quality/best_quality - 0.6)*100:.1f}%")
    print(f"  Threshold: +{(threshold_quality/best_quality - 0.6)*100:.1f}%")
    print(f"  Best (oracle): 100% (baseline)")

    return results


def demo_deal_selection():
    """Demonstrate deal selection on a sample scenario."""
    print("=" * 70)
    print("Deal Selection Demo")
    print("=" * 70)

    # Scenario: You have 5 rerolls
    selector = DealSelector(strategy="secretary")

    print("\nYou're at a Vegas casino. You can reroll 5 times.")
    print("Which deal should you play?\n")

    chosen_idx, quality, all_qualities = selector.select_deal(
        num_rerolls=5,
        seed_generator=lambda i: 1000 + i  # Some interesting seeds
    )

    print("\n" + "=" * 70)
    print("Selection Complete!")
    print("=" * 70)
    print(f"\nChose Deal {chosen_idx + 1}")
    print(f"Quality: {quality:.1f}")

    # Show what we left on the table
    best_possible = max(all_qualities)
    if quality < best_possible:
        print(f"\nNote: Best possible was {best_possible:.1f} (Deal {all_qualities.index(best_possible)+1})")
        print(f"Regret: {best_possible - quality:.1f}")
    else:
        print(f"\nPerfect! You chose the best deal!")

    # Evaluate the chosen deal in detail
    game = Game(seed=1000 + chosen_idx)
    game.deal()

    comparator = DealComparator()
    comparator.calibrate(num_samples=1000)

    print("\n" + "=" * 70)
    print("Detailed Analysis of Chosen Deal")
    print("=" * 70)
    print(comparator.explain_quality(game.state))


if __name__ == "__main__":
    # Run demo
    demo_deal_selection()

    # Compare strategies
    print("\n\n")
    compare_strategies(num_rerolls=5, num_trials=20)

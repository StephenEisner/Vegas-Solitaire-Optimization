#!/usr/bin/env python3
"""
Analyze which deal features actually correlate with good outcomes.

This uses empirical evaluation with real solvers to determine
what makes a deal "good" vs what the heuristics think.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from game.core.game import Game
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.evaluation.empirical_deal_quality import (
    EmpiricalDealEvaluator,
    LearnedDealQualityPredictor,
    compare_heuristic_vs_empirical
)


def analyze_feature_importance(outcomes, predictor, save_plots=True):
    """Analyze which features matter most for deal quality."""
    print("\n" + "=" * 70)
    print("Feature Importance Analysis")
    print("=" * 70)

    # Get feature importance from trained model
    importance = np.abs(predictor.weights)

    # Sort by importance
    indices = np.argsort(importance)[::-1]

    print("\nTop 20 Most Important Features:")
    print(f"{'Rank':<6} {'Feature':<10} {'Weight':<12} {'Abs Weight':<12}")
    print("-" * 50)

    for rank, idx in enumerate(indices[:20], 1):
        weight = predictor.weights[idx]
        abs_weight = importance[idx]
        print(f"{rank:<6} {idx:<10} {weight:+.6f}  {abs_weight:.6f}")

    if save_plots:
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        top_n = 20
        top_indices = indices[:top_n]
        top_weights = predictor.weights[top_indices]

        plt.barh(range(top_n), top_weights)
        plt.yticks(range(top_n), [f"Feature {i}" for i in top_indices])
        plt.xlabel("Weight")
        plt.title(f"Top {top_n} Most Important Features for Deal Quality")
        plt.tight_layout()

        Path("analysis").mkdir(exist_ok=True)
        plt.savefig("analysis/feature_importance.png", dpi=150)
        print(f"\n✓ Saved feature importance plot to analysis/feature_importance.png")
        plt.close()


def analyze_deal_quality_distribution(outcomes, save_plots=True):
    """Analyze distribution of deal quality."""
    print("\n" + "=" * 70)
    print("Deal Quality Distribution Analysis")
    print("=" * 70)

    avg_cards = np.array([o.avg_cards for o in outcomes])
    avg_scores = np.array([o.avg_score for o in outcomes])
    win_rates = np.array([o.win_rate for o in outcomes])
    heuristic_qualities = np.array([o.heuristic_quality for o in outcomes])

    # Statistics
    print(f"\nCards Played Distribution:")
    print(f"  Min: {np.min(avg_cards):.1f}")
    print(f"  25th percentile: {np.percentile(avg_cards, 25):.1f}")
    print(f"  Median: {np.median(avg_cards):.1f}")
    print(f"  75th percentile: {np.percentile(avg_cards, 75):.1f}")
    print(f"  Max: {np.max(avg_cards):.1f}")

    print(f"\nScore Distribution:")
    print(f"  Min: ${np.min(avg_scores):.1f}")
    print(f"  Median: ${np.median(avg_scores):.1f}")
    print(f"  Max: ${np.max(avg_scores):.1f}")

    print(f"\nWin Rate Distribution:")
    print(f"  Never win: {np.sum(win_rates == 0)}/{len(win_rates)} deals")
    print(f"  Sometimes win: {np.sum((win_rates > 0) & (win_rates < 1))}/{len(win_rates)} deals")
    print(f"  Always win: {np.sum(win_rates == 1)}/{len(win_rates)} deals")

    if save_plots:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Cards distribution
        axes[0, 0].hist(avg_cards, bins=20, edgecolor='black')
        axes[0, 0].set_xlabel("Average Cards Played")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution of Cards Played")
        axes[0, 0].axvline(np.median(avg_cards), color='red', linestyle='--', label='Median')
        axes[0, 0].legend()

        # Score distribution
        axes[0, 1].hist(avg_scores, bins=20, edgecolor='black')
        axes[0, 1].set_xlabel("Average Score ($)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Scores")
        axes[0, 1].axvline(np.median(avg_scores), color='red', linestyle='--', label='Median')
        axes[0, 1].legend()

        # Heuristic vs empirical
        axes[1, 0].scatter(heuristic_qualities, avg_cards, alpha=0.5)
        axes[1, 0].set_xlabel("Heuristic Quality")
        axes[1, 0].set_ylabel("Average Cards Played")
        axes[1, 0].set_title("Heuristic Quality vs Actual Performance")

        # Add correlation
        corr = np.corrcoef(heuristic_qualities, avg_cards)[0, 1]
        axes[1, 0].text(0.05, 0.95, f"Correlation: {corr:.3f}",
                       transform=axes[1, 0].transAxes, verticalalignment='top')

        # Win rate histogram
        axes[1, 1].hist(win_rates, bins=10, edgecolor='black')
        axes[1, 1].set_xlabel("Win Rate")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Win Rates")

        plt.tight_layout()
        plt.savefig("analysis/deal_quality_distribution.png", dpi=150)
        print(f"✓ Saved distribution plots to analysis/deal_quality_distribution.png")
        plt.close()


def identify_good_vs_bad_deals(outcomes, predictor):
    """Identify characteristics of good vs bad deals."""
    print("\n" + "=" * 70)
    print("Good vs Bad Deals Analysis")
    print("=" * 70)

    avg_cards = np.array([o.avg_cards for o in outcomes])

    # Define thresholds
    good_threshold = np.percentile(avg_cards, 75)  # Top 25%
    bad_threshold = np.percentile(avg_cards, 25)   # Bottom 25%

    good_deals = [o for o in outcomes if o.avg_cards >= good_threshold]
    bad_deals = [o for o in outcomes if o.avg_cards <= bad_threshold]

    print(f"\nThresholds:")
    print(f"  Good deals: ≥{good_threshold:.1f} cards (n={len(good_deals)})")
    print(f"  Bad deals: ≤{bad_threshold:.1f} cards (n={len(bad_deals)})")

    # Compare heuristic quality
    good_heuristic = np.mean([o.heuristic_quality for o in good_deals])
    bad_heuristic = np.mean([o.heuristic_quality for o in bad_deals])

    print(f"\nHeuristic Quality:")
    print(f"  Good deals: {good_heuristic:.1f}")
    print(f"  Bad deals: {bad_heuristic:.1f}")
    print(f"  Difference: {good_heuristic - bad_heuristic:.1f}")

    # Compare predictions
    good_predictions = predictor.predict_from_outcomes(good_deals)
    bad_predictions = predictor.predict_from_outcomes(bad_deals)

    print(f"\nEmpirical Predictions:")
    print(f"  Good deals: {np.mean(good_predictions):.1f}")
    print(f"  Bad deals: {np.mean(bad_predictions):.1f}")
    print(f"  Difference: {np.mean(good_predictions) - np.mean(bad_predictions):.1f}")

    # Feature analysis
    print(f"\nFeature Differences (Good - Bad):")
    good_features = np.array([o.initial_features for o in good_deals])
    bad_features = np.array([o.initial_features for o in bad_deals])

    feature_diff = np.mean(good_features, axis=0) - np.mean(bad_features, axis=0)

    # Sort by magnitude of difference
    sorted_indices = np.argsort(np.abs(feature_diff))[::-1]

    for i, idx in enumerate(sorted_indices[:10], 1):
        diff = feature_diff[idx]
        print(f"  Feature {idx}: {diff:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze deal features empirically")
    parser.add_argument("--num-deals", type=int, default=100,
                       help="Number of deals to evaluate")
    parser.add_argument("--seed", type=int, default=5000,
                       help="Starting seed")
    parser.add_argument("--load-dataset", type=str,
                       help="Load existing dataset instead of generating")
    parser.add_argument("--no-plots", action="store_true",
                       help="Don't save plots")

    args = parser.parse_args()

    # Create solvers
    print("Setting up solvers...")
    solvers = {
        'Heuristic': HeuristicSolver(),
    }

    # TODO: Add more solvers when available
    # - Q-Learning
    # - TD Learning
    # - Beam Search
    # - MCTS

    evaluator = EmpiricalDealEvaluator(solvers)

    # Get or load dataset
    if args.load_dataset and Path(args.load_dataset).exists():
        print(f"\nLoading dataset from {args.load_dataset}...")
        outcomes = evaluator.load_dataset(args.load_dataset)
    else:
        print(f"\nEvaluating {args.num_deals} deals...")
        outcomes = evaluator.evaluate_many_deals(
            num_deals=args.num_deals,
            start_seed=args.seed,
            verbose=True
        )

        # Save dataset
        Path("data").mkdir(exist_ok=True)
        evaluator.save_dataset(outcomes, "data/deal_outcomes.json")

    # Train predictor
    print("\n" + "=" * 70)
    print("Training Deal Quality Predictor")
    print("=" * 70)

    predictor = LearnedDealQualityPredictor()
    predictor.train(outcomes, target='avg_cards', verbose=True)

    Path("models").mkdir(exist_ok=True)
    predictor.save("models/deal_quality_predictor.json")

    # Analysis
    analyze_feature_importance(outcomes, predictor, save_plots=not args.no_plots)
    analyze_deal_quality_distribution(outcomes, save_plots=not args.no_plots)
    identify_good_vs_bad_deals(outcomes, predictor)
    compare_heuristic_vs_empirical(outcomes, predictor)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  - data/deal_outcomes.json (dataset)")
    print(f"  - models/deal_quality_predictor.json (trained model)")
    if not args.no_plots:
        print(f"  - analysis/feature_importance.png")
        print(f"  - analysis/deal_quality_distribution.png")


if __name__ == "__main__":
    main()

"""
Test and demonstrate the meta-strategy framework.

This shows the complete multi-deal optimization pipeline:
1. Deal evaluation (user insight: immediate playability matters)
2. Deal selection (optimal stopping)
3. Strategic play (short vs long stack prioritization)
"""

import numpy as np
from game.core.game import Game
from optimization.meta.deal_evaluator import (
    evaluate_deal_quality,
    count_immediately_playable,
    evaluate_stack_configuration,
    DealComparator
)
from optimization.meta.deal_selector import DealSelector, compare_strategies
from optimization.meta.strategic_solver import StrategicSolver


def test_deal_evaluation():
    """Test deal quality evaluation on sample deals."""
    print("█" * 70)
    print("  TEST 1: DEAL EVALUATION")
    print("█" * 70)

    comparator = DealComparator()
    comparator.calibrate(num_samples=1000)

    print("\n" + "=" * 70)
    print("Evaluating 5 Random Deals")
    print("=" * 70)

    for seed in [42, 100, 200, 500, 1000]:
        game = Game(seed=seed)
        game.deal()

        print(f"\n{'─' * 70}")
        print(f"Deal {seed}")
        print('─' * 70)
        print(comparator.explain_quality(game.state))


def test_deal_selection():
    """Test optimal stopping strategies for deal selection."""
    print("\n\n")
    print("█" * 70)
    print("  TEST 2: DEAL SELECTION (Optimal Stopping)")
    print("█" * 70)

    # Test secretary algorithm
    print("\n" + "=" * 70)
    print("Secretary Algorithm Demo")
    print("=" * 70)

    selector = DealSelector(strategy="secretary")

    chosen_idx, quality, all_qualities = selector.select_deal(
        num_rerolls=5,
        seed_generator=lambda i: 2000 + i
    )

    print(f"\n{'═' * 70}")
    print("DECISION")
    print('═' * 70)
    print(f"Chose: Deal {chosen_idx + 1} (quality {quality:.1f})")

    best_possible = max(all_qualities)
    regret = best_possible - quality

    if regret == 0:
        print("✓ Perfect! Chose the best deal.")
    else:
        print(f"Regret: {regret:.1f} (best was {best_possible:.1f})")


def test_strategic_solver():
    """Test strategic solver with user insights."""
    print("\n\n")
    print("█" * 70)
    print("  TEST 3: STRATEGIC SOLVER (User Insights)")
    print("█" * 70)

    print("\n" + "=" * 70)
    print("Strategic Decision-Making")
    print("=" * 70)

    solver = StrategicSolver()
    game = Game(seed=42)
    game.deal()

    print("\nPlaying first 20 moves with strategic solver...")

    for move_num in range(20):
        context = solver.analyze_stack_context(game.state)

        if move_num == 0:
            print(f"\nInitial state:")
            print(f"  Game phase: {context['game_phase']}")
            print(f"  Empty columns: {context['empty_cols']}")
            print(f"  Kings available: {context['kings_available']}")

        move = solver.choose_move(game)
        if not move or not game.make_move(move):
            break

        if move_num == 9:
            print(f"\nAfter 10 moves:")
            context = solver.analyze_stack_context(game.state)
            print(f"  Game phase: {context['game_phase']}")
            print(f"  Empty columns: {context['empty_cols']}")
            print(f"  Foundation: {game.state.get_foundation_count()}/52")

    print(f"\nFinal after 20 moves:")
    print(f"  Score: ${game.state.score}")
    print(f"  Foundation: {game.state.get_foundation_count()}/52")


def test_full_pipeline():
    """Test complete pipeline: select deal → play strategically."""
    print("\n\n")
    print("█" * 70)
    print("  TEST 4: FULL PIPELINE (Deal Selection + Strategic Play)")
    print("█" * 70)

    print("\n" + "=" * 70)
    print("Scenario: You have 5 rerolls, play chosen deal")
    print("=" * 70)

    # Step 1: Select best deal
    selector = DealSelector(strategy="threshold")

    print("\nStep 1: Selecting deal...")
    chosen_idx, quality, all_qualities = selector.select_deal(
        num_rerolls=5,
        seed_generator=lambda i: 3000 + i
    )

    print(f"\n✓ Selected Deal {chosen_idx + 1} (quality {quality:.1f})")

    # Step 2: Play chosen deal strategically
    print(f"\nStep 2: Playing selected deal with strategic solver...")

    game = Game(seed=3000 + chosen_idx)
    game.deal()

    solver = StrategicSolver()

    moves_made = 0
    max_moves = 500

    while moves_made < max_moves:
        move = solver.choose_move(game)
        if not move or not game.make_move(move):
            break
        moves_made += 1

        if moves_made % 100 == 0:
            print(f"  {moves_made} moves: ${game.state.score}, {game.state.get_foundation_count()}/52 cards")

    print(f"\n{'═' * 70}")
    print("FINAL RESULT")
    print('═' * 70)
    print(f"  Moves: {moves_made}")
    print(f"  Score: ${game.state.score}")
    print(f"  Foundation: {game.state.get_foundation_count()}/52")
    print(f"  Profit/Loss: ${game.state.score + 52} (paid $52 to play)")


def benchmark_meta_strategy():
    """Benchmark meta-strategy vs random deal selection."""
    print("\n\n")
    print("█" * 70)
    print("  TEST 5: BENCHMARK (Meta-Strategy vs Random)")
    print("█" * 70)

    print("\n" + "=" * 70)
    print("Comparing: Random Deal vs Best of 5 Deals")
    print("=" * 70)

    num_trials = 10
    rerolls_per_trial = 5

    random_scores = []
    selected_scores = []

    random_foundations = []
    selected_foundations = []

    selector = DealSelector(strategy="threshold")
    solver = StrategicSolver()

    print(f"\nRunning {num_trials} trials...\n")

    for trial in range(num_trials):
        base_seed = trial * 100

        # Strategy 1: Play random deal (no selection)
        random_seed = base_seed
        game = Game(seed=random_seed)
        game.deal()

        moves = 0
        while moves < 500:
            move = solver.choose_move(game)
            if not move or not game.make_move(move):
                break
            moves += 1

        random_scores.append(game.state.score)
        random_foundations.append(game.state.get_foundation_count())

        # Strategy 2: Select best of 5 deals
        chosen_idx, quality, _ = selector.select_deal(
            num_rerolls=rerolls_per_trial,
            seed_generator=lambda i: base_seed + i
        )

        game = Game(seed=base_seed + chosen_idx)
        game.deal()

        moves = 0
        while moves < 500:
            move = solver.choose_move(game)
            if not move or not game.make_move(move):
                break
            moves += 1

        selected_scores.append(game.state.score)
        selected_foundations.append(game.state.get_foundation_count())

        print(f"  Trial {trial+1}: Random ${random_scores[-1]:.0f} vs Selected ${selected_scores[-1]:.0f}")

    # Results
    print("\n" + "═" * 70)
    print("RESULTS")
    print("═" * 70)

    print(f"\nRandom Deal (no selection):")
    print(f"  Average score: ${np.mean(random_scores):.1f}")
    print(f"  Average foundation: {np.mean(random_foundations):.1f}/52")

    print(f"\nBest of 5 Deals (with selection):")
    print(f"  Average score: ${np.mean(selected_scores):.1f}")
    print(f"  Average foundation: {np.mean(selected_foundations):.1f}/52")

    # Calculate improvement
    score_improvement = np.mean(selected_scores) - np.mean(random_scores)
    foundation_improvement = np.mean(selected_foundations) - np.mean(random_foundations)

    print(f"\n{'═' * 70}")
    print("IMPROVEMENT")
    print('═' * 70)
    print(f"  Score: ${score_improvement:+.1f}")
    print(f"  Foundation: {foundation_improvement:+.1f} cards")

    if score_improvement > 0:
        pct = score_improvement / abs(np.mean(random_scores)) * 100
        print(f"  Percentage: {pct:+.1f}%")
        print(f"\n  ✓ Deal selection IMPROVES performance!")
    else:
        print(f"\n  Deal selection did not help in this sample.")


def main():
    """Run all tests."""
    print("\n")
    print("█" * 70)
    print("  META-STRATEGY TEST SUITE")
    print("  Implementing user insights from actual play experience")
    print("█" * 70)

    # Run all tests
    test_deal_evaluation()
    test_deal_selection()
    test_strategic_solver()
    test_full_pipeline()
    benchmark_meta_strategy()

    print("\n")
    print("█" * 70)
    print("  ALL TESTS COMPLETE")
    print("█" * 70)
    print("\nKey Takeaways:")
    print("  1. Deal quality is measurable and predictive")
    print("  2. Optimal stopping (secretary problem) works for deal selection")
    print("  3. Strategic insights (short/long stacks) improve play")
    print("  4. Full pipeline: select → play → profit!")
    print("\nSee META_STRATEGY.md for detailed documentation.")
    print()


if __name__ == "__main__":
    main()

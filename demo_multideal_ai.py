#!/usr/bin/env python3
"""
Demonstration of Multi-Deal Mode with AI Optimization.

This script showcases:
1. Q-Learning for move selection
2. TD Learning for value estimation
3. Intelligent deal selection (secretary problem, learned policies)
4. Bankroll management and reroll decisions
5. Comparison of different strategies

Usage:
    python demo_multideal_ai.py [--train] [--compare] [--play]
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from game.core.game import Game
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.meta.multideal_mode import MultidealManager, DealSelector, DealAcceptor
from optimization.meta.ai_deal_strategy import (
    LearnedDealSelector,
    LearnedDealAcceptor,
    create_ai_deal_strategy,
    compare_deal_strategies
)


def train_q_learning_policy(num_episodes: int = 1000, save_path: str = "policies/q_policy.json"):
    """
    Train a Q-Learning policy for Vegas Solitaire.

    Args:
        num_episodes: Number of training episodes
        save_path: Where to save trained policy
    """
    from optimization.solvers.rl.q_learning import train_q_policy

    print("=" * 70)
    print("Training Q-Learning Policy")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Save path: {save_path}")
    print()

    policy = train_q_policy(
        num_episodes=num_episodes,
        alpha=0.005,
        gamma=0.95,
        epsilon=0.15,
        save_path=save_path
    )

    print(f"\n✓ Q-Learning policy saved to {save_path}")
    return policy


def train_td_learning_policy(num_episodes: int = 1000, save_path: str = "policies/td_policy.json"):
    """
    Train a TD Learning policy for Vegas Solitaire.

    Args:
        num_episodes: Number of training episodes
        save_path: Where to save trained policy
    """
    from optimization.solvers.rl.td_learning import train_td_policy

    print("=" * 70)
    print("Training TD Learning Policy")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Save path: {save_path}")
    print()

    policy = train_td_policy(
        num_episodes=num_episodes,
        alpha=0.01,
        gamma=0.95,
        lambda_=0.7,
        save_path=save_path
    )

    print(f"\n✓ TD Learning policy saved to {save_path}")
    return policy


def demo_single_multideal_session():
    """
    Demonstrate a single multi-deal session with AI optimization.
    """
    print("=" * 70)
    print("Multi-Deal Mode: Single Session Demo")
    print("=" * 70)
    print()
    print("Scenario: You start with $500 at a Vegas casino.")
    print("You can reroll deals up to 3 times (costs $5 per reroll).")
    print("Each game costs $52 to play, pays $5 per card in foundation.")
    print("Rerolls reset when you accept and play a deal.")
    print()

    # Create solver for playing deals
    solver = HeuristicSolver()

    # Create deal selector and acceptor
    selector = DealSelector(strategy="secretary")
    acceptor = DealAcceptor(strategy="threshold")

    # Create multideal manager
    manager = MultidealManager(
        starting_bankroll=500.0,
        cost_per_game=52.0,
        reward_per_card=5.0,
        max_rerolls=3,
        reroll_cost=5.0,
        solver=solver,
        selector=selector,
        acceptor=acceptor
    )

    # Run session
    session = manager.run_session(
        max_deals=10,
        start_seed=42,
        verbose=True
    )

    # Summary
    print("\n" + "=" * 70)
    print("Session Summary")
    print("=" * 70)
    print(f"Starting bankroll: ${session.starting_bankroll:.2f}")
    print(f"Final bankroll:    ${session.final_bankroll:.2f}")
    print(f"Profit/Loss:       ${session.profit:+.2f}")
    print(f"Deals played:      {session.deals_played}")
    print(f"Deals evaluated:   {session.deals_evaluated}")
    print(f"Rerolls used:      {session.total_rerolls_used}")
    print(f"Win rate:          {session.win_rate*100:.1f}%")

    if session.results:
        avg_cards = np.mean([r.cards_played for r in session.results])
        avg_profit = np.mean([r.profit for r in session.results])
        print(f"Avg cards/deal:    {avg_cards:.1f}/52")
        print(f"Avg profit/deal:   ${avg_profit:+.2f}")


def demo_strategy_comparison():
    """
    Compare different deal selection strategies.
    """
    from optimization.meta.deal_selector import compare_strategies

    print("=" * 70)
    print("Deal Selection Strategy Comparison")
    print("=" * 70)
    print()
    print("Comparing three strategies for selecting which deal to play:")
    print("1. Secretary Problem (observe 37%, then take next better)")
    print("2. Threshold (calibrated quality threshold)")
    print("3. Best (oracle - always picks best)")
    print()

    compare_strategies(num_rerolls=5, num_trials=20)


def demo_multideal_with_qlearning():
    """
    Demonstrate multi-deal mode using Q-Learning policy.
    """
    from optimization.solvers.rl.q_learning import QPolicy

    print("=" * 70)
    print("Multi-Deal Mode with Q-Learning")
    print("=" * 70)
    print()

    # Try to load Q-Learning policy
    policy_path = "policies/q_policy.json"
    if Path(policy_path).exists():
        print(f"Loading Q-Learning policy from {policy_path}...")
        policy = QPolicy()
        policy.load(policy_path)
    else:
        print(f"No trained policy found at {policy_path}")
        print("Training a quick Q-Learning policy (500 episodes)...")
        policy = train_q_learning_policy(num_episodes=500, save_path=policy_path)

    print("\n✓ Q-Learning policy loaded")
    print()

    # Create learned deal strategy
    selector = LearnedDealSelector(policy=policy)
    acceptor = LearnedDealAcceptor(policy=policy)

    # Create solver (using heuristic for now)
    # TODO: Could create a PolicyBasedSolver that uses the Q-Learning policy
    solver = HeuristicSolver()

    # Create multideal manager
    manager = MultidealManager(
        starting_bankroll=500.0,
        max_rerolls=3,
        reroll_cost=5.0,
        solver=solver,
        selector=selector,
        acceptor=acceptor
    )

    # Run session
    print("Running multi-deal session with Q-Learning...")
    print()

    session = manager.run_session(
        max_deals=10,
        start_seed=2000,
        verbose=True
    )

    # Summary
    print("\n" + "=" * 70)
    print("Q-Learning Session Results")
    print("=" * 70)
    print(f"Final profit:      ${session.profit:+.2f}")
    print(f"Win rate:          {session.win_rate*100:.1f}%")
    print(f"Rerolls used:      {session.total_rerolls_used}")

    if session.results:
        avg_cards = np.mean([r.cards_played for r in session.results])
        print(f"Avg cards played:  {avg_cards:.1f}/52")


def demo_ai_strategy_comparison():
    """
    Compare AI strategies in multi-deal mode.
    """
    print("=" * 70)
    print("AI Strategy Comparison for Multi-Deal Mode")
    print("=" * 70)
    print()
    print("Comparing different AI approaches:")
    print("- Heuristic (baseline)")
    print("- Q-Learning (if trained)")
    print("- TD Learning (if trained)")
    print()

    compare_deal_strategies(
        num_sessions=5,
        deals_per_session=10,
        start_seed=3000
    )


def demo_reroll_decisions():
    """
    Demonstrate intelligent reroll decisions.
    """
    from optimization.meta.deal_evaluator import evaluate_deal_quality, DealComparator

    print("=" * 70)
    print("Intelligent Reroll Decisions")
    print("=" * 70)
    print()
    print("You have 3 rerolls. Should you accept each deal or reroll?")
    print()

    # Calibrate comparator
    comparator = DealComparator()
    comparator.calibrate(num_samples=1000)

    # Create acceptor
    acceptor = DealAcceptor(strategy="threshold")

    # Generate some deals
    print("Evaluating 10 random deals:\n")
    print(f"{'Deal':<6} {'Quality':<10} {'Decision':<10} {'Rerolls':<10}")
    print("-" * 40)

    rerolls = 3
    for i in range(10):
        game = Game(seed=100 + i)
        game.deal()

        quality = evaluate_deal_quality(game.state)
        accept, _ = acceptor.should_accept(game.state, rerolls, bankroll=500)

        decision = "ACCEPT ✓" if accept else "REROLL ↻"

        print(f"{i+1:<6} {quality:<10.1f} {decision:<10} {rerolls}/3")

        if accept:
            # Play this deal - reset rerolls
            rerolls = 3
        else:
            # Reroll
            rerolls -= 1
            if rerolls == 0:
                rerolls = 3  # Reset for demo

    print()
    print("Decision factors:")
    print(f"  - Quality threshold: {acceptor.quality_threshold:.1f}")
    print(f"  - Reroll cost: $5")
    print(f"  - Expected value must exceed cost of rerolling")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Deal Mode with AI Optimization Demo"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train Q-Learning and TD Learning policies"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different AI strategies"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play a multi-deal session with Q-Learning"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demos"
    )

    args = parser.parse_args()

    # If no args, run basic demos
    if not any([args.train, args.compare, args.play, args.all]):
        print("Multi-Deal Mode with AI Optimization")
        print("=" * 70)
        print()
        print("Running basic demos...")
        print()

        # Demo 1: Single session
        demo_single_multideal_session()

        print("\n\n")

        # Demo 2: Strategy comparison
        demo_strategy_comparison()

        print("\n\n")

        # Demo 3: Reroll decisions
        demo_reroll_decisions()

        print("\n\n")
        print("=" * 70)
        print("For more demos, try:")
        print("  --train     Train Q-Learning and TD Learning policies")
        print("  --play      Play with Q-Learning policy")
        print("  --compare   Compare AI strategies")
        print("  --all       Run all demos")
        print("=" * 70)

        return

    # Train policies
    if args.train or args.all:
        print("\n" + "=" * 70)
        print("TRAINING POLICIES")
        print("=" * 70 + "\n")

        # Create policies directory
        Path("policies").mkdir(exist_ok=True)

        # Train Q-Learning
        train_q_learning_policy(num_episodes=1000)

        print("\n")

        # Train TD Learning
        train_td_learning_policy(num_episodes=1000)

        print("\n")

    # Play with Q-Learning
    if args.play or args.all:
        print("\n" + "=" * 70)
        print("PLAYING WITH Q-LEARNING")
        print("=" * 70 + "\n")

        demo_multideal_with_qlearning()

        print("\n")

    # Compare strategies
    if args.compare or args.all:
        print("\n" + "=" * 70)
        print("COMPARING STRATEGIES")
        print("=" * 70 + "\n")

        demo_ai_strategy_comparison()

        print("\n")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

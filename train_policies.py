"""
Train TD Learning policies and save them for reuse.

This script trains multiple TD Learning configurations and saves
the best-performing policies.
"""

import numpy as np
from pathlib import Path
from optimization.solvers.rl.td_learning import TDLearner, TDPolicy
from game.core.game import Game
from optimization.evaluation.benchmark import Benchmark


def train_td_policy_config(config_name: str,
                           num_episodes: int,
                           alpha: float,
                           gamma: float,
                           lambda_: float,
                           epsilon: float,
                           save_dir: str = "trained_policies"):
    """
    Train a TD policy with specific configuration.

    Args:
        config_name: Name for this configuration
        num_episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        lambda_: Eligibility trace parameter
        epsilon: Exploration rate
        save_dir: Directory to save trained policies

    Returns:
        Trained policy and training statistics
    """
    print("\n" + "=" * 70)
    print(f"Training Configuration: {config_name}")
    print("=" * 70)
    print(f"  Episodes: {num_episodes}")
    print(f"  Learning rate (α): {alpha}")
    print(f"  Discount (γ): {gamma}")
    print(f"  Eligibility (λ): {lambda_}")
    print(f"  Exploration (ε): {epsilon}")
    print()

    # Create learner
    learner = TDLearner(
        alpha=alpha,
        gamma=gamma,
        lambda_=lambda_,
        epsilon=epsilon,
        name=f"TD-{config_name}"
    )

    # Training statistics
    scores = []
    foundations = []
    moves_list = []

    # Training loop
    for episode in range(num_episodes):
        game = Game(seed=episode)
        game.deal()

        score, moves = learner.train_episode(game, verbose=False)
        scores.append(score)
        foundations.append(game.state.get_foundation_count())
        moves_list.append(moves)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            recent_scores = scores[-100:]
            recent_found = foundations[-100:]
            recent_moves = moves_list[-100:]

            print(f"  Episode {episode+1:4d}/{num_episodes}: "
                  f"Avg Score ${np.mean(recent_scores):6.1f}, "
                  f"Avg Foundation {np.mean(recent_found):4.1f}/52, "
                  f"Avg Moves {np.mean(recent_moves):5.1f}")

    # Get final policy
    policy = learner.get_policy()

    # Save policy
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    policy_file = save_path / f"{config_name}.json"
    policy.save(str(policy_file))

    print(f"\n  ✓ Policy saved to: {policy_file}")

    # Final statistics
    final_avg_score = np.mean(scores[-100:])
    final_avg_found = np.mean(foundations[-100:])
    final_avg_moves = np.mean(moves_list[-100:])

    print(f"\n  Final Performance (last 100 episodes):")
    print(f"    Average Score: ${final_avg_score:.1f}")
    print(f"    Average Foundation: {final_avg_found:.1f}/52")
    print(f"    Average Moves: {final_avg_moves:.1f}")

    return policy, {
        'scores': scores,
        'foundations': foundations,
        'moves': moves_list,
        'config': {
            'alpha': alpha,
            'gamma': gamma,
            'lambda': lambda_,
            'epsilon': epsilon,
            'num_episodes': num_episodes
        }
    }


def evaluate_trained_policy(policy: TDPolicy, num_games: int = 100, start_seed: int = 10000):
    """
    Evaluate a trained policy on fresh test games.

    Args:
        policy: Trained policy to evaluate
        num_games: Number of test games
        start_seed: Starting seed for test games

    Returns:
        Evaluation results
    """
    print("\n" + "-" * 70)
    print("Evaluating Trained Policy on Test Set")
    print("-" * 70)

    scores = []
    foundations = []
    moves_list = []
    wins = 0

    for i in range(num_games):
        game = Game(seed=start_seed + i)
        game.deal()

        moves_made = 0
        max_moves = 500

        while moves_made < max_moves:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break

            move = policy.choose_action(game.state, valid_moves)
            if not move or not game.make_move(move):
                break

            moves_made += 1

        scores.append(game.state.score)
        foundations.append(game.state.get_foundation_count())
        moves_list.append(moves_made)

        if game.state.is_winning():
            wins += 1

        if (i + 1) % 20 == 0:
            print(f"  Tested {i+1}/{num_games} games...")

    print(f"\n  Test Results:")
    print(f"    Win Rate: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"    Average Score: ${np.mean(scores):.1f}")
    print(f"    Average Foundation: {np.mean(foundations):.1f}/52")
    print(f"    Average Moves: {np.mean(moves_list):.1f}")
    print(f"    Best Score: ${max(scores)}")
    print(f"    Best Foundation: {max(foundations)}/52")

    return {
        'win_rate': wins / num_games,
        'avg_score': np.mean(scores),
        'avg_foundation': np.mean(foundations),
        'avg_moves': np.mean(moves_list),
        'best_score': max(scores),
        'best_foundation': max(foundations)
    }


def main():
    """Run full training pipeline."""
    print("\n")
    print("█" * 70)
    print("  TD LEARNING TRAINING PIPELINE")
    print("█" * 70)

    # Configuration 1: TD(0) - Basic temporal difference
    policy1, stats1 = train_td_policy_config(
        config_name="td0_standard",
        num_episodes=1000,
        alpha=0.01,
        gamma=0.95,
        lambda_=0.0,  # TD(0)
        epsilon=0.1
    )

    # Evaluate on test set
    eval1 = evaluate_trained_policy(policy1, num_games=100)

    # Configuration 2: TD(λ) with eligibility traces
    policy2, stats2 = train_td_policy_config(
        config_name="td_lambda_standard",
        num_episodes=1000,
        alpha=0.01,
        gamma=0.95,
        lambda_=0.7,  # Eligibility traces
        epsilon=0.1
    )

    # Evaluate on test set
    eval2 = evaluate_trained_policy(policy2, num_games=100)

    # Configuration 3: Higher learning rate
    policy3, stats3 = train_td_policy_config(
        config_name="td0_fast_learning",
        num_episodes=1000,
        alpha=0.05,  # Higher learning rate
        gamma=0.95,
        lambda_=0.0,
        epsilon=0.2  # More exploration
    )

    # Evaluate on test set
    eval3 = evaluate_trained_policy(policy3, num_games=100)

    # Summary
    print("\n" + "█" * 70)
    print("  TRAINING SUMMARY")
    print("█" * 70)

    configs = [
        ("TD(0) Standard", eval1),
        ("TD(λ=0.7) Standard", eval2),
        ("TD(0) Fast Learning", eval3)
    ]

    print("\nTest Set Performance (100 games, seeds 10000-10099):")
    print("-" * 70)
    print(f"{'Configuration':<25} {'Win Rate':>10} {'Avg Score':>12} {'Avg Found':>12}")
    print("-" * 70)

    for name, result in configs:
        print(f"{name:<25} {result['win_rate']*100:>9.1f}% "
              f"${result['avg_score']:>10.1f} {result['avg_foundation']:>10.1f}/52")

    print("\n" + "█" * 70)
    print("  TRAINING COMPLETE!")
    print("  Trained policies saved to: trained_policies/")
    print("█" * 70)
    print()


if __name__ == "__main__":
    main()

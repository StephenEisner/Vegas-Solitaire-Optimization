"""Test script for advanced algorithms (Beam Search and TD Learning)."""

import numpy as np
from game.core.game import Game
from optimization.solvers.search.beam_search import BeamSearch, FeatureBasedBeamSearch
from optimization.solvers.rl.td_learning import TDLearner, train_td_policy
from optimization.features.state_features import StateFeatures


def test_beam_search():
    """Test Beam Search solver."""
    print("=" * 70)
    print("Testing Beam Search")
    print("=" * 70)

    # Test with different beam widths
    for beam_width in [10, 50, 100]:
        print(f"\n--- Beam Width: {beam_width} ---")

        solver = BeamSearch(beam_width=beam_width, max_depth=20)
        game = Game(seed=42)
        game.deal()

        moves_made = 0
        max_moves = 100

        while moves_made < max_moves:
            move = solver.choose_move(game)
            if not move:
                break

            if not game.make_move(move):
                break

            moves_made += 1

            if moves_made % 20 == 0:
                print(f"  Move {moves_made}: Score ${game.state.score}, "
                      f"Foundation {game.state.get_foundation_count()}/52")

        stats = solver.get_stats()
        print(f"\n  Final: {moves_made} moves, ${game.state.score}")
        print(f"  Foundation: {game.state.get_foundation_count()}/52")
        print(f"  Nodes expanded: {stats['nodes_expanded']}")
        print(f"  Max beam size: {stats['max_beam_size']}")


def test_feature_beam_search():
    """Test feature-based beam search."""
    print("\n" + "=" * 70)
    print("Testing Feature-Based Beam Search")
    print("=" * 70)

    solver = FeatureBasedBeamSearch(beam_width=50, max_depth=20)
    game = Game(seed=42)
    game.deal()

    moves_made = 0
    max_moves = 100

    while moves_made < max_moves:
        move = solver.choose_move(game)
        if not move:
            break

        if not game.make_move(move):
            break

        moves_made += 1

    print(f"\n  Final: {moves_made} moves, ${game.state.score}")
    print(f"  Foundation: {game.state.get_foundation_count()}/52")


def test_state_features():
    """Test state feature extraction."""
    print("\n" + "=" * 70)
    print("Testing State Feature Extraction")
    print("=" * 70)

    game = Game(seed=42)
    game.deal()

    # Extract features
    features = StateFeatures.extract_features(game.state)
    print(f"\nBasic features shape: {features.shape}")
    print(f"Feature vector:\n{features}")

    # Test feature description
    print("\n" + StateFeatures.describe_features(features))

    # Test extended features
    extended = StateFeatures.extract_extended_features(game.state)
    print(f"\nExtended features shape: {extended.shape}")


def test_td_learning_quick():
    """Test TD learning with a quick training run."""
    print("\n" + "=" * 70)
    print("Testing TD Learning (Quick Training)")
    print("=" * 70)

    # Quick training (just 50 episodes to verify it works)
    learner = TDLearner(alpha=0.01, gamma=0.95, lambda_=0.0, epsilon=0.2)

    print("\nRunning 50 training episodes...")
    scores = []
    foundations = []

    for episode in range(50):
        game = Game(seed=episode)
        game.deal()

        score, moves = learner.train_episode(game, verbose=False)
        scores.append(score)
        foundations.append(game.state.get_foundation_count())

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1:2d}: Score ${score:4.0f}, "
                  f"Foundation {game.state.get_foundation_count():2d}/52, "
                  f"Moves {moves:3d}")

    print(f"\n  Average score (last 10): ${np.mean(scores[-10:]):.1f}")
    print(f"  Average foundation: {np.mean(foundations[-10:]):.1f}/52")

    # Test the learned policy
    print("\n  Testing learned policy on new game...")
    policy = learner.get_policy()
    test_game = Game(seed=999)
    test_game.deal()

    moves_made = 0
    while moves_made < 100:
        valid_moves = test_game.get_valid_moves()
        if not valid_moves:
            break

        move = policy.choose_action(test_game.state, valid_moves)
        if not move or not test_game.make_move(move):
            break

        moves_made += 1

    print(f"  Test game: ${test_game.state.score}, "
          f"{test_game.state.get_foundation_count()}/52 in {moves_made} moves")


def main():
    """Run all tests."""
    print("\n")
    print("█" * 70)
    print("  ADVANCED ALGORITHMS TEST SUITE")
    print("█" * 70)

    # Test feature extraction first
    test_state_features()

    # Test Beam Search
    test_beam_search()
    test_feature_beam_search()

    # Test TD Learning
    test_td_learning_quick()

    print("\n" + "█" * 70)
    print("  ALL TESTS COMPLETE")
    print("█" * 70)
    print()


if __name__ == "__main__":
    main()

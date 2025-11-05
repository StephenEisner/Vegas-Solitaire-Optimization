#!/usr/bin/env python3
"""
Train machine-generated evaluation functions for Vegas Solitaire.

This script provides a complete pipeline for:
1. Generating training data from self-play
2. Training neural network evaluation functions
3. Evaluating learned models
4. Comparing against baseline solvers

Usage:
    python train_evaluation_function.py --generate --num-games 100
    python train_evaluation_function.py --train --epochs 50
    python train_evaluation_function.py --evaluate --num-tests 20
    python train_evaluation_function.py --all  # Complete pipeline
"""

import argparse
from pathlib import Path
import numpy as np

from game.core.game import Game
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.evaluation.learned_evaluation import (
    TrainingDataGenerator,
    NeuralEvaluationFunction
)
from optimization.solvers.learned_solver import (
    LearnedEvaluationSolver,
    compare_learned_vs_heuristic
)


def generate_training_data(num_games: int,
                          start_seed: int,
                          output_path: str = "data/training_trajectories.json"):
    """Generate training data from self-play."""
    print("=" * 70)
    print("STEP 1: Generating Training Data")
    print("=" * 70)
    print()

    # Create solvers for data generation
    solvers = [
        HeuristicSolver(),
        # Add more solvers for diversity
    ]

    print(f"Generating {num_games} game trajectories...")
    print(f"Using solvers: {[s.name for s in solvers]}")
    print()

    generator = TrainingDataGenerator(solvers)
    trajectories = generator.generate_dataset(
        num_games=num_games,
        start_seed=start_seed,
        verbose=True
    )

    # Save dataset
    generator.save_dataset(output_path)

    return trajectories


def train_evaluation_function(data_path: str = "data/training_trajectories.json",
                             hidden_dims: list = [128, 64, 32],
                             target: str = 'monte_carlo',
                             learning_rate: float = 0.001,
                             batch_size: int = 32,
                             epochs: int = 20,
                             output_path: str = "models/neural_eval.json"):
    """Train neural evaluation function."""
    print("\n" + "=" * 70)
    print("STEP 2: Training Neural Evaluation Function")
    print("=" * 70)
    print()

    # Load training data
    print(f"Loading training data from {data_path}...")
    generator = TrainingDataGenerator([])
    trajectories = generator.load_dataset(data_path)

    # Create network
    print(f"\nCreating neural network...")
    print(f"Architecture: {hidden_dims}")
    network = NeuralEvaluationFunction(hidden_dims=hidden_dims)

    # Train
    history = network.train_supervised(
        trajectories=trajectories,
        target=target,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        verbose=True
    )

    # Save model
    network.save(output_path)

    return network, history


def evaluate_learned_function(model_path: str = "models/neural_eval.json",
                              num_test_games: int = 20):
    """Evaluate learned function against baselines."""
    print("\n" + "=" * 70)
    print("STEP 3: Evaluating Learned Function")
    print("=" * 70)
    print()

    compare_learned_vs_heuristic(
        num_games=num_test_games,
        eval_path=model_path
    )


def visualize_learning(history: dict):
    """Visualize training progress."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training')
        plt.plot(history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.yscale('log')
        plt.title('Training Progress (Log Scale)')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()

        Path("analysis").mkdir(exist_ok=True)
        plt.savefig("analysis/training_progress.png", dpi=150)
        print(f"\n✓ Training progress saved to analysis/training_progress.png")
        plt.close()

    except ImportError:
        print("\n(matplotlib not available - skipping visualization)")


def analyze_learned_function(model_path: str = "models/neural_eval.json"):
    """Analyze what the learned function has learned."""
    print("\n" + "=" * 70)
    print("Analyzing Learned Function")
    print("=" * 70)
    print()

    # Load model
    network = NeuralEvaluationFunction()
    network.load(model_path)

    # Test on various game states
    print("Testing predictions on sample games:\n")

    test_seeds = [42, 100, 500, 1000, 5000]

    for seed in test_seeds:
        game = Game(seed=seed)
        game.deal()

        from optimization.features.state_features import StateFeatures
        features = StateFeatures.extract_features(game.state)

        # Normalize
        features_norm = (features - network.input_mean) / network.input_std
        predicted_value = network.predict(features_norm)

        # Denormalize
        predicted_value = predicted_value * network.target_std + network.target_mean

        # Also get heuristic evaluation
        from optimization.meta.deal_evaluator import evaluate_deal_quality
        heuristic_quality = evaluate_deal_quality(game.state)

        print(f"Seed {seed}:")
        print(f"  Learned prediction: {predicted_value:.2f}")
        print(f"  Heuristic quality: {heuristic_quality:.1f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Train machine-generated evaluation functions"
    )

    # Modes
    parser.add_argument("--generate", action="store_true",
                       help="Generate training data")
    parser.add_argument("--train", action="store_true",
                       help="Train evaluation function")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate learned function")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze learned function")
    parser.add_argument("--all", action="store_true",
                       help="Run complete pipeline")

    # Data generation options
    parser.add_argument("--num-games", type=int, default=100,
                       help="Number of games for training data")
    parser.add_argument("--seed", type=int, default=10000,
                       help="Starting seed for data generation")

    # Training options
    parser.add_argument("--hidden-dims", type=str, default="128,64,32",
                       help="Hidden layer dimensions (comma-separated)")
    parser.add_argument("--target", type=str, default="monte_carlo",
                       choices=["monte_carlo", "td", "final_outcome"],
                       help="Training target type")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of epochs")

    # Evaluation options
    parser.add_argument("--num-tests", type=int, default=20,
                       help="Number of test games")

    # Paths
    parser.add_argument("--data-path", type=str,
                       default="data/training_trajectories.json",
                       help="Path for training data")
    parser.add_argument("--model-path", type=str,
                       default="models/neural_eval.json",
                       help="Path for trained model")

    args = parser.parse_args()

    # Parse hidden dims
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    # Run pipeline
    if args.all:
        # Complete pipeline
        print("=" * 70)
        print("COMPLETE TRAINING PIPELINE")
        print("=" * 70)
        print()

        # Generate data
        trajectories = generate_training_data(
            num_games=args.num_games,
            start_seed=args.seed,
            output_path=args.data_path
        )

        # Train
        network, history = train_evaluation_function(
            data_path=args.data_path,
            hidden_dims=hidden_dims,
            target=args.target,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_path=args.model_path
        )

        # Visualize
        visualize_learning(history)

        # Analyze
        analyze_learned_function(args.model_path)

        # Evaluate
        evaluate_learned_function(
            model_path=args.model_path,
            num_test_games=args.num_tests
        )

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"\nTrained model saved to: {args.model_path}")
        print(f"You can now use this model with LearnedEvaluationSolver")

    else:
        # Individual steps
        if args.generate:
            generate_training_data(
                num_games=args.num_games,
                start_seed=args.seed,
                output_path=args.data_path
            )

        if args.train:
            network, history = train_evaluation_function(
                data_path=args.data_path,
                hidden_dims=hidden_dims,
                target=args.target,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                output_path=args.model_path
            )
            visualize_learning(history)

        if args.analyze:
            analyze_learned_function(args.model_path)

        if args.evaluate:
            evaluate_learned_function(
                model_path=args.model_path,
                num_test_games=args.num_tests
            )

        if not any([args.generate, args.train, args.evaluate, args.analyze]):
            parser.print_help()


if __name__ == "__main__":
    main()

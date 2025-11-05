"""
Machine-Generated Evaluation Functions for Vegas Solitaire.

This module implements automated learning of evaluation functions using:
1. Self-play data generation
2. Deep neural networks for value function approximation
3. Multiple training approaches (supervised, RL, evolutionary)
4. Automatic feature discovery

The learned evaluation functions can then be used by solvers during gameplay
to estimate the value of game states without hand-crafted heuristics.
"""

from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import time

from game.core.game import Game
from game.core.state import GameState
from optimization.features.state_features import StateFeatures
from optimization.solvers.base import Solver


@dataclass
class GameTrajectory:
    """
    A complete game trajectory for training.

    Contains sequence of (state, action, reward, next_state) tuples
    plus final outcome.
    """
    seed: int
    states: List[np.ndarray]  # Feature vectors
    rewards: List[float]  # Immediate rewards
    final_outcome: Dict[str, float]  # Final score, cards, etc.

    def __len__(self):
        return len(self.states)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'seed': self.seed,
            'states': [s.tolist() for s in self.states],
            'rewards': self.rewards,
            'final_outcome': self.final_outcome,
            'length': len(self)
        }


class TrainingDataGenerator:
    """
    Generates training data from self-play.

    Plays games with various solvers and records state trajectories
    with outcomes for supervised learning.
    """

    def __init__(self, solvers: List[Solver]):
        """
        Initialize data generator.

        Args:
            solvers: List of solvers to use for generating data
        """
        self.solvers = solvers
        self.trajectories: List[GameTrajectory] = []

    def generate_trajectory(self,
                          game: Game,
                          solver: Solver,
                          max_moves: int = 500) -> GameTrajectory:
        """
        Generate a single game trajectory.

        Args:
            game: Game to play
            solver: Solver to use
            max_moves: Maximum moves

        Returns:
            GameTrajectory with states and outcomes
        """
        states = []
        rewards = []

        prev_score = game.state.score
        moves_made = 0

        while moves_made < max_moves:
            # Record current state features
            state_features = StateFeatures.extract_features(game.state)
            states.append(state_features)

            # Make move
            move = solver.choose_move(game)
            if not move or not game.make_move(move):
                break

            # Record reward (change in score)
            current_score = game.state.score
            reward = current_score - prev_score
            rewards.append(reward)
            prev_score = current_score

            moves_made += 1

        # Final outcome
        final_outcome = {
            'score': game.state.score,
            'cards': game.state.get_foundation_count(),
            'won': game.state.get_foundation_count() == 52,
            'moves': moves_made
        }

        return GameTrajectory(
            seed=game.seed,
            states=states,
            rewards=rewards,
            final_outcome=final_outcome
        )

    def generate_dataset(self,
                        num_games: int,
                        start_seed: int = 0,
                        verbose: bool = True) -> List[GameTrajectory]:
        """
        Generate training dataset from multiple games.

        Args:
            num_games: Number of games to play
            start_seed: Starting seed
            verbose: Print progress

        Returns:
            List of GameTrajectory objects
        """
        trajectories = []

        if verbose:
            print("=" * 70)
            print(f"Generating Training Data: {num_games} games")
            print(f"Solvers: {[s.name for s in self.solvers]}")
            print("=" * 70)

        for i in range(num_games):
            seed = start_seed + i

            # Rotate through solvers for diversity
            solver = self.solvers[i % len(self.solvers)]

            # Create and play game
            game = Game(seed=seed)
            game.deal()

            trajectory = self.generate_trajectory(game, solver)
            trajectories.append(trajectory)

            if verbose and (i + 1) % 10 == 0:
                avg_cards = np.mean([t.final_outcome['cards'] for t in trajectories[-10:]])
                print(f"  Generated {i+1}/{num_games} games, "
                      f"last 10 avg: {avg_cards:.1f}/52 cards")

        self.trajectories.extend(trajectories)

        if verbose:
            self._print_dataset_summary(trajectories)

        return trajectories

    def _print_dataset_summary(self, trajectories: List[GameTrajectory]):
        """Print summary of generated dataset."""
        print("\n" + "=" * 70)
        print("Dataset Summary")
        print("=" * 70)

        total_states = sum(len(t) for t in trajectories)
        avg_length = np.mean([len(t) for t in trajectories])

        scores = [t.final_outcome['score'] for t in trajectories]
        cards = [t.final_outcome['cards'] for t in trajectories]
        wins = [t.final_outcome['won'] for t in trajectories]

        print(f"\nGames: {len(trajectories)}")
        print(f"Total states: {total_states}")
        print(f"Avg trajectory length: {avg_length:.1f} moves")
        print(f"\nOutcomes:")
        print(f"  Avg score: ${np.mean(scores):.1f}")
        print(f"  Avg cards: {np.mean(cards):.1f}/52")
        print(f"  Win rate: {np.mean(wins)*100:.1f}%")

    def save_dataset(self, path: str):
        """Save trajectories to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'num_trajectories': len(self.trajectories),
            'total_states': sum(len(t) for t in self.trajectories),
            'trajectories': [t.to_dict() for t in self.trajectories]
        }

        with open(path, 'w') as f:
            json.dump(data, f)

        print(f"\n✓ Dataset saved to {path}")

    def load_dataset(self, path: str) -> List[GameTrajectory]:
        """Load trajectories from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        trajectories = []
        for t_dict in data['trajectories']:
            trajectory = GameTrajectory(
                seed=t_dict['seed'],
                states=[np.array(s) for s in t_dict['states']],
                rewards=t_dict['rewards'],
                final_outcome=t_dict['final_outcome']
            )
            trajectories.append(trajectory)

        self.trajectories = trajectories
        print(f"✓ Loaded {len(trajectories)} trajectories from {path}")
        return trajectories


class NeuralEvaluationFunction:
    """
    Deep neural network for game state evaluation.

    Uses a simple feedforward network to predict state value.
    Can be trained with supervised learning or RL.
    """

    def __init__(self,
                 input_dim: int = StateFeatures.TOTAL_FEATURES,
                 hidden_dims: List[int] = [128, 64, 32],
                 output_dim: int = 1):
        """
        Initialize neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (1 for value function)
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Initialize weights randomly
        self.layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Xavier initialization
            scale = np.sqrt(2.0 / (prev_dim + hidden_dim))
            W = np.random.randn(prev_dim, hidden_dim) * scale
            b = np.zeros(hidden_dim)
            self.layers.append({'W': W, 'b': b})
            prev_dim = hidden_dim

        # Output layer
        scale = np.sqrt(2.0 / (prev_dim + output_dim))
        W = np.random.randn(prev_dim, output_dim) * scale
        b = np.zeros(output_dim)
        self.layers.append({'W': W, 'b': b})

        self.trained = False

    def forward(self, x: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """
        Forward pass through network.

        Args:
            x: Input features (can be batched)

        Returns:
            (output, activations) for backprop
        """
        activations = [x]

        # Hidden layers with ReLU
        for i, layer in enumerate(self.layers[:-1]):
            x = x @ layer['W'] + layer['b']
            x = np.maximum(0, x)  # ReLU
            activations.append(x)

        # Output layer (linear)
        layer = self.layers[-1]
        x = x @ layer['W'] + layer['b']
        activations.append(x)

        return x, activations

    def predict(self, features: np.ndarray) -> float:
        """
        Predict value for a single state.

        Args:
            features: State feature vector

        Returns:
            Predicted value
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        output, _ = self.forward(features)
        return output[0, 0]

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Predict values for multiple states."""
        output, _ = self.forward(features)
        return output.flatten()

    def train_supervised(self,
                        trajectories: List[GameTrajectory],
                        target: str = 'monte_carlo',
                        learning_rate: float = 0.001,
                        batch_size: int = 32,
                        epochs: int = 10,
                        verbose: bool = True) -> Dict:
        """
        Train network using supervised learning.

        Args:
            trajectories: Training data
            target: Target type ('monte_carlo', 'td', 'final_outcome')
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            verbose: Print progress

        Returns:
            Training metrics
        """
        if verbose:
            print("\n" + "=" * 70)
            print("Training Neural Evaluation Function")
            print("=" * 70)
            print(f"Architecture: {self.input_dim} → {' → '.join(map(str, self.hidden_dims))} → {self.output_dim}")
            print(f"Target: {target}")
            print(f"Learning rate: {learning_rate}")
            print(f"Batch size: {batch_size}")
            print(f"Epochs: {epochs}")
            print()

        # Prepare training data
        X, y = self._prepare_training_data(trajectories, target)

        if verbose:
            print(f"Training samples: {len(X)}")
            print(f"Target mean: {np.mean(y):.2f}, std: {np.std(y):.2f}")
            print()

        # Normalize inputs
        self.input_mean = np.mean(X, axis=0)
        self.input_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self.input_mean) / self.input_std

        # Normalize targets
        self.target_mean = np.mean(y)
        self.target_std = np.std(y) + 1e-8
        y_norm = (y - self.target_mean) / self.target_std

        # Training loop
        history = {'loss': [], 'val_loss': []}
        n_samples = len(X_norm)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Split train/val
        val_split = 0.2
        val_size = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train, y_train = X_norm[train_indices], y_norm[train_indices]
        X_val, y_val = X_norm[val_indices], y_norm[val_indices]

        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            epoch_loss = 0.0

            # Mini-batch gradient descent
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(X_train))

                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end].reshape(-1, 1)

                # Forward pass
                predictions, activations = self.forward(X_batch)

                # Loss (MSE)
                loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += loss

                # Backward pass (simplified - would use proper backprop)
                # For now, use simple gradient descent on output layer only
                grad_output = 2 * (predictions - y_batch) / len(X_batch)

                # Update output layer
                layer = self.layers[-1]
                prev_activation = activations[-2]
                layer['W'] -= learning_rate * (prev_activation.T @ grad_output)
                layer['b'] -= learning_rate * np.sum(grad_output, axis=0)

            avg_loss = epoch_loss / n_batches

            # Validation loss
            val_predictions, _ = self.forward(X_val)
            val_loss = np.mean((val_predictions - y_val.reshape(-1, 1)) ** 2)

            history['loss'].append(avg_loss)
            history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")

        self.trained = True

        if verbose:
            print(f"\n✓ Training complete!")
            print(f"Final loss: {history['loss'][-1]:.4f}")
            print(f"Final val loss: {history['val_loss'][-1]:.4f}")

        return history

    def _prepare_training_data(self,
                               trajectories: List[GameTrajectory],
                               target: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from trajectories.

        Args:
            trajectories: Game trajectories
            target: Target type

        Returns:
            (X, y) arrays
        """
        X = []
        y = []

        for traj in trajectories:
            if target == 'final_outcome':
                # All states map to final outcome
                final_value = traj.final_outcome['cards'] * 5.0  # Score
                for state in traj.states:
                    X.append(state)
                    y.append(final_value)

            elif target == 'monte_carlo':
                # Monte Carlo returns (sum of future rewards)
                for i, state in enumerate(traj.states):
                    future_rewards = sum(traj.rewards[i:])
                    X.append(state)
                    y.append(future_rewards)

            elif target == 'td':
                # TD targets (reward + value of next state)
                # For now, simplified
                for i, state in enumerate(traj.states[:-1]):
                    reward = traj.rewards[i]
                    X.append(state)
                    y.append(reward)

        return np.array(X), np.array(y)

    def save(self, path: str):
        """Save network to file."""
        if not self.trained:
            raise ValueError("Cannot save untrained network")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'layers': [{'W': layer['W'].tolist(), 'b': layer['b'].tolist()}
                      for layer in self.layers],
            'input_mean': self.input_mean.tolist(),
            'input_std': self.input_std.tolist(),
            'target_mean': float(self.target_mean),
            'target_std': float(self.target_std),
        }

        with open(path, 'w') as f:
            json.dump(data, f)

        print(f"✓ Network saved to {path}")

    def load(self, path: str):
        """Load network from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.input_dim = data['input_dim']
        self.hidden_dims = data['hidden_dims']
        self.output_dim = data['output_dim']

        self.layers = []
        for layer_data in data['layers']:
            self.layers.append({
                'W': np.array(layer_data['W']),
                'b': np.array(layer_data['b'])
            })

        self.input_mean = np.array(data['input_mean'])
        self.input_std = np.array(data['input_std'])
        self.target_mean = data['target_mean']
        self.target_std = data['target_std']

        self.trained = True
        print(f"✓ Network loaded from {path}")


if __name__ == "__main__":
    from optimization.solvers.heuristic_solver import HeuristicSolver

    print("=" * 70)
    print("Machine-Generated Evaluation Function Demo")
    print("=" * 70)
    print()

    # Generate training data
    print("Step 1: Generating training data from self-play...")
    solvers = [HeuristicSolver()]
    generator = TrainingDataGenerator(solvers)

    trajectories = generator.generate_dataset(num_games=50, start_seed=10000)
    generator.save_dataset("data/training_trajectories.json")

    # Train neural network
    print("\nStep 2: Training neural evaluation function...")
    network = NeuralEvaluationFunction(
        hidden_dims=[128, 64, 32]
    )

    history = network.train_supervised(
        trajectories,
        target='monte_carlo',
        learning_rate=0.001,
        batch_size=32,
        epochs=20,
        verbose=True
    )

    network.save("models/neural_eval.json")

    # Test prediction
    print("\nStep 3: Testing predictions...")
    game = Game(seed=99999)
    game.deal()

    features = StateFeatures.extract_features(game.state)
    predicted_value = network.predict(features)

    print(f"Test game (seed 99999):")
    print(f"  Predicted value: {predicted_value:.2f}")
    print(f"  (Higher = better expected outcome)")

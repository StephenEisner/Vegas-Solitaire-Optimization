"""
TD(λ) Learning for Vegas Solitaire.

Temporal Difference learning is a fundamental RL algorithm that learns
a value function V(s) from experience. We then use this to derive a policy.

Key concepts:
- Bootstrapping: Update estimates based on other estimates
- TD error: δ = r + γV(s') - V(s)
- Eligibility traces: Credit recent states more (controlled by λ)

References:
- Sutton & Barto, "Reinforcement Learning", Chapter 12
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

from game.core.game import Game
from game.core.state import GameState
from game.core.moves import Move
from game.core.rules import apply_move, get_valid_moves
from optimization.policies.base import ValueBasedPolicy
from optimization.features.state_features import StateFeatures, compute_state_hash
from optimization.solvers.base import Solver


class TDPolicy(ValueBasedPolicy):
    """
    Policy that uses learned TD value function.

    This policy acts greedily with respect to the learned value function:
    choose the move that leads to the state with highest value.
    """

    def __init__(self,
                 name: str = "TD-Learned",
                 feature_weights: Optional[np.ndarray] = None,
                 use_extended_features: bool = False):
        """
        Initialize TD policy.

        Args:
            name: Policy name
            feature_weights: Learned weights for value function
            use_extended_features: Use extended feature set
        """
        super().__init__(name)
        self.use_extended_features = use_extended_features

        # Initialize weights
        if feature_weights is None:
            feature_dim = StateFeatures.TOTAL_FEATURES
            self.feature_weights = np.zeros(feature_dim, dtype=np.float32)
        else:
            self.feature_weights = feature_weights.copy()

        self.value_function = self.feature_weights  # For compatibility

    def get_state_value(self, state: GameState) -> float:
        """
        Estimate value of a state using learned weights.

        V(s) = w^T φ(s)  (linear function approximation)

        Args:
            state: Game state

        Returns:
            Estimated state value
        """
        if self.use_extended_features:
            features = StateFeatures.extract_extended_features(state)
        else:
            features = StateFeatures.extract_features(state)

        return np.dot(self.feature_weights, features)

    def get_action_value(self, state: GameState, move: Move) -> float:
        """
        Estimate Q(s,a) by looking at V(s') after taking action.

        Q(s,a) ≈ r(s,a) + γV(s')

        Args:
            state: Current state
            move: Move to evaluate

        Returns:
            Estimated action value
        """
        # Simulate taking the move
        next_state = apply_move(state, move)

        # Immediate reward (change in score)
        immediate_reward = next_state.score - state.score

        # Future value (bootstrapped)
        future_value = self.get_state_value(next_state)

        # Combine (using γ=0.95 discount)
        gamma = 0.95
        return immediate_reward + gamma * future_value

    def save(self, path: str) -> None:
        """Save policy to disk."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        data = {
            'name': self.name,
            'feature_weights': self.feature_weights.tolist(),
            'use_extended_features': self.use_extended_features,
            'metadata': self.metadata
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load policy from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.name = data['name']
        self.feature_weights = np.array(data['feature_weights'], dtype=np.float32)
        self.use_extended_features = data.get('use_extended_features', False)
        self.metadata = data.get('metadata', {})
        self.value_function = self.feature_weights


class TDLearner(Solver):
    """
    TD(λ) Learning algorithm with linear function approximation.

    This trains a value function V(s) = w^T φ(s) using temporal difference learning.
    """

    def __init__(self,
                 alpha: float = 0.01,
                 gamma: float = 0.95,
                 lambda_: float = 0.0,
                 epsilon: float = 0.1,
                 use_extended_features: bool = False,
                 name: str = "TD-Learner"):
        """
        Initialize TD learner.

        Args:
            alpha: Learning rate (step size)
            gamma: Discount factor
            lambda_: Eligibility trace decay (0=TD(0), 1=Monte Carlo)
            epsilon: Exploration rate for ε-greedy policy
            use_extended_features: Use extended feature set
            name: Learner name
        """
        super().__init__(name)

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.use_extended_features = use_extended_features

        # Initialize weights
        feature_dim = StateFeatures.TOTAL_FEATURES
        self.weights = np.zeros(feature_dim, dtype=np.float32)

        # Eligibility traces (reset each episode)
        self.eligibility = np.zeros_like(self.weights)

        # Training statistics
        self.episode_count = 0
        self.total_updates = 0

        # Create policy from current weights
        self.policy = TDPolicy(
            feature_weights=self.weights,
            use_extended_features=use_extended_features
        )

    def get_features(self, state: GameState) -> np.ndarray:
        """Extract features from state."""
        if self.use_extended_features:
            return StateFeatures.extract_extended_features(state)
        else:
            return StateFeatures.extract_features(state)

    def get_value(self, state: GameState) -> float:
        """Get current value estimate for state."""
        features = self.get_features(state)
        return np.dot(self.weights, features)

    def choose_move(self, game: Game) -> Optional[Move]:
        """
        Choose move using ε-greedy policy during training.

        Args:
            game: Current game

        Returns:
            Selected move
        """
        valid_moves = get_valid_moves(game.state)
        if not valid_moves:
            return None

        # ε-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random move
            return np.random.choice(valid_moves)
        else:
            # Exploit: use learned policy
            return self.policy.choose_action(game.state, valid_moves)

    def train_episode(self, game: Game, verbose: bool = False) -> Tuple[float, int]:
        """
        Train on one episode using TD(λ).

        Args:
            game: Game to play (will be reset)
            verbose: Print training progress

        Returns:
            (final_score, moves_made)
        """
        # Reset eligibility traces
        self.eligibility = np.zeros_like(self.weights)

        # Play episode
        moves_made = 0
        max_moves = 500

        state_history = []

        while moves_made < max_moves:
            current_state = game.state.copy()
            current_value = self.get_value(current_state)
            current_features = self.get_features(current_state)

            # Choose and make move
            move = self.choose_move(game)
            if not move:
                break

            success = game.make_move(move)
            if not success:
                break

            moves_made += 1
            next_state = game.state
            next_value = self.get_value(next_state)

            # Compute TD error
            reward = next_state.score - current_state.score
            td_error = reward + self.gamma * next_value - current_value

            # Update eligibility trace
            self.eligibility = self.gamma * self.lambda_ * self.eligibility + current_features

            # Update weights
            self.weights += self.alpha * td_error * self.eligibility

            self.total_updates += 1

            state_history.append((current_state, reward, next_state))

        # Episode complete
        self.episode_count += 1

        # Update policy with new weights
        self.policy.feature_weights = self.weights.copy()

        final_score = game.state.score
        if verbose:
            foundation = game.state.get_foundation_count()
            print(f"  Episode {self.episode_count}: Score ${final_score}, "
                  f"Foundation {foundation}/52, Moves {moves_made}")

        return final_score, moves_made

    def train(self, num_episodes: int, start_seed: int = 0, verbose: bool = True) -> TDPolicy:
        """
        Train the value function over multiple episodes.

        Args:
            num_episodes: Number of games to train on
            start_seed: Starting random seed
            verbose: Print progress

        Returns:
            Trained policy
        """
        if verbose:
            print(f"Training TD(λ={self.lambda_}) for {num_episodes} episodes...")
            print(f"  α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
            print()

        scores = []
        foundations = []

        for episode in range(num_episodes):
            # Create new game
            game = Game(seed=start_seed + episode)
            game.deal()

            # Train on this episode
            score, moves = self.train_episode(game, verbose=False)

            scores.append(score)
            foundations.append(game.state.get_foundation_count())

            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                recent_avg_score = np.mean(scores[-100:])
                recent_avg_found = np.mean(foundations[-100:])
                print(f"  Episodes {episode+1:5d}/{num_episodes}: "
                      f"Avg Score ${recent_avg_score:6.1f}, "
                      f"Avg Foundation {recent_avg_found:4.1f}/52")

        if verbose:
            print(f"\nTraining complete!")
            print(f"  Total episodes: {self.episode_count}")
            print(f"  Total updates: {self.total_updates}")
            print(f"  Final avg score (last 100): ${np.mean(scores[-100:]):.1f}")
            print(f"  Final avg foundation: {np.mean(foundations[-100:]):.1f}/52")

        # Return final policy
        return self.policy

    def get_policy(self) -> TDPolicy:
        """Get current policy."""
        # Make sure policy has latest weights
        self.policy.feature_weights = self.weights.copy()
        return self.policy


def train_td_policy(num_episodes: int = 1000,
                   alpha: float = 0.01,
                   gamma: float = 0.95,
                   lambda_: float = 0.0,
                   save_path: Optional[str] = None) -> TDPolicy:
    """
    Convenience function to train a TD policy.

    Args:
        num_episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        lambda_: Eligibility trace parameter
        save_path: Where to save trained policy (optional)

    Returns:
        Trained policy
    """
    learner = TDLearner(alpha=alpha, gamma=gamma, lambda_=lambda_)
    policy = learner.train(num_episodes=num_episodes, verbose=True)

    if save_path:
        policy.save(save_path)
        print(f"\nPolicy saved to: {save_path}")

    return policy

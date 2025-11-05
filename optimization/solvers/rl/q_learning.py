"""
Q-Learning for Vegas Solitaire.

Q-Learning is a model-free RL algorithm that learns the action-value function
Q(s,a) directly, rather than learning V(s) and deriving a policy.

Key concepts:
- Off-policy learning: Learns optimal policy while following ε-greedy
- Q-value update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- Function approximation: Q(s,a) ≈ w^T [φ(s), φ(a)]

References:
- Watkins & Dayan (1992), "Q-learning"
- Sutton & Barto, "Reinforcement Learning", Chapter 6
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

from game.core.game import Game
from game.core.state import GameState
from game.core.moves import Move, MoveType
from game.core.rules import apply_move, get_valid_moves
from optimization.policies.base import ValueBasedPolicy
from optimization.features.state_features import StateFeatures
from optimization.solvers.base import Solver


class QPolicy(ValueBasedPolicy):
    """
    Policy that uses learned Q-function.

    Acts greedily: choose action with highest Q-value.
    """

    def __init__(self,
                 name: str = "Q-Learned",
                 feature_weights: Optional[np.ndarray] = None,
                 use_extended_features: bool = False):
        """
        Initialize Q policy.

        Args:
            name: Policy name
            feature_weights: Learned weights for Q-function
            use_extended_features: Use extended feature set
        """
        super().__init__(name)
        self.use_extended_features = use_extended_features

        # Initialize weights for Q-function
        if feature_weights is None:
            # State features + action features
            state_dim = StateFeatures.TOTAL_FEATURES
            action_dim = 10  # Action encoding dimension
            total_dim = state_dim + action_dim
            self.feature_weights = np.zeros(total_dim, dtype=np.float32)
        else:
            self.feature_weights = feature_weights.copy()

        self.value_function = self.feature_weights  # For compatibility

    def encode_action(self, move: Move) -> np.ndarray:
        """
        Encode a move as a feature vector.

        Features:
        - Move type (one-hot)
        - Source/destination info
        - Card count

        Args:
            move: Move to encode

        Returns:
            Action feature vector (10 dimensions)
        """
        features = np.zeros(10, dtype=np.float32)

        # Move type (one-hot, 5 types)
        move_type_map = {
            MoveType.TABLEAU_TO_FOUNDATION: 0,
            MoveType.TABLEAU_TO_TABLEAU: 1,
            MoveType.WASTE_TO_FOUNDATION: 2,
            MoveType.WASTE_TO_TABLEAU: 3,
            MoveType.DRAW: 4,
        }

        move_type_idx = move_type_map.get(move.move_type, 4)
        features[move_type_idx] = 1.0

        # Source column (normalized to 0-1)
        if hasattr(move, 'from_col') and move.from_col is not None:
            features[5] = move.from_col / 7.0

        # Destination column (normalized to 0-1)
        if hasattr(move, 'to_col') and move.to_col is not None:
            features[6] = move.to_col / 7.0

        # Card count (normalized)
        if hasattr(move, 'card_count') and move.card_count is not None:
            features[7] = move.card_count / 13.0

        # Foundation index
        if hasattr(move, 'foundation_idx') and move.foundation_idx is not None:
            features[8] = move.foundation_idx / 4.0

        # Is this a revealing move? (flips hidden card)
        features[9] = 1.0 if getattr(move, 'reveals_card', False) else 0.0

        return features

    def get_state_action_features(self, state: GameState, move: Move) -> np.ndarray:
        """
        Get combined state-action features for Q(s,a).

        Args:
            state: Game state
            move: Action

        Returns:
            Combined feature vector
        """
        if self.use_extended_features:
            state_features = StateFeatures.extract_extended_features(state)
        else:
            state_features = StateFeatures.extract_features(state)

        action_features = self.encode_action(move)

        # Concatenate
        return np.concatenate([state_features, action_features])

    def get_action_value(self, state: GameState, move: Move) -> float:
        """
        Compute Q(s,a) using linear function approximation.

        Q(s,a) = w^T [φ(s), φ(a)]

        Args:
            state: Current state
            move: Action to evaluate

        Returns:
            Q-value
        """
        features = self.get_state_action_features(state, move)

        # Ensure feature vector matches weight dimension
        if len(features) != len(self.feature_weights):
            # Pad or truncate as needed
            if len(features) < len(self.feature_weights):
                features = np.pad(features, (0, len(self.feature_weights) - len(features)))
            else:
                features = features[:len(self.feature_weights)]

        return np.dot(self.feature_weights, features)

    def get_state_value(self, state: GameState) -> float:
        """
        Estimate V(s) = max_a Q(s,a).

        Args:
            state: Game state

        Returns:
            Estimated state value
        """
        valid_moves = get_valid_moves(state)
        if not valid_moves:
            return 0.0

        return max(self.get_action_value(state, move) for move in valid_moves)

    def choose_action(self, state: GameState, valid_moves: List[Move]) -> Move:
        """
        Choose best action greedily based on Q-values.

        Args:
            state: Current state
            valid_moves: Available actions

        Returns:
            Best move
        """
        if not valid_moves:
            return None

        # Compute Q-value for each action
        q_values = [self.get_action_value(state, move) for move in valid_moves]

        # Choose action with highest Q-value
        best_idx = np.argmax(q_values)
        return valid_moves[best_idx]

    def save(self, path: str) -> None:
        """Save policy to disk."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

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


class QLearner(Solver):
    """
    Q-Learning algorithm with linear function approximation.

    Learns Q(s,a) = w^T [φ(s), φ(a)] using Q-learning update rule.
    """

    def __init__(self,
                 alpha: float = 0.01,
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 use_extended_features: bool = False,
                 name: str = "Q-Learner"):
        """
        Initialize Q-learner.

        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            use_extended_features: Use extended feature set
            name: Learner name
        """
        super().__init__(name)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_extended_features = use_extended_features

        # Initialize Q-function weights
        state_dim = StateFeatures.TOTAL_FEATURES
        action_dim = 10
        total_dim = state_dim + action_dim
        self.weights = np.zeros(total_dim, dtype=np.float32)

        # Training statistics
        self.episode_count = 0
        self.total_updates = 0

        # Create policy
        self.policy = QPolicy(
            feature_weights=self.weights,
            use_extended_features=use_extended_features
        )

    def get_q_value(self, state: GameState, move: Move) -> float:
        """
        Get Q(s,a) for current weights.

        Args:
            state: Game state
            move: Action

        Returns:
            Q-value
        """
        return self.policy.get_action_value(state, move)

    def get_max_q_value(self, state: GameState) -> float:
        """
        Get max_a Q(s,a) for a state.

        Args:
            state: Game state

        Returns:
            Maximum Q-value over actions
        """
        valid_moves = get_valid_moves(state)
        if not valid_moves:
            return 0.0

        return max(self.get_q_value(state, move) for move in valid_moves)

    def choose_move(self, game: Game) -> Optional[Move]:
        """
        Choose move using ε-greedy policy.

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
            # Exploit: greedy w.r.t Q-function
            return self.policy.choose_action(game.state, valid_moves)

    def train_episode(self, game: Game, verbose: bool = False) -> Tuple[float, int]:
        """
        Train on one episode using Q-learning.

        Q-learning update:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            game: Game to play
            verbose: Print progress

        Returns:
            (final_score, moves_made)
        """
        moves_made = 0
        max_moves = 500

        while moves_made < max_moves:
            current_state = game.state.copy()

            # Choose action
            move = self.choose_move(game)
            if not move:
                break

            # Current Q-value
            current_q = self.get_q_value(current_state, move)

            # Take action
            success = game.make_move(move)
            if not success:
                break

            moves_made += 1
            next_state = game.state

            # Reward (change in score)
            reward = next_state.score - current_state.score

            # Max Q-value for next state
            max_next_q = self.get_max_q_value(next_state)

            # Q-learning update: TD error
            td_error = reward + self.gamma * max_next_q - current_q

            # Update weights: w ← w + α * td_error * ∇Q(s,a)
            # For linear approximation: ∇Q(s,a) = φ(s,a)
            features = self.policy.get_state_action_features(current_state, move)

            # Ensure feature dimension matches weights
            if len(features) != len(self.weights):
                if len(features) < len(self.weights):
                    features = np.pad(features, (0, len(self.weights) - len(features)))
                else:
                    features = features[:len(self.weights)]

            self.weights += self.alpha * td_error * features
            self.total_updates += 1

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

    def train(self, num_episodes: int, start_seed: int = 0, verbose: bool = True) -> QPolicy:
        """
        Train Q-function over multiple episodes.

        Args:
            num_episodes: Number of training episodes
            start_seed: Starting random seed
            verbose: Print progress

        Returns:
            Trained policy
        """
        if verbose:
            print(f"Training Q-Learning for {num_episodes} episodes...")
            print(f"  α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
            print()

        scores = []
        foundations = []

        for episode in range(num_episodes):
            # Create new game
            game = Game(seed=start_seed + episode)
            game.deal()

            # Train on episode
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

        return self.policy

    def get_policy(self) -> QPolicy:
        """Get current policy."""
        self.policy.feature_weights = self.weights.copy()
        return self.policy


def train_q_policy(num_episodes: int = 1000,
                   alpha: float = 0.01,
                   gamma: float = 0.95,
                   epsilon: float = 0.1,
                   save_path: Optional[str] = None) -> QPolicy:
    """
    Convenience function to train a Q-learning policy.

    Args:
        num_episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        save_path: Where to save trained policy (optional)

    Returns:
        Trained Q-policy
    """
    learner = QLearner(alpha=alpha, gamma=gamma, epsilon=epsilon)
    policy = learner.train(num_episodes=num_episodes, verbose=True)

    if save_path:
        policy.save(save_path)
        print(f"\nPolicy saved to: {save_path}")

    return policy


if __name__ == "__main__":
    # Quick test
    print("Training Q-Learning policy...")
    policy = train_q_policy(num_episodes=500, alpha=0.005, epsilon=0.15)

    # Test on a few games
    print("\n" + "=" * 70)
    print("Testing trained policy...")
    print("=" * 70)

    from optimization.solvers.heuristic_solver import HeuristicSolver

    test_scores = []
    for seed in range(10):
        game = Game(seed=1000 + seed)
        game.deal()

        # Play with Q-policy
        solver = HeuristicSolver(policy=policy)
        result = solver.solve(game, max_moves=500)

        test_scores.append(result['score'])
        print(f"Game {seed+1}: Score ${result['score']}, "
              f"Foundation {result['foundation_count']}/52")

    print(f"\nAverage score: ${np.mean(test_scores):.1f}")

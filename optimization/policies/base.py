"""
Base classes for learned policies.

Policies are reusable decision-makers that can be trained once and then
used to play many games. This separates training (expensive) from inference (fast).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import numpy as np

from game.core.state import GameState
from game.core.moves import Move


class Policy(ABC):
    """
    Base class for game-playing policies.

    A policy maps from game states to actions (moves). Policies can be:
    - Hand-crafted (like our heuristic)
    - Learned from experience (RL)
    - Hybrid (combining both)

    All policies must be saveable/loadable for reuse.
    """

    def __init__(self, name: str):
        """
        Initialize policy.

        Args:
            name: Human-readable name for this policy
        """
        self.name = name
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def choose_action(self, state: GameState, valid_moves: List[Move]) -> Optional[Move]:
        """
        Choose an action given the current state and valid moves.

        Args:
            state: Current game state
            valid_moves: List of legal moves in this state

        Returns:
            Chosen move, or None if no move should be made
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save policy to disk for later reuse.

        Args:
            path: File path to save to (will create parent dirs if needed)
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load policy from disk.

        Args:
            path: File path to load from

        Raises:
            FileNotFoundError: If policy file doesn't exist
            ValueError: If file format is invalid
        """
        pass

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata about this policy (training time, hyperparams, etc)."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata."""
        return self.metadata.get(key, default)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ValueBasedPolicy(Policy):
    """
    Policy that derives actions from a learned value function.

    This is the base class for Q-Learning, SARSA, TD-Learning, etc.
    These algorithms learn V(s) or Q(s,a) and then act greedily.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.value_function: Optional[Any] = None  # Subclasses define type

    @abstractmethod
    def get_state_value(self, state: GameState) -> float:
        """
        Get estimated value of a state.

        Args:
            state: Game state to evaluate

        Returns:
            Estimated value (higher is better)
        """
        pass

    @abstractmethod
    def get_action_value(self, state: GameState, move: Move) -> float:
        """
        Get estimated value of taking an action in a state.

        Args:
            state: Current game state
            move: Move to evaluate

        Returns:
            Estimated Q-value of (state, move) pair
        """
        pass

    def choose_action(self, state: GameState, valid_moves: List[Move]) -> Optional[Move]:
        """
        Choose action with highest Q-value (greedy policy).

        Args:
            state: Current game state
            valid_moves: Legal moves

        Returns:
            Best move according to learned values
        """
        if not valid_moves:
            return None

        # Evaluate all valid moves
        move_values = [(move, self.get_action_value(state, move)) for move in valid_moves]

        # Return move with highest value
        best_move, best_value = max(move_values, key=lambda x: x[1])
        return best_move


class DirectPolicy(Policy):
    """
    Policy that directly maps states to action probabilities.

    This is the base class for Policy Gradient methods, Actor-Critic, etc.
    These algorithms learn π(a|s) directly without an explicit value function.
    """

    @abstractmethod
    def get_action_probabilities(self, state: GameState, valid_moves: List[Move]) -> Dict[Move, float]:
        """
        Get probability distribution over actions.

        Args:
            state: Current game state
            valid_moves: Legal moves

        Returns:
            Dictionary mapping each valid move to its probability
        """
        pass

    def choose_action(self, state: GameState, valid_moves: List[Move]) -> Optional[Move]:
        """
        Sample action from learned policy distribution.

        Args:
            state: Current game state
            valid_moves: Legal moves

        Returns:
            Sampled move (stochastic during training, greedy for deployment)
        """
        if not valid_moves:
            return None

        # Get action probabilities
        action_probs = self.get_action_probabilities(state, valid_moves)

        # Sample from distribution (for stochastic policy)
        # Or take argmax (for deterministic deployment)
        moves = list(action_probs.keys())
        probs = list(action_probs.values())

        # Normalize to ensure sum = 1.0
        prob_sum = sum(probs)
        if prob_sum > 0:
            probs = [p / prob_sum for p in probs]
        else:
            # Uniform if all zeros
            probs = [1.0 / len(moves)] * len(moves)

        # Sample
        chosen_idx = np.random.choice(len(moves), p=probs)
        return moves[chosen_idx]


class HybridPolicy(Policy):
    """
    Policy that combines multiple strategies.

    Examples:
    - Heuristic + learned value function
    - MCTS with learned rollout policy
    - Ensemble of multiple learned policies
    """

    def __init__(self, name: str, policies: List[Policy], weights: Optional[List[float]] = None):
        """
        Initialize hybrid policy.

        Args:
            name: Policy name
            policies: List of component policies
            weights: Optional weights for combining policies (must sum to 1.0)
        """
        super().__init__(name)
        self.policies = policies

        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(policies)] * len(policies)
        else:
            assert len(weights) == len(policies), "Weights must match policies"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
            self.weights = weights

    def choose_action(self, state: GameState, valid_moves: List[Move]) -> Optional[Move]:
        """
        Combine multiple policies to choose action.

        Default: Weighted voting (each policy votes, weighted by weight).
        Subclasses can override for more sophisticated combination.
        """
        if not valid_moves:
            return None

        # Get vote from each policy
        move_scores: Dict[Move, float] = {move: 0.0 for move in valid_moves}

        for policy, weight in zip(self.policies, self.weights):
            chosen = policy.choose_action(state, valid_moves)
            if chosen:
                move_scores[chosen] += weight

        # Return move with highest weighted vote
        best_move = max(move_scores.keys(), key=lambda m: move_scores[m])
        return best_move

    def save(self, path: str) -> None:
        """Save hybrid policy (saves all component policies)."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            'name': self.name,
            'policy_types': [p.__class__.__name__ for p in self.policies],
            'weights': self.weights,
            'metadata': self.metadata
        }

        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save each component policy
        for i, policy in enumerate(self.policies):
            policy_path = path_obj.parent / f"{path_obj.stem}_policy{i}{path_obj.suffix}"
            policy.save(str(policy_path))

    def load(self, path: str) -> None:
        """Load hybrid policy (loads all component policies)."""
        # Subclasses should implement based on their specific needs
        raise NotImplementedError("Hybrid policy loading requires specific implementation")

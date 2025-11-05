"""
Beam Search solver for Vegas Solitaire.

Beam search is a memory-efficient best-first search that keeps only the
top K states at each depth. It's faster than full breadth-first search
and uses less memory than best-first search.

Key parameters:
- beam_width: How many states to keep at each depth (K)
- max_depth: Maximum search depth
- evaluation_fn: Function to score states (higher = better)
"""

from typing import List, Tuple, Callable, Optional
import heapq
from dataclasses import dataclass, field

from game.core.game import Game
from game.core.state import GameState
from game.core.moves import Move
from game.core.rules import get_valid_moves, apply_move
from optimization.solvers.base import Solver
from optimization.features.state_features import StateFeatures
import numpy as np


@dataclass(order=True)
class BeamNode:
    """
    Node in the beam search.

    Ordered by score for use in priority queue.
    """
    score: float = field(compare=True)
    state: GameState = field(compare=False)
    path: List[Move] = field(default_factory=list, compare=False)
    depth: int = field(default=0, compare=False)


class BeamSearch(Solver):
    """
    Beam Search solver with configurable evaluation function.

    Beam search explores the game tree level by level, keeping only
    the K best states at each level.
    """

    def __init__(self,
                 beam_width: int = 100,
                 max_depth: int = 50,
                 evaluation_fn: Optional[Callable[[GameState], float]] = None,
                 name: Optional[str] = None):
        """
        Initialize beam search solver.

        Args:
            beam_width: Number of states to keep at each depth
            max_depth: Maximum search depth
            evaluation_fn: Function to score states (defaults to simple heuristic)
            name: Solver name (auto-generated if None)
        """
        if name is None:
            name = f"BeamSearch(w={beam_width},d={max_depth})"
        super().__init__(name=name)

        self.beam_width = beam_width
        self.max_depth = max_depth

        # Use provided evaluation function or default
        if evaluation_fn is None:
            self.evaluation_fn = self._default_evaluation
        else:
            self.evaluation_fn = evaluation_fn

        # Statistics
        self.nodes_expanded = 0
        self.max_beam_size = 0

    def _default_evaluation(self, state: GameState) -> float:
        """
        Default evaluation function: score states higher is better.

        Uses simple heuristic:
        - Foundation cards (most important)
        - Game score
        - Hidden cards revealed
        - Empty columns
        """
        score = 0.0

        # Foundation cards (highest priority) - each worth 100
        foundation_count = state.get_foundation_count()
        score += foundation_count * 100

        # Winning is extremely valuable
        if state.is_winning():
            score += 10000

        # Game score (Vegas scoring) - normalize
        score += (state.score + 52) / 10.0  # Roughly 0-25 range

        # Revealed cards (each worth 5)
        total_visible = sum(len(state.get_tableau_visible_cards(col)) for col in range(7))
        score += total_visible * 5

        # Empty columns (each worth 20) - valuable for Kings
        empty_cols = sum(1 for col in state.tableau if len(col) == 0)
        score += empty_cols * 20

        # Penalty for hidden cards
        total_hidden = sum(state.tableau_hidden)
        score -= total_hidden * 2

        # Penalty for many passes through deck
        score -= state.passes_through_deck * 10

        return score

    def choose_move(self, game: Game) -> Optional[Move]:
        """
        Choose the best move using beam search.

        Performs beam search to look ahead and find the best move sequence,
        then returns the first move of the best path.

        Args:
            game: Current game state

        Returns:
            Best move according to beam search, or None if no moves
        """
        self.nodes_expanded = 0
        self.max_beam_size = 0

        # Get initial valid moves
        valid_moves = get_valid_moves(game.state)
        if not valid_moves:
            return None

        # If only one move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]

        # Initialize beam with root node
        root_score = self.evaluation_fn(game.state)
        root = BeamNode(score=root_score, state=game.state.copy(), path=[], depth=0)

        current_beam = [root]

        # Beam search main loop
        best_node = root
        best_score = root_score

        for depth in range(self.max_depth):
            next_beam = []

            # Expand all nodes in current beam
            for node in current_beam:
                self.nodes_expanded += 1

                # Get valid moves from this state
                moves = get_valid_moves(node.state)

                if not moves:
                    # Terminal node (no moves) - keep it for evaluation
                    next_beam.append(node)
                    continue

                # Generate all successor states
                for move in moves:
                    new_state = apply_move(node.state, move)
                    new_path = node.path + [move]
                    new_score = self.evaluation_fn(new_state)

                    new_node = BeamNode(
                        score=new_score,
                        state=new_state,
                        path=new_path,
                        depth=depth + 1
                    )

                    next_beam.append(new_node)

                    # Track best node seen
                    if new_score > best_score:
                        best_score = new_score
                        best_node = new_node

            # Check if we found a winning state
            for node in next_beam:
                if node.state.is_winning():
                    # Found a win! Return first move of winning path
                    return node.path[0] if node.path else None

            # Prune to beam width - keep only top K nodes
            if len(next_beam) > self.beam_width:
                # Use heap to efficiently find top K
                next_beam = heapq.nlargest(self.beam_width, next_beam, key=lambda n: n.score)

            self.max_beam_size = max(self.max_beam_size, len(next_beam))

            # If beam is empty, we've exhausted all paths
            if not next_beam:
                break

            current_beam = next_beam

        # Didn't find a clear win, return first move of best path found
        if best_node.path:
            return best_node.path[0]

        # Fallback to first valid move
        return valid_moves[0]

    def get_stats(self) -> dict:
        """Get statistics from last search."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'max_beam_size': self.max_beam_size
        }


class FeatureBasedBeamSearch(BeamSearch):
    """
    Beam search using learned feature weights.

    This allows combining beam search with learned evaluation functions.
    """

    def __init__(self,
                 beam_width: int = 100,
                 max_depth: int = 50,
                 feature_weights: Optional[np.ndarray] = None):
        """
        Initialize feature-based beam search.

        Args:
            beam_width: Number of states to keep
            max_depth: Maximum search depth
            feature_weights: Learned weights for features (if None, uses default)
        """
        # Use feature-based evaluation
        def feature_eval(state: GameState) -> float:
            features = StateFeatures.extract_features(state)
            if feature_weights is None:
                # Default weights - favor foundation cards and visible cards
                weights = np.zeros(StateFeatures.TOTAL_FEATURES)
                weights[0:4] = 100.0  # Foundation counts
                weights[4:11] = 5.0   # Visible depth
                weights[11:18] = -2.0 # Hidden (negative)
                weights[18] = -10.0   # Stock
                weights[19] = -5.0    # Waste
                weights[20] = -20.0   # Passes
                weights[21] = 20.0    # Empty columns
                weights[22] = 10.0    # Total visible
                weights[23] = -5.0    # Total hidden
            else:
                weights = feature_weights

            return np.dot(features, weights)

        super().__init__(
            beam_width=beam_width,
            max_depth=max_depth,
            evaluation_fn=feature_eval,
            name=f"FeatureBeam(w={beam_width},d={max_depth})"
        )
        self.feature_weights = feature_weights


def create_beam_search_variants() -> List[BeamSearch]:
    """
    Create multiple beam search variants with different parameters.

    Useful for benchmarking to find best configuration.

    Returns:
        List of beam search solvers with different configurations
    """
    variants = []

    # Vary beam width
    for width in [10, 50, 100, 200]:
        variants.append(BeamSearch(beam_width=width, max_depth=30))

    # Vary depth
    for depth in [20, 40, 60]:
        variants.append(BeamSearch(beam_width=100, max_depth=depth))

    # Feature-based variants
    variants.append(FeatureBasedBeamSearch(beam_width=100, max_depth=30))

    return variants

"""
Monte Carlo Tree Search (MCTS) solver for Vegas Solitaire.

MCTS is a best-first search algorithm that uses random sampling to build
a search tree. It balances exploration and exploitation using the UCB1 formula.

The algorithm has four phases:
1. Selection: Navigate tree using UCB1
2. Expansion: Add new node to tree
3. Simulation: Random playout to terminal state
4. Backpropagation: Update statistics up the tree
"""

from typing import Optional, List, Dict, Any
import math
import random
import time
from game.core.game import Game
from game.core.moves import Move
from game.core.state import GameState
from optimization.solvers.base import Solver


class MCTSNode:
    """
    Node in the MCTS search tree.

    Each node represents a game state and tracks statistics about
    the rewards obtained from this state.

    Attributes:
        state: Game state at this node
        parent: Parent node (None for root)
        move: Move that led to this node
        children: Child nodes
        visits: Number of times this node was visited
        total_reward: Sum of rewards from simulations through this node
        untried_moves: Moves not yet explored from this state
    """

    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None,
                 move: Optional[Move] = None):
        """
        Initialize MCTS node.

        Args:
            state: Game state at this node
            parent: Parent node
            move: Move that led to this node
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves: List[Move] = []

    def get_average_reward(self) -> float:
        """
        Get average reward from this node.

        Returns:
            Average reward, or 0 if never visited
        """
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def is_fully_expanded(self) -> bool:
        """Check if all moves from this node have been tried."""
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.state.is_winning() or len(self.untried_moves) == 0 and len(self.children) == 0

    def ucb1(self, exploration_constant: float = 1.41) -> float:
        """
        Calculate UCB1 (Upper Confidence Bound) value.

        UCB1 balances exploitation (average reward) and exploration (rarely visited).

        Args:
            exploration_constant: Controls exploration vs exploitation (√2 is common)

        Returns:
            UCB1 value
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have infinite value

        if self.parent is None:
            return self.get_average_reward()

        exploitation = self.get_average_reward()
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.41) -> 'MCTSNode':
        """
        Select the best child using UCB1.

        Args:
            exploration_constant: UCB1 exploration parameter

        Returns:
            Child node with highest UCB1 value
        """
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))

    def most_visited_child(self) -> 'MCTSNode':
        """
        Get the most visited child (for final move selection).

        Returns:
            Child with most visits
        """
        return max(self.children, key=lambda c: c.visits)


class MCTSSolver(Solver):
    """
    MCTS solver for Vegas Solitaire.

    Uses Monte Carlo Tree Search to find good moves by building a search tree
    and using random simulations to evaluate positions.

    Attributes:
        simulations_per_move: Number of MCTS iterations per move
        exploration_constant: UCB1 exploration parameter
        rng: Random number generator
        max_simulation_depth: Maximum depth for simulations
    """

    def __init__(self, simulations_per_move: int = 1000,
                 exploration_constant: float = 1.41,
                 max_simulation_depth: int = 500,
                 seed: Optional[int] = None):
        """
        Initialize MCTS solver.

        Args:
            simulations_per_move: Number of simulations to run per move
            exploration_constant: UCB1 parameter (√2 ≈ 1.41 is standard)
            max_simulation_depth: Maximum moves in simulation rollout
            seed: Random seed for reproducibility
        """
        super().__init__(name=f"MCTS({simulations_per_move})")
        self.simulations_per_move = simulations_per_move
        self.exploration_constant = exploration_constant
        self.max_simulation_depth = max_simulation_depth
        self.rng = random.Random(seed)

    def choose_move(self, game: Game) -> Optional[Move]:
        """
        Choose the best move using MCTS.

        Args:
            game: Current game state

        Returns:
            Best move according to MCTS, or None if no moves
        """
        # Create root node
        root = MCTSNode(game.state.copy())

        # Get valid moves
        from game.core.rules import get_valid_moves
        root.untried_moves = get_valid_moves(game.state)

        if not root.untried_moves:
            return None

        # Run MCTS iterations
        for _ in range(self.simulations_per_move):
            # 1. Selection: Navigate to promising leaf node
            node = self._select(root)

            # 2. Expansion: Add a child node
            if not node.is_terminal() and node.untried_moves:
                node = self._expand(node)

            # 3. Simulation: Run random playout
            reward = self._simulate(node)

            # 4. Backpropagation: Update statistics
            self._backpropagate(node, reward)

        # Choose best move (most visited child)
        if not root.children:
            return root.untried_moves[0] if root.untried_moves else None

        best_child = root.most_visited_child()
        return best_child.move

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Navigate to a leaf node using UCB1.

        Args:
            node: Starting node (usually root)

        Returns:
            Selected leaf node
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node

            if node.children:
                node = node.best_child(self.exploration_constant)
            else:
                return node

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add a new child node to the tree.

        Args:
            node: Node to expand

        Returns:
            Newly created child node
        """
        if not node.untried_moves:
            return node

        # Choose a random untried move
        move = self.rng.choice(node.untried_moves)
        node.untried_moves.remove(move)

        # Apply move to get new state
        from game.core.rules import apply_move
        new_state = apply_move(node.state, move)

        # Create child node
        child = MCTSNode(new_state, parent=node, move=move)

        # Get valid moves for child
        from game.core.rules import get_valid_moves
        child.untried_moves = get_valid_moves(child.state)

        node.children.append(child)

        return child

    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: Random playout from this node to terminal state.

        Args:
            node: Node to simulate from

        Returns:
            Reward from simulation (normalized score)
        """
        state = node.state.copy()

        from game.core.rules import get_valid_moves, apply_move

        moves = 0
        while moves < self.max_simulation_depth:
            # Check if game is won
            if state.is_winning():
                # Winning is very valuable
                # Normalize score to roughly 0-1 range
                # Win: score is usually $100-$200
                return (state.score + 52) / 200.0

            # Get valid moves
            valid_moves = get_valid_moves(state)

            if not valid_moves:
                break

            # Choose random move
            move = self.rng.choice(valid_moves)

            # Apply move
            state = apply_move(state, move)
            moves += 1

        # Game ended without winning
        # Normalize score: -52 (worst) to 0 (break even) to positive
        # Map to roughly 0-1 range where 0.5 is break even
        normalized_score = (state.score + 52) / 200.0
        return max(0.0, normalized_score)  # Clamp to non-negative

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase: Update statistics up the tree.

        Args:
            node: Leaf node to backpropagate from
            reward: Reward from simulation
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

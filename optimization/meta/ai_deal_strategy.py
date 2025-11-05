"""
AI-powered deal selection strategies for multi-deal mode.

This module integrates learned policies (Q-Learning, TD Learning) with
heuristic strategies to make optimal decisions about:
1. Which deal to accept from rerolls
2. Whether to reroll or play
3. Risk management based on bankroll

Combines:
- Learned value functions (Q-Learning, TD Learning)
- Heuristic deal quality evaluation
- Search algorithms (Beam Search, MCTS)
- Strategic insights from domain knowledge
"""

from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from pathlib import Path

from game.core.game import Game
from game.core.state import GameState
from optimization.meta.deal_evaluator import evaluate_deal_quality, DealComparator
from optimization.meta.multideal_mode import DealSelector, DealAcceptor
from optimization.policies.base import Policy
from optimization.solvers.base import Solver


class LearnedDealSelector(DealSelector):
    """
    Deal selector that uses learned value functions.

    Uses Q-Learning or TD Learning policies to estimate deal value.
    """

    def __init__(self,
                 policy: Optional[Policy] = None,
                 solver: Optional[Solver] = None,
                 simulation_depth: int = 50):
        """
        Initialize learned deal selector.

        Args:
            policy: Learned policy (Q-Learning or TD Learning)
            solver: Solver to use for evaluating deals
            simulation_depth: How many moves to simulate for evaluation
        """
        super().__init__(strategy="learned")
        self.policy = policy
        self.solver = solver
        self.simulation_depth = simulation_depth

    def estimate_deal_value(self, game: Game) -> float:
        """
        Estimate expected value of a deal using learned policy.

        Options:
        1. Use policy's value function directly: V(s)
        2. Run short simulation to estimate outcome
        3. Combine with heuristic quality

        Args:
            game: Dealt game

        Returns:
            Estimated expected value (in dollars)
        """
        if self.policy is None:
            # Fallback to heuristic
            return evaluate_deal_quality(game.state)

        # Method 1: Use value function directly
        if hasattr(self.policy, 'get_state_value'):
            state_value = self.policy.get_state_value(game.state)

            # Convert value to expected score
            # Value function typically estimates future rewards
            # Scale to meaningful range (0-260 for Vegas scoring)
            estimated_score = state_value

            return estimated_score

        # Method 2: Run simulation
        elif self.solver is not None:
            # Make a copy and simulate
            sim_game = Game(seed=game.seed)
            sim_game.state = game.state.copy()

            moves_made = 0
            while moves_made < self.simulation_depth:
                move = self.solver.choose_move(sim_game)
                if not move or not sim_game.make_move(move):
                    break
                moves_made += 1

            # Return foundation count as value estimate
            return sim_game.state.get_foundation_count() * 5.0

        # Fallback: heuristic
        return evaluate_deal_quality(game.state)

    def select_from_rerolls(self,
                           deals: List[Game],
                           evaluator: Optional[Callable] = None) -> Tuple[int, float]:
        """
        Select best deal using learned policy.

        Args:
            deals: List of dealt games
            evaluator: Optional custom evaluator (ignored, uses learned value)

        Returns:
            (selected_index, value_estimate)
        """
        if not deals:
            return None, 0.0

        # Evaluate all deals using learned policy
        values = [self.estimate_deal_value(game) for game in deals]

        # Select best
        best_idx = np.argmax(values)
        return best_idx, values[best_idx]


class LearnedDealAcceptor(DealAcceptor):
    """
    Deal acceptor that uses learned value functions.

    Makes acceptance decisions by comparing expected value to cost.
    """

    def __init__(self,
                 policy: Optional[Policy] = None,
                 solver: Optional[Solver] = None,
                 risk_tolerance: float = 1.0):
        """
        Initialize learned deal acceptor.

        Args:
            policy: Learned policy
            solver: Solver for simulations
            risk_tolerance: How risk-averse (0.5=conservative, 2.0=aggressive)
        """
        super().__init__(strategy="value", value_function=None)
        self.policy = policy
        self.solver = solver
        self.risk_tolerance = risk_tolerance

        # Create value function
        if policy and hasattr(policy, 'get_state_value'):
            self.value_function = policy.get_state_value

    def estimate_expected_profit(self, state: GameState) -> float:
        """
        Estimate expected profit from playing this deal.

        Expected profit = E[cards_played] * $5 - $52

        Args:
            state: Deal state

        Returns:
            Expected profit in dollars
        """
        if self.value_function:
            # Use learned value function
            value = self.value_function(state)

            # Value function estimates future rewards
            # Convert to expected cards played (0-52)
            # Assume value is roughly proportional to cards
            expected_cards = min(52, max(0, value / 5.0))

            expected_score = expected_cards * 5.0
            expected_profit = expected_score - 52.0

            return expected_profit
        else:
            # Fallback to heuristic
            quality = evaluate_deal_quality(state)

            # Empirically calibrate quality to expected profit
            # Quality ~50 → ~20 cards → -$52 (loss)
            # Quality ~80 → ~30 cards → -$2 (small loss)
            # Quality ~100 → ~40 cards → +$48 (profit)
            expected_cards = quality * 0.4
            expected_score = expected_cards * 5.0
            expected_profit = expected_score - 52.0

            return expected_profit

    def should_accept(self,
                     state: GameState,
                     rerolls_remaining: int,
                     bankroll: float) -> Tuple[bool, float]:
        """
        Decide whether to accept deal using learned policy.

        Decision rule:
        - Accept if expected_profit > -reroll_cost
        - Adjust for rerolls remaining
        - Consider bankroll constraints

        Args:
            state: Deal state
            rerolls_remaining: Rerolls left
            bankroll: Current bankroll

        Returns:
            (should_accept, estimated_value)
        """
        expected_profit = self.estimate_expected_profit(state)

        # Quality for logging
        quality = evaluate_deal_quality(state)

        # Decision factors
        reroll_cost = 5.0  # Standard reroll cost

        # Baseline: accept if profit > -reroll_cost
        accept_threshold = -reroll_cost * self.risk_tolerance

        if rerolls_remaining == 0:
            # Must accept (no more rerolls)
            return True, quality

        if expected_profit >= accept_threshold:
            # Good enough deal
            return True, quality

        if rerolls_remaining == 1:
            # Last reroll - be more lenient
            if expected_profit >= accept_threshold * 0.7:
                return True, quality

        # Consider bankroll constraints
        if bankroll < 100:  # Low bankroll
            # Be more conservative - need wins
            if expected_profit >= 0:
                return True, quality
        elif bankroll > 500:  # High bankroll
            # Can afford to be pickier
            if expected_profit >= accept_threshold * 1.3:
                return True, quality

        # Otherwise, reject (reroll)
        return False, quality


class HybridDealStrategy:
    """
    Combines multiple strategies for robust deal selection.

    Uses ensemble of:
    - Heuristic quality evaluation
    - Learned value functions
    - Statistical analysis (secretary problem)
    """

    def __init__(self,
                 policies: List[Policy] = None,
                 weights: Optional[List[float]] = None):
        """
        Initialize hybrid strategy.

        Args:
            policies: List of learned policies to ensemble
            weights: Weights for each policy (default: equal)
        """
        self.policies = policies or []
        self.weights = weights or [1.0 / len(self.policies)] * len(self.policies)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        self.comparator = DealComparator()

    def evaluate_deal_ensemble(self, state: GameState) -> float:
        """
        Evaluate deal using ensemble of policies.

        Args:
            state: Deal state

        Returns:
            Ensemble evaluation score
        """
        scores = []

        # Heuristic evaluation
        heuristic_score = evaluate_deal_quality(state)
        scores.append(heuristic_score)

        # Learned policy evaluations
        for policy in self.policies:
            if hasattr(policy, 'get_state_value'):
                value = policy.get_state_value(state)
                # Normalize to similar scale as heuristic
                normalized_value = value / 5.0  # Convert $ to cards
                scores.append(normalized_value)

        # Weighted average
        if self.weights and len(scores) == len(self.weights):
            ensemble_score = sum(s * w for s, w in zip(scores, self.weights))
        else:
            ensemble_score = np.mean(scores)

        return ensemble_score


def create_ai_deal_strategy(strategy_type: str,
                            policy_path: Optional[str] = None,
                            **kwargs) -> Tuple[DealSelector, DealAcceptor]:
    """
    Factory function to create AI deal strategies.

    Args:
        strategy_type: Type of strategy
            - "q_learning": Use Q-Learning policy
            - "td_learning": Use TD Learning policy
            - "hybrid": Ensemble of multiple approaches
            - "heuristic": Pure heuristic (baseline)
        policy_path: Path to saved policy (optional)
        **kwargs: Additional strategy parameters

    Returns:
        (selector, acceptor) tuple
    """
    if strategy_type == "q_learning":
        from optimization.solvers.rl.q_learning import QPolicy

        # Load or create Q-Learning policy
        if policy_path and Path(policy_path).exists():
            policy = QPolicy()
            policy.load(policy_path)
        else:
            policy = None

        selector = LearnedDealSelector(policy=policy)
        acceptor = LearnedDealAcceptor(policy=policy)

        return selector, acceptor

    elif strategy_type == "td_learning":
        from optimization.solvers.rl.td_learning import TDPolicy

        # Load or create TD Learning policy
        if policy_path and Path(policy_path).exists():
            policy = TDPolicy()
            policy.load(policy_path)
        else:
            policy = None

        selector = LearnedDealSelector(policy=policy)
        acceptor = LearnedDealAcceptor(policy=policy)

        return selector, acceptor

    elif strategy_type == "hybrid":
        # Create hybrid strategy with multiple policies
        policies = []

        # Try to load Q-Learning policy
        q_path = kwargs.get('q_policy_path')
        if q_path and Path(q_path).exists():
            from optimization.solvers.rl.q_learning import QPolicy
            q_policy = QPolicy()
            q_policy.load(q_path)
            policies.append(q_policy)

        # Try to load TD Learning policy
        td_path = kwargs.get('td_policy_path')
        if td_path and Path(td_path).exists():
            from optimization.solvers.rl.td_learning import TDPolicy
            td_policy = TDPolicy()
            td_policy.load(td_path)
            policies.append(td_policy)

        hybrid = HybridDealStrategy(policies=policies)

        # Use hybrid evaluation
        selector = DealSelector(strategy="quality")  # Will use heuristic
        acceptor = DealAcceptor(strategy="threshold")  # Baseline

        return selector, acceptor

    else:  # "heuristic" or default
        # Pure heuristic strategy
        selector = DealSelector(strategy="quality")
        acceptor = DealAcceptor(strategy="threshold")

        return selector, acceptor


def compare_deal_strategies(num_sessions: int = 5,
                            deals_per_session: int = 10,
                            start_seed: int = 0):
    """
    Compare different AI deal strategies.

    Tests:
    - Heuristic (baseline)
    - Q-Learning (if available)
    - TD Learning (if available)
    - Hybrid ensemble

    Args:
        num_sessions: Number of test sessions
        deals_per_session: Deals per session
        start_seed: Starting random seed
    """
    from optimization.meta.multideal_mode import MultidealManager
    from optimization.solvers.heuristic_solver import HeuristicSolver

    print("=" * 70)
    print("AI Deal Strategy Comparison")
    print("=" * 70)
    print(f"Sessions: {num_sessions}, Deals per session: {deals_per_session}")
    print()

    # Create solver for playing deals
    solver = HeuristicSolver()

    # Test strategies
    strategies = [
        ("Heuristic", "heuristic", {}),
        ("Q-Learning", "q_learning", {"policy_path": "policies/q_policy.json"}),
        ("TD-Learning", "td_learning", {"policy_path": "policies/td_policy.json"}),
    ]

    results = {}

    for strategy_name, strategy_type, kwargs in strategies:
        print(f"\n{'='*70}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*70}")

        try:
            selector, acceptor = create_ai_deal_strategy(strategy_type, **kwargs)
        except Exception as e:
            print(f"  Skipping (error: {e})")
            continue

        session_profits = []
        session_winrates = []

        for session_id in range(num_sessions):
            manager = MultidealManager(
                starting_bankroll=500.0,
                max_rerolls=3,
                reroll_cost=5.0,
                solver=solver,
                selector=selector,
                acceptor=acceptor
            )

            session = manager.run_session(
                max_deals=deals_per_session,
                start_seed=start_seed + session_id * 100,
                verbose=False
            )

            session_profits.append(session.profit)
            session_winrates.append(session.win_rate)

            print(f"  Session {session_id+1}: "
                  f"Profit ${session.profit:+.2f}, "
                  f"Win rate {session.win_rate*100:.1f}%, "
                  f"Rerolls {session.total_rerolls_used}")

        # Summary
        avg_profit = np.mean(session_profits)
        avg_winrate = np.mean(session_winrates)

        results[strategy_name] = {
            'avg_profit': avg_profit,
            'avg_winrate': avg_winrate,
            'profits': session_profits
        }

        print(f"\n  Average profit: ${avg_profit:+.2f}")
        print(f"  Average win rate: {avg_winrate*100:.1f}%")

    # Final comparison
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    for strategy_name, data in results.items():
        print(f"\n{strategy_name}:")
        print(f"  Avg profit: ${data['avg_profit']:+.2f}")
        print(f"  Avg win rate: {data['avg_winrate']*100:.1f}%")


if __name__ == "__main__":
    # Quick test
    print("Testing AI Deal Strategies...")
    print()

    # Test heuristic strategy
    selector, acceptor = create_ai_deal_strategy("heuristic")
    print(f"Created heuristic strategy: {selector}, {acceptor}")

    # Test with a few deals
    games = []
    for seed in range(5):
        game = Game(seed=seed)
        game.deal()
        games.append(game)

    selected_idx, quality = selector.select_from_rerolls(games)
    print(f"\nSelected deal {selected_idx} with quality {quality:.1f}")

    # Test acceptance
    for i, game in enumerate(games):
        accept, q = acceptor.should_accept(game.state, rerolls_remaining=3, bankroll=500)
        print(f"Deal {i}: {'ACCEPT' if accept else 'REJECT'} (quality {q:.1f})")

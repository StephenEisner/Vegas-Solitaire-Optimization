"""
Multi-Deal Mode for Vegas Solitaire.

This implements a realistic casino scenario where:
1. You start with a bankroll
2. You can reroll deals (limited number of times)
3. Rerolls reset when you accept and play a deal
4. Goal: Maximize profit over multiple deals

Key decisions:
- Should I accept this deal or reroll?
- Which deal should I select from available rerolls?
- When should I quit (bankroll management)?

This uses AI algorithms (Q-Learning, TD Learning, Beam Search, MCTS)
to make optimal decisions.
"""

from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum

from game.core.game import Game
from game.core.state import GameState
from optimization.meta.deal_evaluator import evaluate_deal_quality, DealComparator
from optimization.solvers.base import Solver


class DealDecision(Enum):
    """Decision about a deal."""
    ACCEPT = "accept"
    REJECT = "reject"
    PLAY = "play"


@dataclass
class DealResult:
    """Result of playing a deal."""
    seed: int
    initial_quality: float
    final_score: int
    cards_played: int
    profit: int  # Score - cost
    moves_made: int
    won: bool


@dataclass
class MultidealSession:
    """Results from a multideal session."""
    starting_bankroll: float
    final_bankroll: float
    deals_played: int
    deals_evaluated: int
    total_rerolls_used: int
    results: List[DealResult]
    profit: float
    win_rate: float


class DealSelector:
    """
    Intelligent deal selection using AI algorithms.

    Decides which deal to play from available rerolls.
    """

    def __init__(self, strategy: str = "quality"):
        """
        Initialize deal selector.

        Args:
            strategy: Selection strategy
                - "quality": Use deal quality heuristic
                - "learned": Use learned value function
                - "secretary": Secretary problem algorithm
                - "threshold": Threshold-based selection
        """
        self.strategy = strategy
        self.comparator = DealComparator()

    def select_from_rerolls(self,
                           deals: List[Game],
                           evaluator: Optional[Callable] = None) -> Tuple[int, float]:
        """
        Select best deal from available rerolls.

        Args:
            deals: List of dealt games
            evaluator: Optional custom evaluator function

        Returns:
            (selected_index, quality_score)
        """
        if not deals:
            return None, 0.0

        if evaluator is None:
            evaluator = evaluate_deal_quality

        # Evaluate all deals
        qualities = []
        for game in deals:
            quality = evaluator(game.state)
            qualities.append(quality)

        # Select based on strategy
        if self.strategy == "quality":
            # Simple: pick best quality
            best_idx = np.argmax(qualities)
            return best_idx, qualities[best_idx]

        elif self.strategy == "secretary":
            # Secretary problem: observe first 37%, then take next better
            n = len(deals)
            observe_count = max(1, int(n * 0.37))

            best_observed = max(qualities[:observe_count])

            # Look for better deal
            for i in range(observe_count, n):
                if qualities[i] > best_observed:
                    return i, qualities[i]

            # Fallback: best overall
            best_idx = np.argmax(qualities)
            return best_idx, qualities[best_idx]

        elif self.strategy == "threshold":
            # Take first above calibrated threshold
            if not hasattr(self.comparator, 'quality_threshold'):
                self.comparator.calibrate(num_samples=1000)

            threshold = self.comparator.quality_threshold

            for i, quality in enumerate(qualities):
                if quality >= threshold:
                    return i, quality

            # No deal above threshold, take best
            best_idx = np.argmax(qualities)
            return best_idx, qualities[best_idx]

        else:
            # Default: best quality
            best_idx = np.argmax(qualities)
            return best_idx, qualities[best_idx]


class DealAcceptor:
    """
    Decides whether to accept a deal or keep rerolling.

    Uses learned value functions or heuristics to make this decision.
    """

    def __init__(self,
                 strategy: str = "threshold",
                 quality_threshold: Optional[float] = None,
                 value_function: Optional[Callable] = None):
        """
        Initialize deal acceptor.

        Args:
            strategy: Acceptance strategy
                - "threshold": Accept if quality > threshold
                - "value": Use learned value function
                - "conservative": High threshold (take fewer risks)
                - "aggressive": Low threshold (play more deals)
            quality_threshold: Custom quality threshold
            value_function: Learned value function for "value" strategy
        """
        self.strategy = strategy
        self.quality_threshold = quality_threshold
        self.value_function = value_function

        # Calibrate default threshold
        if self.quality_threshold is None:
            comparator = DealComparator()
            comparator.calibrate(num_samples=1000)

            if strategy == "conservative":
                self.quality_threshold = comparator.quality_threshold * 1.2
            elif strategy == "aggressive":
                self.quality_threshold = comparator.quality_threshold * 0.8
            else:
                self.quality_threshold = comparator.quality_threshold

    def should_accept(self,
                     state: GameState,
                     rerolls_remaining: int,
                     bankroll: float) -> Tuple[bool, float]:
        """
        Decide whether to accept this deal.

        Args:
            state: Current deal state
            rerolls_remaining: Number of rerolls left
            bankroll: Current bankroll

        Returns:
            (should_accept, quality_score)
        """
        quality = evaluate_deal_quality(state)

        # Strategy-based decision
        if self.strategy in ["threshold", "conservative", "aggressive"]:
            # Accept if above threshold
            accept = quality >= self.quality_threshold

            # Adjust based on rerolls remaining
            if rerolls_remaining == 0:
                # Must accept (no more rerolls)
                accept = True
            elif rerolls_remaining == 1:
                # Last reroll - be more lenient
                accept = quality >= self.quality_threshold * 0.9

        elif self.strategy == "value":
            # Use learned value function
            if self.value_function is None:
                # Fallback to threshold
                accept = quality >= self.quality_threshold
            else:
                expected_value = self.value_function(state)
                # Accept if expected value > cost to play
                accept = expected_value > 52  # Cost to play one game

        else:
            # Default: threshold
            accept = quality >= self.quality_threshold

        return accept, quality


class MultidealManager:
    """
    Manages multi-deal sessions with bankroll and reroll management.

    This is the main interface for playing multiple deals with AI optimization.
    """

    def __init__(self,
                 starting_bankroll: float = 500.0,
                 cost_per_game: float = 52.0,
                 reward_per_card: float = 5.0,
                 max_rerolls: int = 3,
                 reroll_cost: float = 5.0,
                 solver: Optional[Solver] = None,
                 selector: Optional[DealSelector] = None,
                 acceptor: Optional[DealAcceptor] = None):
        """
        Initialize multideal manager.

        Args:
            starting_bankroll: Initial money
            cost_per_game: Cost to play one game ($52 in Vegas rules)
            reward_per_card: Reward per card in foundation ($5 in Vegas rules)
            max_rerolls: Number of rerolls available (reset when deal accepted)
            reroll_cost: Cost to reroll a deal
            solver: AI solver to play deals
            selector: Deal selection strategy
            acceptor: Deal acceptance strategy
        """
        self.starting_bankroll = starting_bankroll
        self.bankroll = starting_bankroll
        self.cost_per_game = cost_per_game
        self.reward_per_card = reward_per_card
        self.max_rerolls = max_rerolls
        self.reroll_cost = reroll_cost

        # Components
        self.solver = solver
        self.selector = selector or DealSelector(strategy="quality")
        self.acceptor = acceptor or DealAcceptor(strategy="threshold")

        # Session tracking
        self.rerolls_remaining = max_rerolls
        self.deals_played = 0
        self.deals_evaluated = 0
        self.total_rerolls_used = 0
        self.results: List[DealResult] = []

    def reset_session(self):
        """Reset session state."""
        self.bankroll = self.starting_bankroll
        self.rerolls_remaining = self.max_rerolls
        self.deals_played = 0
        self.deals_evaluated = 0
        self.total_rerolls_used = 0
        self.results = []

    def can_afford_game(self) -> bool:
        """Check if we can afford to play another game."""
        return self.bankroll >= self.cost_per_game

    def can_afford_reroll(self) -> bool:
        """Check if we can afford to reroll."""
        return self.bankroll >= self.reroll_cost and self.rerolls_remaining > 0

    def evaluate_deal(self, game: Game) -> Tuple[DealDecision, float]:
        """
        Evaluate a single deal and decide whether to accept.

        Args:
            game: Dealt game

        Returns:
            (decision, quality)
        """
        self.deals_evaluated += 1

        # Check acceptance
        accept, quality = self.acceptor.should_accept(
            game.state,
            self.rerolls_remaining,
            self.bankroll
        )

        if accept:
            return DealDecision.ACCEPT, quality
        elif self.can_afford_reroll():
            return DealDecision.REJECT, quality
        else:
            # Can't reroll, must accept
            return DealDecision.ACCEPT, quality

    def reroll_deal(self, seed_generator: Callable) -> Game:
        """
        Reroll to get a new deal.

        Args:
            seed_generator: Function to generate next seed

        Returns:
            New game with new deal
        """
        if not self.can_afford_reroll():
            raise ValueError("Cannot afford reroll")

        # Pay reroll cost
        self.bankroll -= self.reroll_cost
        self.rerolls_remaining -= 1
        self.total_rerolls_used += 1

        # Generate new deal
        seed = seed_generator()
        game = Game(seed=seed)
        game.deal()

        return game

    def play_deal(self, game: Game, quality: float) -> DealResult:
        """
        Play a deal and record results.

        Args:
            game: Game to play
            quality: Pre-evaluated deal quality

        Returns:
            Deal result
        """
        if not self.can_afford_game():
            raise ValueError("Cannot afford to play game")

        # Pay to play
        self.bankroll -= self.cost_per_game

        # Play the game
        moves_made = 0
        max_moves = 500

        if self.solver is None:
            # No solver - play random moves for testing
            from game.core.rules import get_valid_moves

            while moves_made < max_moves:
                valid_moves = get_valid_moves(game.state)
                if not valid_moves:
                    break

                move = np.random.choice(valid_moves)
                if not game.make_move(move):
                    break
                moves_made += 1
        else:
            # Use solver's choose_move method
            while moves_made < max_moves:
                move = self.solver.choose_move(game)
                if not move:
                    break

                if not game.make_move(move):
                    break
                moves_made += 1

        # Calculate rewards
        cards_played = game.state.get_foundation_count()
        score = cards_played * self.reward_per_card
        profit = score - self.cost_per_game

        # Update bankroll
        self.bankroll += score

        # Record result
        deal_result = DealResult(
            seed=game.seed,
            initial_quality=quality,
            final_score=score,
            cards_played=cards_played,
            profit=profit,
            moves_made=moves_made,
            won=(cards_played == 52)
        )

        self.results.append(deal_result)
        self.deals_played += 1

        # Reset rerolls after playing a deal
        self.rerolls_remaining = self.max_rerolls

        return deal_result

    def run_session(self,
                   max_deals: int = 10,
                   start_seed: int = 0,
                   verbose: bool = True) -> MultidealSession:
        """
        Run a complete multideal session.

        Args:
            max_deals: Maximum number of deals to play
            start_seed: Starting random seed
            verbose: Print progress

        Returns:
            Session results
        """
        self.reset_session()

        if verbose:
            print("=" * 70)
            print(f"Multi-Deal Session Starting")
            print("=" * 70)
            print(f"Starting bankroll: ${self.starting_bankroll:.2f}")
            print(f"Max deals: {max_deals}")
            print(f"Max rerolls per deal: {self.max_rerolls}")
            print(f"Cost per game: ${self.cost_per_game:.2f}")
            print(f"Reroll cost: ${self.reroll_cost:.2f}")
            print()

        seed_counter = start_seed

        def next_seed():
            nonlocal seed_counter
            seed = seed_counter
            seed_counter += 1
            return seed

        while self.deals_played < max_deals and self.can_afford_game():
            if verbose:
                print(f"\n--- Deal {self.deals_played + 1}/{max_deals} ---")
                print(f"Bankroll: ${self.bankroll:.2f}, Rerolls: {self.rerolls_remaining}/{self.max_rerolls}")

            # Generate initial deal
            game = Game(seed=next_seed())
            game.deal()

            # Evaluate and potentially reroll
            while True:
                decision, quality = self.evaluate_deal(game)

                if verbose:
                    print(f"  Deal quality: {quality:.1f} - {decision.value}")

                if decision == DealDecision.ACCEPT:
                    # Play this deal
                    result = self.play_deal(game, quality)

                    if verbose:
                        print(f"  → Played: {result.cards_played}/52 cards, "
                              f"${result.final_score} score, "
                              f"${result.profit:+.0f} profit")
                    break

                elif decision == DealDecision.REJECT:
                    # Reroll
                    if not self.can_afford_reroll():
                        # Must play this one
                        result = self.play_deal(game, quality)
                        if verbose:
                            print(f"  → Forced to play (no rerolls)")
                        break
                    else:
                        game = self.reroll_deal(next_seed)
                        if verbose:
                            print(f"  → Rerolled")
                else:
                    # Shouldn't happen
                    break

        # Session complete
        if verbose:
            print("\n" + "=" * 70)
            print("Session Complete")
            print("=" * 70)

        total_profit = self.bankroll - self.starting_bankroll
        wins = sum(1 for r in self.results if r.won)
        win_rate = wins / len(self.results) if self.results else 0

        if verbose:
            print(f"Final bankroll: ${self.bankroll:.2f}")
            print(f"Total profit: ${total_profit:+.2f}")
            print(f"Deals played: {self.deals_played}")
            print(f"Deals evaluated: {self.deals_evaluated}")
            print(f"Total rerolls: {self.total_rerolls_used}")
            print(f"Win rate: {win_rate*100:.1f}% ({wins}/{len(self.results)})")

            if self.results:
                avg_cards = np.mean([r.cards_played for r in self.results])
                avg_profit = np.mean([r.profit for r in self.results])
                print(f"Avg cards played: {avg_cards:.1f}/52")
                print(f"Avg profit per deal: ${avg_profit:+.2f}")

        return MultidealSession(
            starting_bankroll=self.starting_bankroll,
            final_bankroll=self.bankroll,
            deals_played=self.deals_played,
            deals_evaluated=self.deals_evaluated,
            total_rerolls_used=self.total_rerolls_used,
            results=self.results,
            profit=total_profit,
            win_rate=win_rate
        )


def demo_multideal():
    """Demonstrate multi-deal mode."""
    from optimization.solvers.heuristic_solver import HeuristicSolver

    print("=" * 70)
    print("Multi-Deal Mode Demo")
    print("=" * 70)
    print()

    # Create solver
    solver = HeuristicSolver()

    # Create multideal manager with different strategies
    strategies = [
        ("Aggressive", DealAcceptor(strategy="aggressive")),
        ("Threshold", DealAcceptor(strategy="threshold")),
        ("Conservative", DealAcceptor(strategy="conservative")),
    ]

    for strategy_name, acceptor in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*70}")

        manager = MultidealManager(
            starting_bankroll=500.0,
            max_rerolls=3,
            reroll_cost=5.0,
            solver=solver,
            acceptor=acceptor
        )

        session = manager.run_session(max_deals=10, start_seed=1000, verbose=False)

        print(f"Profit: ${session.profit:+.2f}")
        print(f"Win rate: {session.win_rate*100:.1f}%")
        print(f"Rerolls used: {session.total_rerolls_used}")
        print(f"Avg cards: {np.mean([r.cards_played for r in session.results]):.1f}/52")


if __name__ == "__main__":
    demo_multideal()

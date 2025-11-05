"""
Deal evaluation for multi-deal optimization.

This module implements quick evaluation of deal quality based on
observable features at deal time.

Key insights (from user experience):
1. Don't play deals without many immediately playable cards
2. Stack depth matters - short stacks good for Kings, long stacks for information
3. Initial configuration strongly predicts winnability
"""

import numpy as np
from typing import Dict, List, Tuple
from game.core.state import GameState
from game.core.moves import Move, MoveType
from game.core.rules import get_valid_moves
from game.core.card import Rank


def count_immediately_playable(state: GameState) -> int:
    """
    Count cards that can be played immediately or very quickly.

    User insight: "Don't play deals without many cards that can be
    immediately or quickly used"

    Returns:
        Number of immediately playable moves
    """
    playable = 0

    # 1. Visible Aces (can start foundations immediately)
    for col in range(7):
        visible = state.get_tableau_visible_cards(col)
        for card in visible:
            if card.rank == Rank.ACE:
                playable += 3  # Aces are very playable!

    # 2. Cards already on foundation
    playable += state.get_foundation_count()

    # 3. Tableau-to-foundation moves available now
    valid_moves = get_valid_moves(state)
    foundation_moves = [m for m in valid_moves
                       if m.move_type in [MoveType.TABLEAU_TO_FOUNDATION,
                                         MoveType.WASTE_TO_FOUNDATION]]
    playable += len(foundation_moves) * 2

    # 4. Easy tableau-to-tableau moves (single cards)
    easy_tableau_moves = [m for m in valid_moves
                         if m.move_type == MoveType.TABLEAU_TO_TABLEAU
                         and m.card_count == 1]
    playable += len(easy_tableau_moves)

    return playable


def evaluate_stack_configuration(state: GameState) -> Dict[str, float]:
    """
    Analyze stack depths and configuration.

    User insight: "Sometimes prioritize short stacks (for Kings),
    sometimes long stacks (for hidden cards)"

    Returns:
        Dictionary of stack configuration metrics
    """
    visible_depths = []
    hidden_counts = []

    for col in range(7):
        visible = state.get_tableau_visible_cards(col)
        hidden = state.tableau_hidden[col]

        visible_depths.append(len(visible))
        hidden_counts.append(hidden)

    # Count empty columns
    empty_cols = sum(1 for col in state.tableau if len(col) == 0)

    # Count Kings visible (can use empty columns)
    kings_visible = 0
    for col in range(7):
        visible = state.get_tableau_visible_cards(col)
        for card in visible:
            if card.rank == Rank.KING:
                kings_visible += 1

    return {
        'avg_visible_depth': np.mean(visible_depths) if visible_depths else 0,
        'max_visible_depth': max(visible_depths) if visible_depths else 0,
        'total_hidden': sum(hidden_counts),
        'max_hidden': max(hidden_counts) if hidden_counts else 0,
        'short_stacks': sum(1 for d in visible_depths if d <= 2),  # User insight!
        'long_stacks': sum(1 for d in visible_depths if d >= 5),   # User insight!
        'empty_cols': empty_cols,
        'kings_visible': kings_visible,
        'empty_king_synergy': 1 if (empty_cols > 0 and kings_visible > 0) else 0
    }


def estimate_hidden_potential(state: GameState) -> float:
    """
    Estimate value of hidden cards (probabilistic).

    More hidden cards = more potential, but also more uncertainty.
    Deep stacks might have valuable cards.

    Returns:
        Estimated value from hidden cards
    """
    score = 0.0

    total_hidden = sum(state.tableau_hidden)
    cards_dealt = 28  # Initial deal
    cards_remaining = 52 - cards_dealt

    # Count Aces already visible
    aces_visible = 0
    for col in range(7):
        visible = state.get_tableau_visible_cards(col)
        for card in visible:
            if card.rank == Rank.ACE:
                aces_visible += 1

    # Expected Aces in hidden cards
    aces_remaining = 4 - aces_visible
    if cards_remaining > 0:
        expected_hidden_aces = aces_remaining * (total_hidden / cards_remaining)
        score += expected_hidden_aces * 15.0  # Aces very valuable

    # Bonus for deep stacks (user insight: "work on getting out deeper hidden cards")
    for col in range(7):
        hidden = state.tableau_hidden[col]
        if hidden >= 4:  # Deep stack
            score += hidden * 1.5  # Potential value

    # Balance: Too many hidden cards is also risky
    if total_hidden > 20:
        score -= (total_hidden - 20) * 0.5  # Penalty for excessive hidden

    return score


def evaluate_sequence_quality(state: GameState) -> float:
    """
    Evaluate quality of visible sequences.

    Good sequences = cards already in playable order.

    Returns:
        Sequence quality score
    """
    score = 0.0

    for col in range(7):
        visible = state.get_tableau_visible_cards(col)

        if len(visible) < 2:
            continue

        # Count valid sequences
        sequence_length = 1
        for i in range(len(visible) - 1):
            if visible[i + 1].can_stack_on(visible[i]):
                sequence_length += 1
            else:
                # Sequence broken
                if sequence_length >= 3:
                    score += sequence_length * 2  # Reward long sequences
                sequence_length = 1

        # Check final sequence
        if sequence_length >= 3:
            score += sequence_length * 2

    return score


def evaluate_deal_quality(state: GameState) -> float:
    """
    Overall deal quality score incorporating all factors and user insights.

    Higher score = better deal to play.

    Key factors:
    1. Immediate playability (critical - user insight!)
    2. Stack configuration (short/long mix)
    3. Hidden potential
    4. Sequence quality

    Returns:
        Deal quality score (typically 0-100)
    """
    score = 0.0

    # 1. Immediate playability (MOST IMPORTANT - user insight)
    immediately_playable = count_immediately_playable(state)
    score += immediately_playable * 8.0  # Heavy weight

    # User rule: Minimum threshold
    if immediately_playable < 3:
        score -= 30.0  # Strong penalty for unplayable deals

    # 2. Stack configuration
    config = evaluate_stack_configuration(state)

    # Empty columns (good for Kings)
    score += config['empty_cols'] * 10.0

    # King-empty synergy (user insight)
    if config['empty_king_synergy']:
        score += 15.0

    # Prefer mix of short and long stacks (user insight: "sometimes... sometimes...")
    if config['short_stacks'] >= 2 and config['long_stacks'] >= 2:
        score += 12.0  # Flexibility!
    elif config['short_stacks'] >= 3:
        score += 8.0   # Can create empty columns
    elif config['long_stacks'] >= 4:
        score += 6.0   # Information potential

    # Not too many hidden cards
    if config['total_hidden'] <= 15:
        score += 10.0  # Less uncertainty

    # 3. Hidden potential
    potential = estimate_hidden_potential(state)
    score += potential * 0.4  # Lower weight (speculative)

    # 4. Sequence quality
    sequences = evaluate_sequence_quality(state)
    score += sequences * 0.5

    # 5. Foundation progress (if any cards already there)
    foundation_count = state.get_foundation_count()
    score += foundation_count * 15.0  # Good head start!

    return score


class DealComparator:
    """
    Compare multiple deals and select the best one.
    """

    def __init__(self):
        """Initialize deal comparator."""
        self.quality_threshold = None
        self._calibrated = False

    def calibrate(self, num_samples: int = 1000):
        """
        Calibrate quality threshold from random deals.

        Finds typical distribution of deal quality.

        Args:
            num_samples: Number of random deals to sample
        """
        from game.core.game import Game

        qualities = []
        for seed in range(num_samples):
            game = Game(seed=seed)
            game.deal()
            quality = evaluate_deal_quality(game.state)
            qualities.append(quality)

        # Set threshold at 70th percentile
        self.quality_threshold = np.percentile(qualities, 70)
        self._calibrated = True

        print(f"Calibrated on {num_samples} deals")
        print(f"  Mean quality: {np.mean(qualities):.1f}")
        print(f"  Median quality: {np.median(qualities):.1f}")
        print(f"  70th percentile: {self.quality_threshold:.1f}")
        print(f"  90th percentile: {np.percentile(qualities, 90):.1f}")

    def compare_deals(self, states: List[GameState]) -> List[Tuple[int, float]]:
        """
        Compare multiple deals and rank them.

        Args:
            states: List of game states to compare

        Returns:
            List of (index, quality) tuples sorted by quality (best first)
        """
        ranked = []
        for i, state in enumerate(states):
            quality = evaluate_deal_quality(state)
            ranked.append((i, quality))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def select_best(self, states: List[GameState]) -> Tuple[int, float]:
        """
        Select the best deal from a list.

        Args:
            states: List of game states

        Returns:
            (index_of_best, quality_score)
        """
        ranked = self.compare_deals(states)
        return ranked[0]

    def should_play(self, state: GameState) -> bool:
        """
        Decide if this deal is good enough to play.

        Requires calibration first.

        Args:
            state: Game state to evaluate

        Returns:
            True if quality exceeds threshold
        """
        if not self._calibrated:
            raise ValueError("Must calibrate() before using should_play()")

        quality = evaluate_deal_quality(state)
        return quality >= self.quality_threshold

    def explain_quality(self, state: GameState) -> str:
        """
        Generate human-readable explanation of deal quality.

        Args:
            state: Game state to explain

        Returns:
            Formatted string explaining the evaluation
        """
        quality = evaluate_deal_quality(state)
        playable = count_immediately_playable(state)
        config = evaluate_stack_configuration(state)
        potential = estimate_hidden_potential(state)
        sequences = evaluate_sequence_quality(state)

        lines = [
            f"Deal Quality Score: {quality:.1f}",
            "",
            "Breakdown:",
            f"  Immediately playable: {playable} cards",
            f"  Empty columns: {config['empty_cols']}",
            f"  Short stacks (≤2 visible): {config['short_stacks']}",
            f"  Long stacks (≥5 visible): {config['long_stacks']}",
            f"  Kings visible: {config['kings_visible']}",
            f"  Total hidden: {config['total_hidden']}",
            f"  Hidden potential: {potential:.1f}",
            f"  Sequence quality: {sequences:.1f}",
            "",
        ]

        # Recommendation
        if self._calibrated:
            if quality >= self.quality_threshold:
                lines.append(f"✓ PLAY THIS DEAL (above threshold {self.quality_threshold:.1f})")
            else:
                lines.append(f"✗ Skip this deal (below threshold {self.quality_threshold:.1f})")

        # User insights
        if playable < 3:
            lines.append("  ⚠️  Warning: Very few immediately playable cards")

        if config['empty_king_synergy']:
            lines.append("  ✓ Good: Empty columns + visible Kings")

        if config['short_stacks'] >= 2 and config['long_stacks'] >= 2:
            lines.append("  ✓ Good: Mix of short and long stacks (flexibility)")

        return "\n".join(lines)


def demo_deal_evaluation():
    """Demonstrate deal evaluation on a few random deals."""
    from game.core.game import Game

    print("=" * 70)
    print("Deal Quality Evaluation Demo")
    print("=" * 70)

    # Calibrate
    comparator = DealComparator()
    print("\nCalibrating...")
    comparator.calibrate(num_samples=1000)

    print("\n" + "=" * 70)
    print("Evaluating Sample Deals")
    print("=" * 70)

    # Evaluate a few deals
    for seed in [42, 100, 200, 500, 1000]:
        game = Game(seed=seed)
        game.deal()

        print(f"\n--- Deal {seed} ---")
        print(comparator.explain_quality(game.state))


if __name__ == "__main__":
    demo_deal_evaluation()

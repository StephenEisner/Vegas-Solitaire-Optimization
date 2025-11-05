"""
Feature extraction from game states for machine learning.

Good features are crucial for RL performance. We want features that:
1. Capture important game state information
2. Generalize across similar positions
3. Are computationally cheap to compute
4. Have reasonable dimensionality
"""

import numpy as np
from typing import List, Dict
from game.core.state import GameState
from game.core.card import Rank, Suit


class StateFeatures:
    """
    Extract numerical features from game states.

    Features are designed to be:
    - Informative: Capture state relevant to winning
    - Normalized: Values in reasonable ranges
    - Compact: Not too many dimensions
    """

    # Feature dimensions
    FOUNDATION_FEATURES = 4      # One per suit
    TABLEAU_DEPTH_FEATURES = 7   # Max visible depth per column
    TABLEAU_HIDDEN_FEATURES = 7  # Hidden card counts
    STOCK_WASTE_FEATURES = 3     # Stock size, waste size, passes
    GAME_STATE_FEATURES = 3      # Empty columns, total visible, total hidden

    TOTAL_FEATURES = (FOUNDATION_FEATURES + TABLEAU_DEPTH_FEATURES +
                     TABLEAU_HIDDEN_FEATURES + STOCK_WASTE_FEATURES +
                     GAME_STATE_FEATURES)  # 24 total

    @staticmethod
    def extract_features(state: GameState) -> np.ndarray:
        """
        Extract feature vector from game state.

        Returns:
            Feature vector of shape (24,) with values roughly in [0, 1]
        """
        features = []

        # 1. Foundation counts (4 features) - Progress toward goal
        # Normalize by max possible (13 cards per suit)
        for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
            count = len(state.foundations[suit])
            features.append(count / 13.0)

        # 2. Tableau visible depth (7 features) - How much we can see
        # Deeper visibility = more options
        for col in range(7):
            visible = state.get_tableau_visible_cards(col)
            depth = len(visible)
            # Normalize by typical max (~13)
            features.append(min(depth / 13.0, 1.0))

        # 3. Tableau hidden counts (7 features) - Unknown information
        # More hidden = less information
        for col in range(7):
            hidden = state.tableau_hidden[col]
            # Normalize by typical max (~7)
            features.append(min(hidden / 7.0, 1.0))

        # 4. Stock and waste (3 features)
        # Stock size (normalized by initial 24 cards)
        features.append(len(state.stock) / 24.0)

        # Waste size (normalized by max possible ~24)
        features.append(len(state.waste) / 24.0)

        # Passes through deck (normalized by max 3)
        features.append(state.passes_through_deck / 3.0)

        # 5. Game state features (3 features)
        # Empty tableau columns (normalized by max 7)
        empty_cols = sum(1 for col in state.tableau if len(col) == 0)
        features.append(empty_cols / 7.0)

        # Total visible cards (normalized by 52)
        total_visible = sum(len(state.get_tableau_visible_cards(col)) for col in range(7))
        features.append(total_visible / 52.0)

        # Total hidden cards (normalized by initial ~28)
        total_hidden = sum(state.tableau_hidden)
        features.append(total_hidden / 28.0)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def extract_extended_features(state: GameState) -> np.ndarray:
        """
        Extract more detailed feature vector (higher dimensional).

        This includes additional features like:
        - Specific card locations
        - Sequence information
        - Suit distribution

        Returns:
            Extended feature vector (higher dimension, more expressive)
        """
        # Start with basic features
        basic = StateFeatures.extract_features(state).tolist()

        # Add extended features

        # 6. Longest sequences in each column (7 features)
        for col in range(7):
            visible = state.get_tableau_visible_cards(col)
            if visible:
                # Count longest valid sequence
                max_seq = 1
                current_seq = 1
                for i in range(len(visible) - 1):
                    if visible[i + 1].can_stack_on(visible[i]):
                        current_seq += 1
                        max_seq = max(max_seq, current_seq)
                    else:
                        current_seq = 1
                basic.append(max_seq / 13.0)
            else:
                basic.append(0.0)

        # 7. Available foundation moves (1 feature)
        foundation_moves = 0
        for col in range(7):
            visible = state.get_tableau_visible_cards(col)
            if visible:
                top_card = visible[-1]
                # Check if can go to foundation
                foundation_cards = state.foundations[top_card.suit]
                if not foundation_cards:
                    if top_card.rank.numeric_value == 1:
                        foundation_moves += 1
                else:
                    if top_card.rank.numeric_value == foundation_cards[-1].rank.numeric_value + 1:
                        foundation_moves += 1
        # Also check waste
        if state.waste:
            top_waste = state.waste[-1]
            foundation_cards = state.foundations[top_waste.suit]
            if not foundation_cards:
                if top_waste.rank.numeric_value == 1:
                    foundation_moves += 1
            else:
                if top_waste.rank.numeric_value == foundation_cards[-1].rank.numeric_value + 1:
                    foundation_moves += 1

        basic.append(foundation_moves / 8.0)  # Max 8 possible at once

        # 8. Kings not in empty columns (1 feature)
        # Having Kings not in empty columns is wasteful
        kings_misplaced = 0
        for col in range(7):
            visible = state.get_tableau_visible_cards(col)
            if visible:
                for card in visible:
                    if card.rank.numeric_value == 13:
                        # King found in non-empty column (or empty column with cards below)
                        if len(state.tableau[col]) > 1 or state.tableau_hidden[col] > 0:
                            kings_misplaced += 1
        basic.append(kings_misplaced / 4.0)  # Max 4 kings

        return np.array(basic, dtype=np.float32)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return human-readable names for each feature."""
        names = []

        # Foundation counts
        for suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades']:
            names.append(f'foundation_{suit.lower()}')

        # Tableau depth
        for i in range(7):
            names.append(f'tableau_{i}_visible_depth')

        # Hidden counts
        for i in range(7):
            names.append(f'tableau_{i}_hidden_count')

        # Stock/waste
        names.extend(['stock_size', 'waste_size', 'passes_through_deck'])

        # Game state
        names.extend(['empty_columns', 'total_visible', 'total_hidden'])

        return names

    @staticmethod
    def describe_features(features: np.ndarray) -> str:
        """
        Create human-readable description of feature vector.

        Args:
            features: Feature vector from extract_features()

        Returns:
            Formatted string describing the features
        """
        names = StateFeatures.get_feature_names()
        lines = ["State Features:"]
        lines.append("-" * 50)

        for name, value in zip(names, features):
            # Denormalize some values for readability
            if 'foundation' in name:
                actual = int(value * 13)
                lines.append(f"  {name:30s}: {actual:2d}/13 ({value:.2f})")
            elif 'passes' in name:
                actual = int(value * 3)
                lines.append(f"  {name:30s}: {actual:1d}/3  ({value:.2f})")
            elif 'empty' in name:
                actual = int(value * 7)
                lines.append(f"  {name:30s}: {actual:1d}/7  ({value:.2f})")
            else:
                lines.append(f"  {name:30s}: {value:.3f}")

        return "\n".join(lines)


def compute_state_hash(features: np.ndarray) -> int:
    """
    Compute hash of feature vector for state aggregation.

    This allows grouping similar states together in tabular RL.

    Args:
        features: Feature vector

    Returns:
        Integer hash
    """
    # Quantize features to reduce state space
    # Round to nearest 0.1
    quantized = np.round(features * 10).astype(int)
    return hash(tuple(quantized))

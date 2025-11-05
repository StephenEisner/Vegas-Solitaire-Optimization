"""
Empirical Deal Quality Evaluation using AI Solvers.

Instead of hand-crafted heuristics, this module uses actual AI solvers
to simulate playing deals and measures their outcomes empirically.

Key insights:
1. Run multiple solvers on each deal to get expected value
2. Build dataset of (deal_features, outcome) pairs
3. Train supervised learning model to predict deal quality
4. Analyze which features actually correlate with good outcomes

This is data-driven rather than heuristic-driven!
"""

from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

from game.core.game import Game
from game.core.state import GameState
from optimization.features.state_features import StateFeatures
from optimization.solvers.base import Solver
from optimization.meta.deal_evaluator import evaluate_deal_quality


@dataclass
class DealOutcome:
    """Results from playing a deal."""
    seed: int

    # Initial features
    initial_features: np.ndarray
    heuristic_quality: float

    # Outcomes from different solvers
    outcomes: Dict[str, Dict]  # solver_name -> {score, cards, moves, time}

    # Aggregate metrics
    avg_score: float
    avg_cards: float
    max_score: float
    min_score: float
    win_rate: float  # Fraction of solvers that won

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'seed': self.seed,
            'initial_features': self.initial_features.tolist(),
            'heuristic_quality': self.heuristic_quality,
            'outcomes': self.outcomes,
            'avg_score': self.avg_score,
            'avg_cards': self.avg_cards,
            'max_score': self.max_score,
            'min_score': self.min_score,
            'win_rate': self.win_rate
        }


class EmpiricalDealEvaluator:
    """
    Evaluates deal quality by actually playing deals with multiple solvers.

    This gives us ground truth about which deals are good vs bad.
    """

    def __init__(self, solvers: Dict[str, Solver]):
        """
        Initialize empirical evaluator.

        Args:
            solvers: Dictionary of {name: solver} to use for evaluation
        """
        self.solvers = solvers
        self.outcomes_cache: Dict[int, DealOutcome] = {}

    def evaluate_deal(self,
                     game: Game,
                     max_moves: int = 500,
                     verbose: bool = False) -> DealOutcome:
        """
        Evaluate a deal by playing it with all solvers.

        Args:
            game: Game with deal to evaluate
            max_moves: Max moves per solver
            verbose: Print progress

        Returns:
            DealOutcome with results from all solvers
        """
        seed = game.seed

        # Check cache
        if seed in self.outcomes_cache:
            return self.outcomes_cache[seed]

        # Extract initial features
        initial_state = game.state.copy()
        initial_features = StateFeatures.extract_features(initial_state)
        heuristic_quality = evaluate_deal_quality(initial_state)

        # Play with each solver
        outcomes = {}
        scores = []
        cards = []
        wins = []

        for solver_name, solver in self.solvers.items():
            # Create fresh game
            test_game = Game(seed=seed)
            test_game.deal()

            # Play the game
            import time
            start_time = time.time()
            moves_made = 0

            while moves_made < max_moves:
                move = solver.choose_move(test_game)
                if not move or not test_game.make_move(move):
                    break
                moves_made += 1

            elapsed = time.time() - start_time

            # Record outcome
            final_cards = test_game.state.get_foundation_count()
            final_score = final_cards * 5  # Vegas scoring
            won = (final_cards == 52)

            outcomes[solver_name] = {
                'score': final_score,
                'cards': final_cards,
                'moves': moves_made,
                'time': elapsed,
                'won': won
            }

            scores.append(final_score)
            cards.append(final_cards)
            wins.append(1 if won else 0)

            if verbose:
                print(f"  {solver_name}: {final_cards}/52 cards, ${final_score}, "
                      f"{'WIN' if won else 'LOSS'}")

        # Aggregate results
        outcome = DealOutcome(
            seed=seed,
            initial_features=initial_features,
            heuristic_quality=heuristic_quality,
            outcomes=outcomes,
            avg_score=np.mean(scores),
            avg_cards=np.mean(cards),
            max_score=max(scores),
            min_score=min(scores),
            win_rate=np.mean(wins)
        )

        # Cache it
        self.outcomes_cache[seed] = outcome

        return outcome

    def evaluate_many_deals(self,
                           num_deals: int,
                           start_seed: int = 0,
                           verbose: bool = True) -> List[DealOutcome]:
        """
        Evaluate many deals to build a dataset.

        Args:
            num_deals: Number of deals to evaluate
            start_seed: Starting seed
            verbose: Print progress

        Returns:
            List of DealOutcome objects
        """
        outcomes = []

        if verbose:
            print("=" * 70)
            print(f"Empirical Deal Evaluation: {num_deals} deals")
            print(f"Solvers: {', '.join(self.solvers.keys())}")
            print("=" * 70)
            print()

        for i in range(num_deals):
            seed = start_seed + i
            game = Game(seed=seed)
            game.deal()

            if verbose and i % 10 == 0:
                print(f"Evaluating deal {i+1}/{num_deals} (seed {seed})...")

            outcome = self.evaluate_deal(game, verbose=False)
            outcomes.append(outcome)

        if verbose:
            self._print_summary(outcomes)

        return outcomes

    def _print_summary(self, outcomes: List[DealOutcome]):
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("Evaluation Summary")
        print("=" * 70)

        avg_scores = [o.avg_score for o in outcomes]
        avg_cards = [o.avg_cards for o in outcomes]
        win_rates = [o.win_rate for o in outcomes]
        heuristic_qualities = [o.heuristic_quality for o in outcomes]

        print(f"\nOverall Statistics:")
        print(f"  Avg score: ${np.mean(avg_scores):.1f} (std ${np.std(avg_scores):.1f})")
        print(f"  Avg cards: {np.mean(avg_cards):.1f}/52 (std {np.std(avg_cards):.1f})")
        print(f"  Avg win rate: {np.mean(win_rates)*100:.1f}%")
        print(f"  Heuristic quality: {np.mean(heuristic_qualities):.1f} "
              f"(std {np.std(heuristic_qualities):.1f})")

        # Correlation analysis
        corr_heuristic_score = np.corrcoef(heuristic_qualities, avg_scores)[0, 1]
        corr_heuristic_cards = np.corrcoef(heuristic_qualities, avg_cards)[0, 1]

        print(f"\nHeuristic Correlation:")
        print(f"  With avg score: {corr_heuristic_score:.3f}")
        print(f"  With avg cards: {corr_heuristic_cards:.3f}")

        # Per-solver statistics
        print(f"\nPer-Solver Performance:")
        solver_names = list(outcomes[0].outcomes.keys())

        for solver_name in solver_names:
            scores = [o.outcomes[solver_name]['score'] for o in outcomes]
            cards = [o.outcomes[solver_name]['cards'] for o in outcomes]
            wins = [o.outcomes[solver_name]['won'] for o in outcomes]

            print(f"\n  {solver_name}:")
            print(f"    Avg score: ${np.mean(scores):.1f}")
            print(f"    Avg cards: {np.mean(cards):.1f}/52")
            print(f"    Win rate: {np.mean(wins)*100:.1f}%")

    def save_dataset(self, outcomes: List[DealOutcome], path: str):
        """Save outcomes dataset to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'num_deals': len(outcomes),
            'solvers': list(self.solvers.keys()),
            'outcomes': [o.to_dict() for o in outcomes]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Dataset saved to {path}")

    def load_dataset(self, path: str) -> List[DealOutcome]:
        """Load outcomes dataset from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        outcomes = []
        for o_dict in data['outcomes']:
            outcome = DealOutcome(
                seed=o_dict['seed'],
                initial_features=np.array(o_dict['initial_features']),
                heuristic_quality=o_dict['heuristic_quality'],
                outcomes=o_dict['outcomes'],
                avg_score=o_dict['avg_score'],
                avg_cards=o_dict['avg_cards'],
                max_score=o_dict['max_score'],
                min_score=o_dict['min_score'],
                win_rate=o_dict['win_rate']
            )
            outcomes.append(outcome)

        print(f"✓ Loaded {len(outcomes)} outcomes from {path}")
        return outcomes


class LearnedDealQualityPredictor:
    """
    Supervised learning model to predict deal quality from empirical data.

    Learns to predict expected score/cards from initial deal features.
    """

    def __init__(self):
        """Initialize predictor."""
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.trained = False

        # Statistics for normalization
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def train(self,
             outcomes: List[DealOutcome],
             target: str = 'avg_cards',
             regularization: float = 0.1,
             verbose: bool = True) -> None:
        """
        Train predictor using linear regression on empirical outcomes.

        Args:
            outcomes: Training data
            target: What to predict ('avg_cards', 'avg_score', 'win_rate')
            regularization: L2 regularization strength
            verbose: Print training info
        """
        if verbose:
            print(f"\nTraining Deal Quality Predictor...")
            print(f"  Training samples: {len(outcomes)}")
            print(f"  Target: {target}")

        # Extract features and targets
        X = np.array([o.initial_features for o in outcomes])

        if target == 'avg_cards':
            y = np.array([o.avg_cards for o in outcomes])
        elif target == 'avg_score':
            y = np.array([o.avg_score for o in outcomes])
        elif target == 'win_rate':
            y = np.array([o.win_rate for o in outcomes])
        else:
            raise ValueError(f"Unknown target: {target}")

        # Normalize features
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        X_norm = (X - self.feature_mean) / self.feature_std

        # Normalize targets
        self.target_mean = np.mean(y)
        self.target_std = np.std(y) + 1e-8
        y_norm = (y - self.target_mean) / self.target_std

        # Ridge regression: w = (X^T X + λI)^-1 X^T y
        n_features = X_norm.shape[1]
        XtX = X_norm.T @ X_norm
        Xty = X_norm.T @ y_norm

        # Add regularization
        reg_matrix = regularization * np.eye(n_features)

        # Solve
        self.weights = np.linalg.solve(XtX + reg_matrix, Xty)
        self.bias = 0.0  # Already centered

        self.trained = True

        # Evaluate on training set
        if verbose:
            y_pred = self.predict_from_outcomes(outcomes)
            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))
            r2 = 1 - mse / np.var(y)

            print(f"\nTraining Results:")
            print(f"  MSE: {mse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.3f}")

            # Feature importance
            print(f"\nTop 10 Most Important Features:")
            importance = np.abs(self.weights)
            top_indices = np.argsort(importance)[-10:][::-1]

            for idx in top_indices:
                weight = self.weights[idx]
                print(f"  Feature {idx}: {weight:+.3f}")

    def predict(self, features: np.ndarray) -> float:
        """
        Predict deal quality from features.

        Args:
            features: Feature vector

        Returns:
            Predicted value (cards, score, or win rate)
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        # Normalize
        features_norm = (features - self.feature_mean) / self.feature_std

        # Predict
        pred_norm = np.dot(self.weights, features_norm) + self.bias

        # Denormalize
        pred = pred_norm * self.target_std + self.target_mean

        return pred

    def predict_from_state(self, state: GameState) -> float:
        """Predict deal quality from game state."""
        features = StateFeatures.extract_features(state)
        return self.predict(features)

    def predict_from_outcomes(self, outcomes: List[DealOutcome]) -> np.ndarray:
        """Predict for multiple outcomes."""
        return np.array([self.predict(o.initial_features) for o in outcomes])

    def save(self, path: str):
        """Save model to file."""
        if not self.trained:
            raise ValueError("Cannot save untrained model")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'weights': self.weights.tolist(),
            'bias': self.bias,
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
            'target_mean': self.target_mean,
            'target_std': self.target_std,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.weights = np.array(data['weights'])
        self.bias = data['bias']
        self.feature_mean = np.array(data['feature_mean'])
        self.feature_std = np.array(data['feature_std'])
        self.target_mean = data['target_mean']
        self.target_std = data['target_std']
        self.trained = True

        print(f"✓ Model loaded from {path}")


def compare_heuristic_vs_empirical(outcomes: List[DealOutcome],
                                   predictor: LearnedDealQualityPredictor):
    """
    Compare heuristic quality vs empirical quality.

    Shows if hand-crafted heuristics align with actual solver performance.
    """
    print("\n" + "=" * 70)
    print("Heuristic vs Empirical Comparison")
    print("=" * 70)

    # Extract data
    heuristic_qualities = np.array([o.heuristic_quality for o in outcomes])
    avg_cards = np.array([o.avg_cards for o in outcomes])
    avg_scores = np.array([o.avg_score for o in outcomes])
    empirical_predictions = predictor.predict_from_outcomes(outcomes)

    # Correlations
    corr_h_cards = np.corrcoef(heuristic_qualities, avg_cards)[0, 1]
    corr_h_scores = np.corrcoef(heuristic_qualities, avg_scores)[0, 1]
    corr_e_cards = np.corrcoef(empirical_predictions, avg_cards)[0, 1]
    corr_e_scores = np.corrcoef(empirical_predictions, avg_scores)[0, 1]

    print(f"\nCorrelations with Actual Performance:")
    print(f"  Heuristic → Avg Cards: {corr_h_cards:.3f}")
    print(f"  Heuristic → Avg Score: {corr_h_scores:.3f}")
    print(f"  Empirical → Avg Cards: {corr_e_cards:.3f}")
    print(f"  Empirical → Avg Score: {corr_e_scores:.3f}")

    # Find deals where heuristic and empirical disagree
    print(f"\nDisagreements (Heuristic HIGH, Empirical LOW):")

    # Normalize both to 0-1 for comparison
    h_norm = (heuristic_qualities - np.min(heuristic_qualities)) / (np.max(heuristic_qualities) - np.min(heuristic_qualities))
    e_norm = (empirical_predictions - np.min(empirical_predictions)) / (np.max(empirical_predictions) - np.min(empirical_predictions))

    disagreements = h_norm - e_norm
    top_disagree = np.argsort(disagreements)[-5:][::-1]

    for idx in top_disagree:
        o = outcomes[idx]
        print(f"  Seed {o.seed}: Heuristic {o.heuristic_quality:.1f}, "
              f"Predicted {empirical_predictions[idx]:.1f} cards, "
              f"Actual {o.avg_cards:.1f} cards")

    print(f"\nDisagreements (Heuristic LOW, Empirical HIGH):")
    bottom_disagree = np.argsort(disagreements)[:5]

    for idx in bottom_disagree:
        o = outcomes[idx]
        print(f"  Seed {o.seed}: Heuristic {o.heuristic_quality:.1f}, "
              f"Predicted {empirical_predictions[idx]:.1f} cards, "
              f"Actual {o.avg_cards:.1f} cards")


if __name__ == "__main__":
    from optimization.solvers.heuristic_solver import HeuristicSolver

    print("=" * 70)
    print("Empirical Deal Quality Evaluation")
    print("=" * 70)
    print()

    # Create solvers
    solvers = {
        'Heuristic': HeuristicSolver(),
    }

    # Try to load Q-Learning if available
    try:
        from optimization.solvers.rl.q_learning import QPolicy
        q_policy = QPolicy()
        if Path("policies/q_policy.json").exists():
            q_policy.load("policies/q_policy.json")
            # Would need a PolicySolver wrapper
            print("Q-Learning policy loaded (not yet integrated)")
    except:
        pass

    # Evaluate deals
    evaluator = EmpiricalDealEvaluator(solvers)

    print("Evaluating 100 deals empirically...")
    outcomes = evaluator.evaluate_many_deals(num_deals=100, start_seed=5000, verbose=True)

    # Save dataset
    evaluator.save_dataset(outcomes, "data/deal_outcomes.json")

    # Train predictor
    print("\n" + "=" * 70)
    predictor = LearnedDealQualityPredictor()
    predictor.train(outcomes, target='avg_cards', verbose=True)
    predictor.save("models/deal_quality_predictor.json")

    # Compare
    compare_heuristic_vs_empirical(outcomes, predictor)

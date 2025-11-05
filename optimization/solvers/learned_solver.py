"""
Solver that uses machine-learned evaluation functions.

This solver uses neural networks or other learned models
to evaluate game states, rather than hand-crafted heuristics.
"""

from typing import Optional, List
import numpy as np

from game.core.game import Game
from game.core.moves import Move
from game.core.state import GameState
from game.core.rules import get_valid_moves, apply_move
from optimization.solvers.base import Solver
from optimization.features.state_features import StateFeatures
from optimization.evaluation.learned_evaluation import NeuralEvaluationFunction


class LearnedEvaluationSolver(Solver):
    """
    Solver that uses a learned evaluation function.

    At each state, looks ahead 1 move and chooses the move
    that leads to the highest-valued next state according
    to the learned model.
    """

    def __init__(self,
                 evaluation_function: NeuralEvaluationFunction,
                 name: str = "Learned-Eval"):
        """
        Initialize learned evaluation solver.

        Args:
            evaluation_function: Trained neural network
            name: Solver name
        """
        super().__init__(name)
        self.eval_fn = evaluation_function

        if not evaluation_function.trained:
            raise ValueError("Evaluation function must be trained first")

    def evaluate_state(self, state: GameState) -> float:
        """
        Evaluate a game state using learned function.

        Args:
            state: Game state to evaluate

        Returns:
            Predicted value
        """
        features = StateFeatures.extract_features(state)

        # Normalize using training statistics
        features_norm = (features - self.eval_fn.input_mean) / self.eval_fn.input_std

        # Predict
        value_norm = self.eval_fn.predict(features_norm)

        # Denormalize
        value = value_norm * self.eval_fn.target_std + self.eval_fn.target_mean

        return value

    def choose_move(self, game: Game) -> Optional[Move]:
        """
        Choose best move using learned evaluation function.

        Strategy:
        1. Get all valid moves
        2. For each move, simulate and evaluate resulting state
        3. Choose move leading to highest-valued state

        Args:
            game: Current game

        Returns:
            Best move according to learned evaluation
        """
        valid_moves = get_valid_moves(game.state)

        if not valid_moves:
            return None

        # Evaluate each move
        best_move = None
        best_value = float('-inf')

        for move in valid_moves:
            # Simulate move
            next_state = apply_move(game.state, move)

            # Evaluate resulting state
            value = self.evaluate_state(next_state)

            # Add immediate reward (change in score)
            immediate_reward = next_state.score - game.state.score
            total_value = immediate_reward + value

            if total_value > best_value:
                best_value = total_value
                best_move = move

        return best_move


class EnsembleSolver(Solver):
    """
    Combines multiple evaluation functions in an ensemble.

    Uses weighted average of predictions from multiple models.
    """

    def __init__(self,
                 evaluators: List[tuple],  # [(weight, evaluator), ...]
                 name: str = "Ensemble"):
        """
        Initialize ensemble solver.

        Args:
            evaluators: List of (weight, evaluator) tuples
            name: Solver name
        """
        super().__init__(name)
        self.evaluators = evaluators

        # Normalize weights
        total_weight = sum(w for w, _ in evaluators)
        self.evaluators = [(w / total_weight, e) for w, e in evaluators]

    def evaluate_state(self, state: GameState) -> float:
        """Evaluate using weighted ensemble."""
        total_value = 0.0

        for weight, evaluator in self.evaluators:
            if isinstance(evaluator, LearnedEvaluationSolver):
                value = evaluator.evaluate_state(state)
            elif callable(evaluator):
                # Custom evaluation function
                value = evaluator(state)
            else:
                raise ValueError(f"Unknown evaluator type: {type(evaluator)}")

            total_value += weight * value

        return total_value

    def choose_move(self, game: Game) -> Optional[Move]:
        """Choose move using ensemble evaluation."""
        valid_moves = get_valid_moves(game.state)

        if not valid_moves:
            return None

        best_move = None
        best_value = float('-inf')

        for move in valid_moves:
            next_state = apply_move(game.state, move)
            value = self.evaluate_state(next_state)

            immediate_reward = next_state.score - game.state.score
            total_value = immediate_reward + value

            if total_value > best_value:
                best_value = total_value
                best_move = move

        return best_move


def compare_learned_vs_heuristic(num_games: int = 20,
                                 eval_path: str = "models/neural_eval.json"):
    """
    Compare learned evaluation function vs heuristic solver.

    Args:
        num_games: Number of games to test
        eval_path: Path to trained evaluation function
    """
    from optimization.solvers.heuristic_solver import HeuristicSolver

    print("=" * 70)
    print("Learned Evaluation vs Heuristic Comparison")
    print("=" * 70)
    print()

    # Load learned evaluator
    print(f"Loading learned evaluation from {eval_path}...")
    eval_fn = NeuralEvaluationFunction()
    eval_fn.load(eval_path)

    learned_solver = LearnedEvaluationSolver(eval_fn)
    heuristic_solver = HeuristicSolver()

    # Test on games
    learned_scores = []
    learned_cards = []
    heuristic_scores = []
    heuristic_cards = []

    print(f"\nTesting on {num_games} games...")
    print()

    for seed in range(1000, 1000 + num_games):
        # Test learned solver
        game = Game(seed=seed)
        game.deal()

        moves = 0
        while moves < 500:
            move = learned_solver.choose_move(game)
            if not move or not game.make_move(move):
                break
            moves += 1

        learned_scores.append(game.state.score)
        learned_cards.append(game.state.get_foundation_count())

        # Test heuristic solver
        game = Game(seed=seed)
        game.deal()

        moves = 0
        while moves < 500:
            move = heuristic_solver.choose_move(game)
            if not move or not game.make_move(move):
                break
            moves += 1

        heuristic_scores.append(game.state.score)
        heuristic_cards.append(game.state.get_foundation_count())

        if (seed - 999) % 5 == 0:
            print(f"  Game {seed}: "
                  f"Learned {learned_cards[-1]}/52, "
                  f"Heuristic {heuristic_cards[-1]}/52")

    # Results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\nLearned Evaluation Solver:")
    print(f"  Avg score: ${np.mean(learned_scores):.1f}")
    print(f"  Avg cards: {np.mean(learned_cards):.1f}/52")
    print(f"  Win rate: {np.sum(np.array(learned_cards) == 52) / len(learned_cards) * 100:.1f}%")

    print(f"\nHeuristic Solver:")
    print(f"  Avg score: ${np.mean(heuristic_scores):.1f}")
    print(f"  Avg cards: {np.mean(heuristic_cards):.1f}/52")
    print(f"  Win rate: {np.sum(np.array(heuristic_cards) == 52) / len(heuristic_cards) * 100:.1f}%")

    # Comparison
    score_diff = np.mean(learned_scores) - np.mean(heuristic_scores)
    cards_diff = np.mean(learned_cards) - np.mean(heuristic_cards)

    print(f"\nDifference (Learned - Heuristic):")
    print(f"  Score: ${score_diff:+.1f}")
    print(f"  Cards: {cards_diff:+.1f}")

    if cards_diff > 0:
        print(f"\n✓ Learned evaluation is better by {cards_diff:.1f} cards on average!")
    elif cards_diff < 0:
        print(f"\n✗ Heuristic is better by {-cards_diff:.1f} cards on average")
    else:
        print(f"\n→ Both perform equally")


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_learned_vs_heuristic(num_games=20)
    else:
        print("Usage:")
        print("  python learned_solver.py --compare")
        print()
        print("This will compare learned evaluation vs heuristic solver.")
        print("First run: python -m optimization.evaluation.learned_evaluation")
        print("to generate training data and train the model.")

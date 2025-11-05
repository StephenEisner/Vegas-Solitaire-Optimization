"""
Strategic solver that incorporates domain insights.

User insights:
1. Sometimes prioritize SHORT stacks (to create empty columns for Kings)
2. Sometimes prioritize LONG stacks (to reveal deeply hidden cards)
3. Context matters - different strategies for different game phases
"""

from typing import List, Optional, Tuple
from game.core.game import Game
from game.core.state import GameState
from game.core.moves import Move, MoveType
from game.core.rules import get_valid_moves
from game.core.card import Rank
from optimization.solvers.base import Solver


class StrategicSolver(Solver):
    """
    Solver that uses strategic insights for move prioritization.

    Combines heuristic evaluation with context-aware strategy selection.
    """

    def __init__(self, base_solver: Optional[Solver] = None):
        """
        Initialize strategic solver.

        Args:
            base_solver: Underlying solver to use (optional)
        """
        super().__init__(name="Strategic")
        self.base_solver = base_solver

    def estimate_game_phase(self, state: GameState) -> str:
        """
        Determine current game phase.

        Returns:
            'early', 'mid', or 'late'
        """
        foundation_count = state.get_foundation_count()
        passes = state.passes_through_deck

        if foundation_count < 10 and passes == 0:
            return 'early'  # Still exploring, revealing cards
        elif foundation_count < 25 and passes <= 2:
            return 'mid'    # Building foundations actively
        else:
            return 'late'   # Finishing or stuck

    def count_kings_available(self, state: GameState) -> int:
        """Count Kings that are visible and could be moved."""
        kings = 0

        # Tableau Kings
        for col in range(7):
            visible = state.get_tableau_visible_cards(col)
            for card in visible:
                if card.rank == Rank.KING:
                    # King is movable if it's the top of a sequence
                    # or if there's an empty column
                    kings += 1

        # Waste King
        if state.waste and state.waste[-1].rank == Rank.KING:
            kings += 1

        return kings

    def analyze_stack_context(self, state: GameState) -> dict:
        """
        Analyze strategic context for stack prioritization.

        User insights:
        - Need empty columns? → Prioritize short stacks
        - Need information? → Prioritize long stacks
        """
        empty_cols = sum(1 for col in state.tableau if len(col) == 0)
        kings_available = self.count_kings_available(state)
        game_phase = self.estimate_game_phase(state)

        # Strategic priorities
        need_empty_columns = (empty_cols == 0 and kings_available > 0)
        need_information = (game_phase == 'early')

        return {
            'empty_cols': empty_cols,
            'kings_available': kings_available,
            'game_phase': game_phase,
            'need_empty_columns': need_empty_columns,
            'need_information': need_information
        }

    def prioritize_tableau_moves(self, state: GameState,
                                 moves: List[Move]) -> List[Tuple[Move, float]]:
        """
        Rank tableau-to-tableau moves based on strategic context.

        Incorporates user insights about short vs long stack strategy.

        Args:
            state: Current game state
            moves: List of valid moves

        Returns:
            List of (move, score) tuples sorted by score (best first)
        """
        context = self.analyze_stack_context(state)

        scored_moves = []

        for move in moves:
            if move.move_type != MoveType.TABLEAU_TO_TABLEAU:
                continue

            score = 0.0

            # Get source column info
            source_col = state.tableau[move.source]
            source_hidden = state.tableau_hidden[move.source]
            source_visible = len(source_col) - source_hidden

            # USER INSIGHT 1: Prioritize SHORT stacks when need empty columns
            if context['need_empty_columns']:
                if source_visible <= 3:  # Short stack
                    score += 25.0
                    if source_visible == move.card_count:  # Will empty the column!
                        score += 40.0  # Very high priority

            # USER INSIGHT 2: Prioritize LONG stacks in early game
            if context['need_information']:
                if source_hidden >= 4:  # Deep hidden cards
                    score += 20.0
                    # Extra bonus for very deep stacks
                    score += source_hidden * 2.0

            # Always value revealing hidden cards
            if source_hidden > 0:
                score += 12.0
                # But more valuable if it reveals from deep stack
                if source_hidden >= 3:
                    score += 8.0

            # Bonus for moving to empty column if we have Kings
            dest_col = state.tableau[move.destination]
            if len(dest_col) == 0:
                if move.card and move.card.rank == Rank.KING:
                    score += 15.0  # King to empty - good!
                else:
                    score -= 20.0  # Non-King to empty - wasteful

            # Penalty for non-progressive moves
            if source_hidden == 0 and len(dest_col) > 0:
                # Not revealing anything, just shuffling
                score -= 10.0
                score -= move.card_count * 3.0  # Bigger penalty for moving more cards

            scored_moves.append((move, score))

        return scored_moves

    def choose_move(self, game: Game) -> Optional[Move]:
        """
        Choose best move using strategic prioritization.

        Args:
            game: Current game

        Returns:
            Best strategic move
        """
        valid_moves = get_valid_moves(game.state)
        if not valid_moves:
            return None

        # Separate moves by type
        foundation_moves = []
        tableau_moves = []
        waste_moves = []
        draw_moves = []

        for move in valid_moves:
            if move.move_type in [MoveType.TABLEAU_TO_FOUNDATION,
                                 MoveType.WASTE_TO_FOUNDATION]:
                foundation_moves.append(move)
            elif move.move_type == MoveType.TABLEAU_TO_TABLEAU:
                tableau_moves.append(move)
            elif move.move_type in [MoveType.WASTE_TO_TABLEAU]:
                waste_moves.append(move)
            elif move.move_type in [MoveType.DRAW, MoveType.RECYCLE]:
                draw_moves.append(move)

        # Priority 1: Foundation moves (always do these)
        if foundation_moves:
            return foundation_moves[0]

        # Priority 2: Strategic tableau moves
        if tableau_moves:
            scored = self.prioritize_tableau_moves(game.state, tableau_moves)
            scored.sort(key=lambda x: x[1], reverse=True)

            # Only take if score is positive (worthwhile)
            if scored and scored[0][1] > 0:
                return scored[0][0]

        # Priority 3: Waste moves (get cards into play)
        if waste_moves:
            return waste_moves[0]

        # Priority 4: Draw (see new cards)
        if draw_moves:
            return draw_moves[0]

        # Fallback: Take any move
        return valid_moves[0]


def compare_strategic_vs_baseline():
    """
    Compare strategic solver against baseline heuristic.

    Tests if user insights improve performance.
    """
    from optimization.solvers.heuristic_solver import HeuristicSolver

    print("=" * 70)
    print("Strategic Solver vs Baseline Heuristic")
    print("=" * 70)

    num_games = 20
    seeds = range(1000, 1000 + num_games)

    strategic_scores = []
    strategic_foundations = []

    heuristic_scores = []
    heuristic_foundations = []

    strategic_solver = StrategicSolver()
    heuristic_solver = HeuristicSolver()

    print(f"\nTesting on {num_games} games...\n")

    for seed in seeds:
        # Test strategic solver
        game = Game(seed=seed)
        game.deal()

        moves = 0
        while moves < 500:
            move = strategic_solver.choose_move(game)
            if not move or not game.make_move(move):
                break
            moves += 1

        strategic_scores.append(game.state.score)
        strategic_foundations.append(game.state.get_foundation_count())

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
        heuristic_foundations.append(game.state.get_foundation_count())

        print(f"  Game {seed}: Strategic ${game.state.score} vs Heuristic ${heuristic_scores[-1]}")

    # Results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\nStrategic Solver:")
    print(f"  Average score: ${np.mean(strategic_scores):.1f}")
    print(f"  Average foundation: {np.mean(strategic_foundations):.1f}/52")

    print(f"\nBaseline Heuristic:")
    print(f"  Average score: ${np.mean(heuristic_scores):.1f}")
    print(f"  Average foundation: {np.mean(heuristic_foundations):.1f}/52")

    # Comparison
    score_improvement = np.mean(strategic_scores) - np.mean(heuristic_scores)
    foundation_improvement = np.mean(strategic_foundations) - np.mean(heuristic_foundations)

    print(f"\nImprovement:")
    print(f"  Score: ${score_improvement:+.1f} ({score_improvement/abs(np.mean(heuristic_scores))*100:+.1f}%)")
    print(f"  Foundation: {foundation_improvement:+.1f} cards")


def demo_strategic_decisions():
    """
    Demonstrate strategic decision-making in action.
    """
    from optimization.meta.deal_evaluator import evaluate_deal_quality

    print("=" * 70)
    print("Strategic Decision-Making Demo")
    print("=" * 70)

    solver = StrategicSolver()

    # Find an interesting game state
    game = Game(seed=42)
    game.deal()

    # Play a few moves to get interesting state
    for _ in range(10):
        move = solver.choose_move(game)
        if not move:
            break
        game.make_move(move)

    print("\nCurrent Game State:")
    print(f"  Foundation: {game.state.get_foundation_count()}/52")
    print(f"  Score: ${game.state.score}")

    # Analyze context
    context = solver.analyze_stack_context(game.state)
    print(f"\nStrategic Context:")
    print(f"  Game phase: {context['game_phase']}")
    print(f"  Empty columns: {context['empty_cols']}")
    print(f"  Kings available: {context['kings_available']}")
    print(f"  Need empty columns: {context['need_empty_columns']}")
    print(f"  Need information: {context['need_information']}")

    # Show move prioritization
    valid_moves = get_valid_moves(game.state)
    tableau_moves = [m for m in valid_moves
                     if m.move_type == MoveType.TABLEAU_TO_TABLEAU]

    if tableau_moves:
        print(f"\nTableau Move Prioritization:")
        scored = solver.prioritize_tableau_moves(game.state, tableau_moves)
        scored.sort(key=lambda x: x[1], reverse=True)

        for i, (move, score) in enumerate(scored[:5], 1):
            print(f"  {i}. {move} (score: {score:.1f})")


if __name__ == "__main__":
    import numpy as np

    # Run demos
    demo_strategic_decisions()

    print("\n\n")
    compare_strategic_vs_baseline()

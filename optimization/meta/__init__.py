"""
Meta-level strategies for Vegas Solitaire.

This module contains high-level strategies:
- Deal evaluation and selection
- Multi-deal mode with bankroll management
- Strategic solver with domain insights
- AI-powered deal strategies
"""

from optimization.meta.deal_evaluator import (
    evaluate_deal_quality,
    DealComparator,
)

from optimization.meta.deal_selector import (
    DealSelector,
)

from optimization.meta.strategic_solver import (
    StrategicSolver,
)

from optimization.meta.multideal_mode import (
    MultidealManager,
    DealAcceptor,
    MultidealSession,
    DealResult,
)

from optimization.meta.ai_deal_strategy import (
    LearnedDealSelector,
    LearnedDealAcceptor,
    HybridDealStrategy,
    create_ai_deal_strategy,
)

__all__ = [
    'evaluate_deal_quality',
    'DealComparator',
    'DealSelector',
    'StrategicSolver',
    'MultidealManager',
    'DealAcceptor',
    'MultidealSession',
    'DealResult',
    'LearnedDealSelector',
    'LearnedDealAcceptor',
    'HybridDealStrategy',
    'create_ai_deal_strategy',
]

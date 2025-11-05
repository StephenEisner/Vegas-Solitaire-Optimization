# Game Implementation

This directory contains the playable implementation of Vegas Solitaire.

## Goals

- Create a fully functional Vegas Solitaire game
- Ensure correct rule implementation
- Provide interfaces for both human play and algorithmic testing
- Enable game state serialization for training/optimization

## Architecture

```
game/
├── README.md           # This file
├── core/
│   ├── card.py        # Card and Deck classes
│   ├── game.py        # Main game logic
│   ├── state.py       # Game state representation
│   └── rules.py       # Rule validation
├── ui/
│   ├── cli.py         # Command-line interface
│   └── gui.py         # (Optional) Graphical interface
├── tests/
│   └── test_game.py   # Unit tests
└── requirements.txt
```

## Core Components

### Card & Deck (`core/card.py`)
- Card class with suit, rank, and color
- Deck class with shuffling and dealing
- Standard 52-card deck representation

### Game State (`core/state.py`)
- Complete game state representation
- Stock (draw pile)
- Waste pile (drawn cards)
- Tableau (7 columns)
- Foundations (4 suit piles)
- Move history
- State serialization for ML training

### Game Logic (`core/game.py`)
- Main game controller
- Move validation and execution
- Win/loss detection
- Score calculation (Vegas rules: -$52 + $5 per foundation card)

### Rules (`core/rules.py`)
- Valid move checking
- Tableau building rules (descending, alternating colors)
- Foundation building rules (ascending, same suit)
- Stock/waste management (draw 3)

## Game Variants to Consider

### Draw Rules
- Draw 3 (standard Vegas)
- Draw 1 (easier variant)
- Unlimited redeals vs limited redeals

### Scoring
- Vegas mode: Pay $52, earn $5 per foundation card (goal: profit)
- Standard mode: Point-based scoring
- Win/loss only

## Interface Options

### Command-Line Interface (Priority)
Simple text-based interface for testing and validation:
```
python -m game.ui.cli
```

### Programmatic API
For optimization algorithms to play games:
```python
from game.core import Game

game = Game()
game.deal()
valid_moves = game.get_valid_moves()
game.make_move(move)
```

### Web Interface (Later)
Could be integrated into the final website/visualization

## Testing Strategy

- Unit tests for all move validations
- Integration tests for full games
- Edge cases (empty piles, multiple valid moves, etc.)
- Random game playouts to ensure stability

## Installation

```bash
cd game
pip install -r requirements.txt
```

## Usage Examples

### Playing a Game
```python
from game.core import Game

game = Game(seed=42)
game.deal()

while not game.is_over():
    moves = game.get_valid_moves()
    # Choose move (human or algorithm)
    game.make_move(selected_move)

print(f"Final score: ${game.score}")
print(f"Cards in foundations: {game.foundation_count}")
```

### State Serialization
```python
# Save state for training
state_dict = game.serialize()

# Restore state
new_game = Game.deserialize(state_dict)
```

## Development Priorities

1. **Core game logic** - Get the rules right
2. **Move generation** - Efficiently find all valid moves
3. **State representation** - Clean API for algorithms
4. **CLI for testing** - Quick validation during development
5. **Comprehensive tests** - Ensure correctness

## Performance Considerations

- Fast move validation (for MCTS simulations)
- Efficient state copying (for tree search)
- Minimal memory footprint
- Support for batched operations (for RL training)

## Next Steps

Once the game implementation is complete:
- Validate against known game scenarios
- Benchmark performance (moves per second)
- Create dataset of sample games
- Prepare for integration with optimization algorithms

## Design Decisions

### Move Representation
Moves should encode:
- Source location (tableau, waste, foundation)
- Destination location
- Number of cards (for tableau sequences)

### State Representation
Should be:
- Hashable (for state caching)
- Serializable (for training datasets)
- Fast to copy (for tree search)

### Random Number Generation
- Seeded RNG for reproducibility
- Support deterministic testing

## Future Enhancements

- Game replay/undo functionality
- Move hints for human players
- Statistics tracking (win rate, average score, etc.)
- Support for variations (Klondike, etc.)

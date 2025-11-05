# Vegas Solitaire Optimization: Quickstart Guide

Get up and running with the Vegas Solitaire project quickly.

## Prerequisites

- Python 3.8+ 
- Git
- (Optional) Node.js 16+ for website development
- (Optional) GPU access for training (Colab/Kaggle work fine)

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vegas-solitaire.git
cd vegas-solitaire
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Tour

### Play a Game

```bash
cd game
python -m ui.cli
```

This launches a command-line interface where you can play Vegas Solitaire manually.

### Run a Simple Solver

```bash
cd optimization
python solvers/random.py --games 100
```

This runs 100 games with random moves to establish a baseline.

### View Statistics

```bash
python evaluation/statistics.py --results baseline_random.json
```

## Project Structure at a Glance

```
vegas-solitaire/
├── game/              # Core game implementation
│   ├── core/         # Game logic, state, rules
│   └── ui/           # Command-line interface
├── optimization/      # Strategy optimization
│   ├── solvers/      # Different solving approaches
│   ├── training/     # ML training scripts
│   └── evaluation/   # Benchmarking and analysis
├── website/          # Interactive tutorial
│   ├── pages/        # Content sections
│   └── components/   # React components
└── docs/             # Documentation
    └── rules.md      # Complete game rules
```

## Development Workflow

### Phase 1: Build the Game (Start Here)

1. **Implement core game logic** (`game/core/`)
   - Card and Deck classes
   - Game state representation
   - Move validation and execution

2. **Test thoroughly** (`game/tests/`)
   - Unit tests for each component
   - Integration tests for full games
   - Edge cases

3. **Create CLI** (`game/ui/cli.py`)
   - Playable interface for testing
   - Debug utilities

### Phase 2: Build Optimization Tools

1. **Create baselines** (`optimization/solvers/`)
   - Random player
   - Simple heuristic player
   - Benchmark performance

2. **Implement MCTS** (`optimization/solvers/mcts.py`)
   - Basic tree search
   - Optimize with caching
   - Evaluate strength

3. **Set up RL training** (`optimization/training/`)
   - Define gym environment
   - Implement policy network
   - Self-play loop

### Phase 3: Build Website/Tutorial

1. **Plan content** (`website/pages/`)
   - Write explanations
   - Design interactive elements

2. **Implement components** (`website/components/`)
   - Game board visualization
   - Policy visualizations
   - Training charts

3. **Integrate everything**
   - Connect game to frontend
   - Load trained models
   - Deploy

## Common Commands

### Development

```bash
# Run tests
pytest game/tests/

# Format code
black .

# Type checking
mypy game/

# Lint
flake8 game/ optimization/
```

### Experimentation

```bash
# Train a model
python optimization/training/train_policy.py --config configs/default.yaml

# Evaluate a model
python optimization/evaluation/benchmark.py --model checkpoints/model_1000.pt

# Compare strategies
python optimization/evaluation/compare.py --baseline random --test mcts
```

### Website Development

```bash
cd website

# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build
```

## Key Files to Understand

### Game Implementation
- `game/core/game.py` - Main game controller
- `game/core/state.py` - State representation
- `game/core/rules.py` - Move validation logic

### Optimization
- `optimization/solvers/mcts.py` - Monte Carlo Tree Search
- `optimization/training/train_policy.py` - RL training loop
- `optimization/models/networks.py` - Neural network architectures

### Website
- `website/components/GameBoard.jsx` - Visual game display
- `website/pages/05-mcts.md` - MCTS explanation
- `website/api/inference.js` - Model prediction API

## Tips for Success

### Starting Out
1. **Read the rules** (`docs/rules.md`) thoroughly
2. **Play some games** manually to understand strategy
3. **Start with tests** - write tests as you build
4. **Keep it simple** - get basics working before optimizing

### Debugging Games
- Add `--seed SEED` flag to reproduce specific games
- Use `--verbose` mode to see all moves
- Implement game replay to step through problematic games
- Visualize game states for complex debugging

### Training Models
- Start with small networks and short training runs
- Monitor loss curves - if they don't decrease, something's wrong
- Validate on held-out games frequently
- Save checkpoints often
- Use TensorBoard for real-time monitoring

### Building the Website
- Start with static content before adding interactivity
- Test on mobile devices early
- Optimize bundle size (code splitting, lazy loading)
- User test with non-technical people

## Troubleshooting

### "Game hangs during random play"
- Check for infinite loops in move generation
- Ensure game-over detection works
- Add move counter safeguard

### "Model doesn't learn anything"
- Verify reward signal is correct
- Check input normalization
- Ensure exploration is happening
- Try simpler task first (e.g., always go to foundation)

### "Website loads slowly"
- Minimize model size
- Use code splitting
- Compress assets
- Consider server-side inference

## Resources

### Learning Resources
- **Reinforcement Learning**: Sutton & Barto textbook
- **MCTS**: Browne et al. survey paper
- **Neural Networks**: Fast.ai courses
- **React**: Official React documentation

### Similar Projects
- AlphaZero for chess/Go
- OpenSpiel for game research
- Gym environments for RL

### Tools
- **PyTorch**: ML framework
- **Ray RLlib**: Scalable RL
- **Streamlit**: Quick web prototypes
- **Jupyter**: Interactive development

## Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check `/docs` folder
- **Examples**: See `/examples` for code samples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

## Next Steps

Ready to start coding? Here's the recommended path:

1. ✅ Read this quickstart
2. ✅ Review complete rules (`docs/rules.md`)
3. ➡️ Implement game logic (`game/core/`)
4. ➡️ Write tests (`game/tests/`)
5. ➡️ Build CLI (`game/ui/cli.py`)
6. ➡️ Move to optimization phase

Or jump directly into the section that interests you most!

---

**Welcome to the Vegas Solitaire Optimization Project!** 

This is a journey into game theory, optimization, and machine learning through the lens of a classic card game. Whether you're here to learn, experiment, or contribute, we hope you enjoy the process.

Happy coding! 🃏🤖

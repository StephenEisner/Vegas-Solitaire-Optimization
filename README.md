# Vegas Solitaire: Game Theory & Optimization

An interactive exploration of Vegas Solitaire through gameplay, mathematical analysis, and reinforcement learning optimization.

## Project Vision

This project aims to create an educational and interactive experience that:
1. Teaches the rules and strategies of Vegas Solitaire
2. Implements optimization algorithms to find optimal play strategies
3. Visualizes the game and decision-making process
4. Demonstrates practical applications of game theory and ML

## Project Structure

```
vegas-solitaire/
├── README.md                 # This file
├── game/                     # Playable game implementation
│   └── README.md
├── optimization/             # Strategy optimization algorithms
│   └── README.md
├── website/                  # Interactive tutorial/visualization
│   └── README.md
└── docs/                     # Additional documentation
    ├── rules.md              # Complete game rules
    └── algorithms.md         # Deep dive on search algorithms
```

## Development Phases

### Phase 1: Game Implementation ✓ (Planned)
- Implement core Vegas Solitaire game logic
- Create playable CLI/GUI version
- Validate game mechanics and rules
- **Deliverable**: Functional game that can be played and tested

### Phase 2: Optimization Algorithms (Planned)
- Implement game state representation
- Develop Monte Carlo Tree Search (MCTS)
- Implement reinforcement learning approaches
- Strategy evaluation and comparison
- **Deliverable**: Python programs that can analyze and optimize gameplay

### Phase 3: Integration & Visualization (Planned)
- Create interactive website/Jupyter notebook
- Integrate optimization algorithms
- Connect to cloud GPU resources for training
- Generate and display optimal policies
- **Deliverable**: Educational interactive experience

## Technology Stack

- **Game**: Python (with potential web port)
- **Optimization**: Python, NumPy, PyTorch/JAX
- **Visualization**: Jupyter notebooks or React/HTML5
- **Compute**: Free cloud GPU (Colab, Kaggle, etc.)

## Game Rules Quick Reference

Vegas Solitaire is played with one deck of cards. The objective is to move all cards to four foundation piles (Ace through King, by suit).

- **Tableau**: 7 columns, cards can be moved in descending order with alternating colors
- **Waste Pile**: Draw 3 cards at a time from the stock
- **Foundations**: Build up from Ace to King by suit
- **Scoring**: In Vegas mode, you pay $52 to play and earn $5 per card in foundations

See `docs/rules.md` for complete rules.

## Getting Started

Each subdirectory contains its own README with specific setup instructions. Start with:

1. `game/` - Build and test the game
2. `optimization/` - Develop optimization algorithms
3. `website/` - Create the interactive experience

## Contributing

This is an open source project. Contributions, ideas, and improvements are welcome!

## License

[Choose your license - MIT, GPL, etc.]

## Roadmap

- [ ] Implement playable Vegas Solitaire game
- [ ] Create game state evaluation framework
- [ ] Implement MCTS solver
- [ ] Implement RL-based solver
- [ ] Build interactive tutorial website
- [ ] Train optimal policies on cloud GPU
- [ ] Publish findings and strategies

## Contact

[Your contact information]

---

*An exploration of optimal decision-making in stochastic games*

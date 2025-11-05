"""Analysis Mode - Deep dive into game statistics and solver performance."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from game.core.game import Game
from optimization.solvers.random_solver import RandomSolver
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.solvers.mcts_solver import MCTSSolver


def show():
    """Display the Analysis Mode page."""
    st.title("📈 Analysis Mode")
    st.markdown("Deep dive into game statistics, move patterns, and solver behavior.")

    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["🎮 Game Analysis", "🤖 Solver Analysis", "📊 Statistics"])

    with tab1:
        show_game_analysis()

    with tab2:
        show_solver_analysis()

    with tab3:
        show_statistics()


def show_game_analysis():
    """Analyze a specific game."""
    st.markdown("### 🎮 Single Game Analysis")

    st.markdown("""
    Analyze how different solvers perform on the same game seed.
    This helps understand solver decision-making on specific board states.
    """)

    col1, col2 = st.columns(2)

    with col1:
        seed = st.number_input("Game Seed", min_value=0, max_value=10000, value=42)

    with col2:
        max_moves = st.number_input("Max Moves", min_value=100, max_value=2000, value=1000, step=100)

    if st.button("🔍 Analyze Game", type="primary"):
        with st.spinner("Analyzing game with all solvers..."):
            # Create solvers
            solvers = [
                ("Random", RandomSolver(seed=None)),
                ("Heuristic", HeuristicSolver()),
                ("MCTS(100)", MCTSSolver(simulations_per_move=100, seed=None)),
            ]

            results = []

            for solver_name, solver in solvers:
                game = Game(seed=seed)
                moves_made = 0

                while moves_made < max_moves:
                    valid_moves = game.get_valid_moves()
                    if not valid_moves:
                        break

                    move = solver.choose_move(game)
                    if not move:
                        break

                    game.apply_move(move)
                    moves_made += 1

                results.append({
                    "Solver": solver_name,
                    "Final Score": game.state.score,
                    "Foundation Cards": game.state.foundation_count,
                    "Moves Made": moves_made,
                    "Won": game.state.foundation_count == 52,
                    "Win Value": "$260" if game.state.foundation_count == 52 else f"${game.state.score}"
                })

            df_results = pd.DataFrame(results)

            # Display results
            st.markdown("#### Results")
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Visualization
            col1, col2 = st.columns(2)

            with col1:
                fig_score = px.bar(
                    df_results,
                    x="Solver",
                    y="Final Score",
                    color="Won",
                    title="Final Score by Solver",
                    text="Win Value"
                )
                st.plotly_chart(fig_score, use_container_width=True)

            with col2:
                fig_foundation = px.bar(
                    df_results,
                    x="Solver",
                    y="Foundation Cards",
                    title="Foundation Cards Placed",
                    text="Foundation Cards"
                )
                fig_foundation.add_hline(y=52, line_dash="dash", line_color="green", annotation_text="Win")
                st.plotly_chart(fig_foundation, use_container_width=True)


def show_solver_analysis():
    """Analyze solver behavior and strategies."""
    st.markdown("### 🤖 Solver Behavior Analysis")

    st.markdown("""
    Understand how different solvers make decisions and their strategic patterns.
    """)

    # Solver characteristics
    st.markdown("#### Solver Characteristics")

    solver_data = {
        "Solver": ["Random", "Heuristic", "MCTS(100)", "MCTS(1000)"],
        "Strategy": ["Uniform random", "Rule-based heuristics", "Tree search", "Tree search"],
        "Time per Move": ["<1ms", "~5ms", "~50ms", "~500ms"],
        "Deterministic": ["No", "Yes", "No (with seed)", "No (with seed)"],
        "Expected Win Rate": ["2-5%", "5-15%", "20-35%", "30-45%"],
        "Best For": ["Baseline", "Fast decisions", "Good balance", "Best performance"]
    }

    df_solvers = pd.DataFrame(solver_data)
    st.dataframe(df_solvers, use_container_width=True, hide_index=True)

    # Move type preferences
    st.markdown("#### Move Type Preferences")

    st.markdown("""
    Different solvers have different preferences for move types:

    - **Random Solver**: Equal preference for all moves
    - **Heuristic Solver**: Prioritizes foundation moves, then reveals, then tableau building
    - **MCTS Solver**: Learns optimal move preferences through simulation
    """)

    # Heuristic weights
    with st.expander("🧠 Heuristic Solver Weights"):
        st.markdown("""
        The Heuristic Solver uses these weighted rules:

        | Move Type | Weight | Reasoning |
        |-----------|--------|-----------|
        | To Foundation | 100 | Directly progresses toward win |
        | Reveal Face-down | 50 | Unlocks new options |
        | Build Empty Column | 80 | Creates space for Kings |
        | Tableau Building | 10-30 | Organize cards |
        | Draw from Stock | 5 | Generate new options |
        """)

    # MCTS parameters
    with st.expander("🌲 MCTS Solver Parameters"):
        st.markdown("""
        MCTS uses these key parameters:

        | Parameter | Default | Description |
        |-----------|---------|-------------|
        | Simulations per Move | 1000 | More = better but slower |
        | Exploration Constant | 1.41 | Balance exploration vs exploitation |
        | Max Simulation Depth | 500 | Limits simulation length |
        | Reward Function | (score+52)/200 | Normalizes Vegas score to [0,1] |

        **UCB1 Formula**: exploitation + c × √(ln(parent_visits) / node_visits)
        """)


def show_statistics():
    """Show general game statistics and probabilities."""
    st.markdown("### 📊 Vegas Solitaire Statistics")

    # Game complexity
    st.markdown("#### Game Complexity")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Cards", "52")
        st.caption("Standard deck")

    with col2:
        st.metric("Possible Deals", "52!")
        st.caption("≈ 8 × 10⁶⁷ combinations")

    with col3:
        st.metric("Avg Game Length", "~150 moves")
        st.caption("Varies by solver")

    # Win rates
    st.markdown("#### Theoretical Win Rates")

    st.markdown("""
    Vegas Solitaire (Draw-3) is notoriously difficult:

    - **Perfect Play (estimated)**: 15-25% of deals are winnable
    - **Human Expert**: 8-15% win rate
    - **Random Play**: 2-5% win rate
    - **Current MCTS**: 30-40% (target)
    """)

    # Scoring breakdown
    st.markdown("#### Scoring Breakdown")

    scoring_data = {
        "Event": ["Game Start", "Card to Foundation", "Win Game (52 cards)", "Break Even", "Best Possible"],
        "Score Change": ["-$52", "+$5", "+$260", "$0", "$260"],
        "Total Score": ["-$52", "Cumulative", "$208", "$0", "$208"],
        "Foundation Cards": ["0", "Variable", "52", "~10-11", "52"]
    }

    df_scoring = pd.DataFrame(scoring_data)
    st.dataframe(df_scoring, use_container_width=True, hide_index=True)

    st.info("""
    **Break-even point**: You need to place ~10-11 cards on the foundation to recover your $52 bet.
    Any game with less than 11 cards on foundation results in a net loss.
    """)

    # Move distribution
    st.markdown("#### Typical Move Distribution")

    st.markdown("""
    In an average game, move types break down approximately as:

    - **Draw**: 15-25% (limited by 3 passes through deck)
    - **Waste to Tableau**: 20-30%
    - **Waste to Foundation**: 10-15%
    - **Tableau to Tableau**: 25-35%
    - **Tableau to Foundation**: 15-25%
    - **Foundation to Tableau**: <5% (rare, usually suboptimal)
    """)

    # Project progress
    st.markdown("---")
    st.markdown("#### 🚀 Project Progress")

    progress_data = {
        "Phase": [
            "Phase 1: Game Engine",
            "Phase 2: Baseline Solvers",
            "Phase 3: Advanced Search",
            "Phase 4: Reinforcement Learning",
            "Phase 5: Evaluation",
            "Phase 6: Web Interface"
        ],
        "Status": ["✅ Complete", "✅ Complete", "🟡 In Progress", "⏳ Planned", "⏳ Planned", "🟡 In Progress"],
        "Components": [
            "Card, State, Moves, Rules, CLI",
            "Random, Heuristic, Benchmark",
            "MCTS, Expectimax, Beam Search",
            "DQN, PPO, A3C",
            "Cross-validation, Analysis",
            "Streamlit App"
        ]
    }

    df_progress = pd.DataFrame(progress_data)
    st.dataframe(df_progress, use_container_width=True, hide_index=True)

"""Benchmark Mode - Compare solver performance."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from optimization.solvers.random_solver import RandomSolver
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.solvers.mcts_solver import MCTSSolver
from optimization.evaluation.benchmark import Benchmark


def show():
    """Display the Benchmark Mode page."""
    st.title("📊 Benchmark Mode")
    st.markdown("Compare the performance of different AI solvers systematically.")

    # Configuration
    st.markdown("### ⚙️ Benchmark Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_games = st.number_input(
            "Number of Games",
            min_value=1,
            max_value=1000,
            value=10,
            step=10,
            help="More games = more reliable results but slower"
        )

    with col2:
        start_seed = st.number_input(
            "Starting Seed",
            min_value=0,
            max_value=10000,
            value=0,
            help="Seed for reproducible results"
        )

    with col3:
        max_moves = st.number_input(
            "Max Moves per Game",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100
        )

    # Solver selection
    st.markdown("### 🤖 Select Solvers")

    col1, col2, col3 = st.columns(3)

    with col1:
        include_random = st.checkbox("🎲 Random Solver", value=True)
    with col2:
        include_heuristic = st.checkbox("🧠 Heuristic Solver", value=True)
    with col3:
        include_mcts = st.checkbox("🌲 MCTS Solver", value=True)

    if include_mcts:
        mcts_sims = st.select_slider(
            "MCTS Simulations per Move",
            options=[10, 50, 100, 250, 500, 1000],
            value=100,
            help="More simulations = better play but slower"
        )

    # Run benchmark button
    st.markdown("---")

    if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):
        if not (include_random or include_heuristic or include_mcts):
            st.error("⚠️ Please select at least one solver!")
            return

        # Create solvers
        solvers = []

        if include_random:
            solvers.append(RandomSolver(seed=42))

        if include_heuristic:
            solvers.append(HeuristicSolver())

        if include_mcts:
            solvers.append(MCTSSolver(simulations_per_move=mcts_sims, seed=42))

        # Run benchmark
        with st.spinner(f"🎮 Running benchmark with {num_games} games..."):
            benchmark = Benchmark(test_set_name=f"webapp_benchmark")

            # Progress tracking
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            results = []
            for idx, solver in enumerate(solvers):
                status_text.text(f"Testing {solver.__class__.__name__}...")

                result = benchmark.evaluate_solver(
                    solver=solver,
                    num_games=num_games,
                    start_seed=start_seed,
                    max_moves=max_moves,
                    verbose=False
                )
                results.append(result)

                progress = (idx + 1) / len(solvers)
                progress_bar.progress(progress)

            progress_bar.empty()
            status_text.empty()

            # Store results in session state
            st.session_state.benchmark_results = results

        st.success("✅ Benchmark complete!")
        st.rerun()

    # Display results if available
    if 'benchmark_results' in st.session_state:
        results = st.session_state.benchmark_results

        st.markdown("---")
        st.markdown("## 📈 Results")

        # Summary table
        st.markdown("### Summary Statistics")

        summary_data = []
        for result in results:
            summary_data.append({
                "Solver": result.solver_name,
                "Win Rate": f"{result.win_rate:.1%}",
                "Avg Score": f"${result.average_score:.0f}",
                "Avg Moves": f"{result.average_moves:.0f}",
                "Avg Time (ms)": f"{result.average_time_ms:.1f}",
                "Total Time (s)": f"{result.total_time:.1f}",
            })

        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # Detailed statistics
        with st.expander("📊 Detailed Statistics"):
            for result in results:
                st.markdown(f"#### {result.solver_name}")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Games Played", result.games_played)
                    st.metric("Wins", result.wins)

                with col2:
                    st.metric("Win Rate", f"{result.win_rate:.1%}")
                    st.metric("Avg Foundation", f"{result.average_foundation_count:.1f}")

                with col3:
                    st.metric("Avg Score", f"${result.average_score:.0f}")
                    st.metric("Best Score", f"${result.best_score}")

                with col4:
                    st.metric("Avg Moves", f"{result.average_moves:.0f}")
                    st.metric("Avg Time", f"{result.average_time_ms:.1f}ms")

                st.markdown("---")

        # Visualizations
        st.markdown("### 📊 Visualizations")

        # Win rate comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Win Rate Comparison")

            win_rates = [result.win_rate * 100 for result in results]
            solver_names = [result.solver_name for result in results]

            fig_win = go.Figure(data=[
                go.Bar(
                    x=solver_names,
                    y=win_rates,
                    text=[f"{wr:.1f}%" for wr in win_rates],
                    textposition='auto',
                )
            ])

            fig_win.update_layout(
                yaxis_title="Win Rate (%)",
                xaxis_title="Solver",
                height=400
            )

            st.plotly_chart(fig_win, use_container_width=True)

        with col2:
            st.markdown("#### Average Score")

            avg_scores = [result.average_score for result in results]

            fig_score = go.Figure(data=[
                go.Bar(
                    x=solver_names,
                    y=avg_scores,
                    text=[f"${s:.0f}" for s in avg_scores],
                    textposition='auto',
                    marker_color='lightgreen'
                )
            ])

            fig_score.update_layout(
                yaxis_title="Average Score ($)",
                xaxis_title="Solver",
                height=400
            )

            st.plotly_chart(fig_score, use_container_width=True)

        # Performance metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Average Moves per Game")

            avg_moves = [result.average_moves for result in results]

            fig_moves = go.Figure(data=[
                go.Bar(
                    x=solver_names,
                    y=avg_moves,
                    text=[f"{m:.0f}" for m in avg_moves],
                    textposition='auto',
                    marker_color='lightblue'
                )
            ])

            fig_moves.update_layout(
                yaxis_title="Average Moves",
                xaxis_title="Solver",
                height=400
            )

            st.plotly_chart(fig_moves, use_container_width=True)

        with col2:
            st.markdown("#### Average Time per Move")

            avg_time_per_move = []
            for result in results:
                time_per_move = (result.average_time_ms / result.average_moves) if result.average_moves > 0 else 0
                avg_time_per_move.append(time_per_move)

            fig_time = go.Figure(data=[
                go.Bar(
                    x=solver_names,
                    y=avg_time_per_move,
                    text=[f"{t:.2f}ms" for t in avg_time_per_move],
                    textposition='auto',
                    marker_color='lightsalmon'
                )
            ])

            fig_time.update_layout(
                yaxis_title="Time per Move (ms)",
                xaxis_title="Solver",
                height=400
            )

            st.plotly_chart(fig_time, use_container_width=True)

        # Foundation count distribution
        st.markdown("#### Foundation Count Distribution")

        fig_foundation = go.Figure()

        for result in results:
            fig_foundation.add_trace(go.Box(
                y=result.foundation_counts,
                name=result.solver_name,
                boxmean='sd'
            ))

        fig_foundation.update_layout(
            yaxis_title="Cards on Foundation",
            xaxis_title="Solver",
            height=400
        )

        st.plotly_chart(fig_foundation, use_container_width=True)

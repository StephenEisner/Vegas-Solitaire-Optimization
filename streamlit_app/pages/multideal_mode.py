"""
Multi-Deal Mode page for Streamlit app.

Interactive UI for playing multiple deals with bankroll management,
reroll decisions, and AI-powered deal selection.
"""

import streamlit as st
import time
from pathlib import Path

from game.core.game import Game
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.meta.multideal_mode import MultidealManager, DealSelector, DealAcceptor
from optimization.meta.ai_deal_strategy import create_ai_deal_strategy
from streamlit_app.components.board_renderer import render_game_board


def render_multideal_mode():
    """Render the multi-deal mode page."""
    st.title("🎰 Multi-Deal Mode")

    st.markdown("""
    Simulate a realistic casino scenario with:
    - Starting bankroll
    - Limited rerolls (reset after accepting a deal)
    - AI-powered deal selection
    - Profit/loss tracking
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Bankroll settings
    st.sidebar.subheader("Bankroll")
    starting_bankroll = st.sidebar.number_input(
        "Starting Bankroll ($)",
        min_value=100,
        max_value=10000,
        value=500,
        step=50
    )

    cost_per_game = st.sidebar.number_input(
        "Cost per Game ($)",
        min_value=1,
        max_value=100,
        value=52
    )

    reward_per_card = st.sidebar.number_input(
        "Reward per Card ($)",
        min_value=1,
        max_value=10,
        value=5
    )

    # Reroll settings
    st.sidebar.subheader("Reroll Settings")
    max_rerolls = st.sidebar.slider(
        "Max Rerolls per Deal",
        min_value=0,
        max_value=10,
        value=3
    )

    reroll_cost = st.sidebar.number_input(
        "Reroll Cost ($)",
        min_value=0,
        max_value=50,
        value=5
    )

    # Strategy settings
    st.sidebar.subheader("Strategy")

    selection_strategy = st.sidebar.selectbox(
        "Deal Selection",
        ["Secretary Problem", "Threshold", "Best (Oracle)", "Q-Learning", "TD Learning"],
        index=0
    )

    acceptance_strategy = st.sidebar.selectbox(
        "Deal Acceptance",
        ["Threshold", "Conservative", "Aggressive", "Learned"],
        index=0
    )

    # Session settings
    st.sidebar.subheader("Session")
    max_deals = st.sidebar.slider(
        "Max Deals to Play",
        min_value=1,
        max_value=50,
        value=10
    )

    start_seed = st.sidebar.number_input(
        "Starting Seed",
        min_value=0,
        max_value=100000,
        value=1000
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Session Status")

        if 'multideal_session' not in st.session_state:
            st.session_state.multideal_session = None
            st.session_state.multideal_results = []

        # Status display
        if st.session_state.multideal_session:
            session = st.session_state.multideal_session

            # Progress
            progress = len(st.session_state.multideal_results) / max_deals
            st.progress(progress)

            # Metrics
            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                st.metric("Bankroll", f"${session['bankroll']:.2f}")

            with col_b:
                profit = session['bankroll'] - starting_bankroll
                st.metric("Profit", f"${profit:+.2f}",
                         delta=f"${profit:+.2f}")

            with col_c:
                st.metric("Deals Played", len(st.session_state.multideal_results))

            with col_d:
                st.metric("Rerolls", f"{session['rerolls']}/{max_rerolls}")

        else:
            st.info("Configure settings and click 'Start Session' to begin")

    with col2:
        st.subheader("🎮 Controls")

        if st.button("🚀 Start New Session", type="primary", use_container_width=True):
            # Create strategy
            selector_map = {
                "Secretary Problem": "secretary",
                "Threshold": "threshold",
                "Best (Oracle)": "best",
                "Q-Learning": "q_learning",
                "TD Learning": "td_learning"
            }

            acceptor_map = {
                "Threshold": "threshold",
                "Conservative": "conservative",
                "Aggressive": "aggressive",
                "Learned": "value"
            }

            try:
                # Create AI strategy if available
                if selection_strategy in ["Q-Learning", "TD Learning"]:
                    strategy_type = selector_map[selection_strategy]
                    policy_path = f"policies/{strategy_type.replace('_', '_')}_policy.json"

                    selector, acceptor = create_ai_deal_strategy(
                        strategy_type=strategy_type,
                        policy_path=policy_path if Path(policy_path).exists() else None
                    )
                else:
                    selector = DealSelector(strategy=selector_map[selection_strategy])
                    acceptor = DealAcceptor(strategy=acceptor_map[acceptance_strategy])

                # Create solver
                solver = HeuristicSolver()

                # Initialize session
                st.session_state.multideal_session = {
                    'manager': MultidealManager(
                        starting_bankroll=starting_bankroll,
                        cost_per_game=cost_per_game,
                        reward_per_card=reward_per_card,
                        max_rerolls=max_rerolls,
                        reroll_cost=reroll_cost,
                        solver=solver,
                        selector=selector,
                        acceptor=acceptor
                    ),
                    'bankroll': starting_bankroll,
                    'rerolls': max_rerolls,
                    'max_deals': max_deals,
                    'start_seed': start_seed
                }
                st.session_state.multideal_results = []
                st.success("Session started!")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to start session: {e}")

        if st.session_state.multideal_session:
            if st.button("⏭️ Play Next Deal", use_container_width=True):
                # Run one deal
                manager = st.session_state.multideal_session['manager']

                # This is simplified - in full version would show deal evaluation
                seed_counter = st.session_state.multideal_session['start_seed'] + len(st.session_state.multideal_results)

                game = Game(seed=seed_counter)
                game.deal()

                # Simple evaluation
                from optimization.meta.deal_evaluator import evaluate_deal_quality
                quality = evaluate_deal_quality(game.state)

                # Auto-accept for demo (full version would use acceptor)
                result = manager.play_deal(game, quality)

                st.session_state.multideal_results.append({
                    'seed': seed_counter,
                    'quality': quality,
                    'cards': result.cards_played,
                    'score': result.final_score,
                    'profit': result.profit,
                    'won': result.won
                })

                st.session_state.multideal_session['bankroll'] = manager.bankroll
                st.session_state.multideal_session['rerolls'] = manager.rerolls_remaining

                st.success(f"Deal played! {result.cards_played}/52 cards, ${result.profit:+.0f} profit")
                st.rerun()

            if st.button("🔄 Reset Session", use_container_width=True):
                st.session_state.multideal_session = None
                st.session_state.multideal_results = []
                st.rerun()

    # Results table
    if st.session_state.multideal_results:
        st.subheader("📋 Deal History")

        import pandas as pd
        df = pd.DataFrame(st.session_state.multideal_results)

        # Format columns
        df['Quality'] = df['quality'].apply(lambda x: f"{x:.1f}")
        df['Result'] = df['cards'].astype(str) + "/52"
        df['Score'] = df['score'].apply(lambda x: f"${x:.0f}")
        df['Profit'] = df['profit'].apply(lambda x: f"${x:+.0f}")
        df['Won'] = df['won'].apply(lambda x: "✓" if x else "✗")

        display_df = df[['seed', 'Quality', 'Result', 'Score', 'Profit', 'Won']]
        display_df.columns = ['Seed', 'Quality', 'Cards', 'Score', 'Profit', 'Won']

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Summary statistics
        st.subheader("📈 Session Summary")

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            avg_cards = df['cards'].mean()
            st.metric("Avg Cards", f"{avg_cards:.1f}/52")

        with col_b:
            total_profit = df['profit'].sum()
            st.metric("Total Profit", f"${total_profit:+.2f}")

        with col_c:
            win_rate = df['won'].mean() * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")

        with col_d:
            avg_quality = df['quality'].mean()
            st.metric("Avg Quality", f"{avg_quality:.1f}")


if __name__ == "__main__":
    render_multideal_mode()

#!/usr/bin/env python3
"""
Vegas Solitaire Optimization - Streamlit Web App

An interactive web application for playing, watching, and analyzing
Vegas Solitaire with AI solvers.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Vegas Solitaire AI",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for card styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .card {
        display: inline-block;
        width: 60px;
        height: 84px;
        border: 2px solid #333;
        border-radius: 6px;
        text-align: center;
        line-height: 84px;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2px;
        background: white;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .card-red {
        color: #dc3545;
    }
    .card-black {
        color: #212529;
    }
    .card-back {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .card-empty {
        background: #f8f9fa;
        border: 2px dashed #adb5bd;
        box-shadow: none;
    }
    .tableau-column {
        display: inline-block;
        vertical-align: top;
        margin: 0 5px;
        min-height: 400px;
    }
    .foundation-pile {
        display: inline-block;
        margin: 0 5px;
    }
    .score-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""

    # Sidebar navigation
    st.sidebar.title("🃏 Vegas Solitaire AI")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home", "🎮 Play Mode", "👀 Watch Mode", "📊 Benchmark", "📈 Analysis",
         "🎰 Multi-Deal Mode", "🔬 Empirical Analysis", "🧠 Learned Evaluation"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This app demonstrates AI solvers for Vegas Solitaire using:
    - 🎲 Random Search
    - 🧠 Heuristic Strategy
    - 🌲 Monte Carlo Tree Search
    - 🧬 Q-Learning & TD Learning
    - 🎰 Multi-Deal Optimization
    - 🔬 Empirical Evaluation
    - 🧠 Machine-Learned Functions
    """)

    # Route to appropriate page
    if page == "🏠 Home":
        show_home()
    elif page == "🎮 Play Mode":
        from streamlit_app.pages import play_mode
        play_mode.show()
    elif page == "👀 Watch Mode":
        from streamlit_app.pages import watch_mode
        watch_mode.show()
    elif page == "📊 Benchmark":
        from streamlit_app.pages import benchmark_mode
        benchmark_mode.show()
    elif page == "📈 Analysis":
        from streamlit_app.pages import analysis_mode
        analysis_mode.show()
    elif page == "🎰 Multi-Deal Mode":
        from streamlit_app.pages import multideal_mode
        multideal_mode.render_multideal_mode()
    elif page == "🔬 Empirical Analysis":
        from streamlit_app.pages import empirical_analysis
        empirical_analysis.render_empirical_analysis()
    elif page == "🧠 Learned Evaluation":
        from streamlit_app.pages import learned_evaluation
        learned_evaluation.render_learned_evaluation()


def show_home():
    """Display the home page."""
    st.markdown('<div class="main-header">🃏 Vegas Solitaire AI</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Welcome to Vegas Solitaire Optimization!

    This interactive application lets you explore different AI approaches to solving Vegas Solitaire,
    a challenging variant of Klondike Solitaire with Vegas scoring rules.

    ### 🎯 Game Rules
    - **Draw-3** from stock pile (3 passes through the deck)
    - **7-column tableau** with standard Klondike stacking
    - **Vegas Scoring**: Start at -$52, earn $5 per card on foundation
    - **Goal**: Maximize profit (win = $208, best possible = $260)

    ### 🤖 Available Solvers
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🎲 Random Solver
        - Baseline performance
        - ~3% win rate
        - Fast execution
        """)

    with col2:
        st.markdown("""
        #### 🧠 Heuristic Solver
        - Hand-crafted strategy
        - Prioritizes foundations
        - Rule-based decisions
        """)

    with col3:
        st.markdown("""
        #### 🌲 MCTS Solver
        - Tree search with UCB1
        - 1000 simulations/move
        - Expected 30-40% wins
        """)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        #### 🧬 Q-Learning
        - Learns action-values Q(s,a)
        - Off-policy RL algorithm
        - Improves with training
        """)

    with col5:
        st.markdown("""
        #### 📊 TD Learning
        - Learns state values V(s)
        - Eligibility traces (TD-λ)
        - Bootstrapped learning
        """)

    with col6:
        st.markdown("""
        #### 🎰 Multi-Deal Mode
        - Bankroll management
        - Reroll optimization
        - Secretary problem
        """)

    col7, col8, col9 = st.columns(3)

    with col7:
        st.markdown("""
        #### 🔬 Empirical Evaluation
        - Data-driven deal quality
        - Solver performance measurement
        - Feature correlation analysis
        """)

    with col8:
        st.markdown("""
        #### 🧠 Machine-Generated Eval
        - Neural network training
        - Automatic pattern discovery
        - Self-play learning
        """)

    with col9:
        st.markdown("""
        #### 🚀 Coming Soon
        - Deep Q-Networks (DQN)
        - Policy Gradient methods
        - AlphaZero-style training
        """)

    st.markdown("---")

    # Quick stats
    st.subheader("📊 Quick Stats")

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Total Tests", "202", delta="All Passing ✅")

    with stat_col2:
        st.metric("Game States", "∞", delta="Combinatorial")

    with stat_col3:
        st.metric("Avg Moves/Game", "~150", delta="Varies by solver")

    with stat_col4:
        st.metric("Best Solver", "MCTS", delta="TBD")

    st.markdown("---")

    # Getting started
    st.subheader("🚀 Getting Started")

    st.markdown("""
    1. **🎮 Play Mode**: Try playing the game yourself with an interactive board
    2. **👀 Watch Mode**: Watch different AI solvers play and learn from their strategies
    3. **📊 Benchmark**: Run systematic comparisons between solvers
    4. **📈 Analysis**: Deep dive into game statistics and solver performance

    Use the sidebar to navigate between modes!
    """)


if __name__ == "__main__":
    main()

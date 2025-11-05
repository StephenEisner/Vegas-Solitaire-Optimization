"""
Empirical Deal Analysis page for Streamlit app.

Analyze deal quality empirically using AI solvers to determine
what makes deals actually worth accepting.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from game.core.game import Game
from optimization.solvers.heuristic_solver import HeuristicSolver
from optimization.evaluation.empirical_deal_quality import (
    EmpiricalDealEvaluator,
    LearnedDealQualityPredictor
)


def render_empirical_analysis():
    """Render the empirical analysis page."""
    st.title("🔬 Empirical Deal Analysis")

    st.markdown("""
    Use AI solvers to empirically evaluate deal quality.
    Learn what deals are actually worth accepting by playing them!
    """)

    # Sidebar
    st.sidebar.header("⚙️ Analysis Settings")

    mode = st.sidebar.radio(
        "Mode",
        ["Evaluate Deals", "Load Existing Dataset", "Compare Predictions"]
    )

    # Main content based on mode
    if mode == "Evaluate Deals":
        render_evaluate_deals()
    elif mode == "Load Existing Dataset":
        render_load_dataset()
    else:
        render_compare_predictions()


def render_evaluate_deals():
    """Render deal evaluation interface."""
    st.subheader("📊 Evaluate New Deals")

    col1, col2 = st.columns(2)

    with col1:
        num_deals = st.number_input(
            "Number of Deals to Evaluate",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

        start_seed = st.number_input(
            "Starting Seed",
            min_value=0,
            max_value=100000,
            value=5000
        )

    with col2:
        st.info("""
        **Note**: Evaluating deals takes time as each deal is played
        with all available solvers.

        - 10 deals: ~30 seconds
        - 100 deals: ~5 minutes
        - 1000 deals: ~50 minutes
        """)

    if st.button("🚀 Start Evaluation", type="primary"):
        with st.spinner("Evaluating deals..."):
            # Create solvers
            solvers = {
                'Heuristic': HeuristicSolver(),
            }

            # Create evaluator
            evaluator = EmpiricalDealEvaluator(solvers)

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Evaluate deals (simplified - full version would update progress)
            outcomes = evaluator.evaluate_many_deals(
                num_deals=num_deals,
                start_seed=start_seed,
                verbose=False
            )

            progress_bar.progress(100)
            status_text.text("✓ Evaluation complete!")

            # Save dataset
            Path("data").mkdir(exist_ok=True)
            evaluator.save_dataset(outcomes, "data/deal_outcomes.json")

            st.session_state.outcomes = outcomes
            st.success(f"✓ Evaluated {len(outcomes)} deals!")

            # Show summary
            display_outcomes_summary(outcomes)


def render_load_dataset():
    """Render dataset loading interface."""
    st.subheader("📂 Load Existing Dataset")

    dataset_path = st.text_input(
        "Dataset Path",
        value="data/deal_outcomes.json"
    )

    if st.button("📥 Load Dataset"):
        if Path(dataset_path).exists():
            try:
                # Load dataset
                solvers = {'Heuristic': HeuristicSolver()}
                evaluator = EmpiricalDealEvaluator(solvers)
                outcomes = evaluator.load_dataset(dataset_path)

                st.session_state.outcomes = outcomes
                st.success(f"✓ Loaded {len(outcomes)} deals!")

                # Show summary
                display_outcomes_summary(outcomes)

            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
        else:
            st.error(f"File not found: {dataset_path}")

    # Show info about dataset format
    with st.expander("ℹ️ Dataset Format"):
        st.markdown("""
        The dataset is a JSON file containing:
        - Deal features (initial state)
        - Outcomes from multiple solvers
        - Statistics (avg score, cards, win rate)

        Generate a dataset using the "Evaluate Deals" mode.
        """)


def render_compare_predictions():
    """Render prediction comparison interface."""
    st.subheader("🔍 Compare Heuristic vs Empirical")

    if 'outcomes' not in st.session_state:
        st.warning("No dataset loaded. Please evaluate or load deals first.")
        return

    outcomes = st.session_state.outcomes

    # Train predictor if not already done
    if 'predictor' not in st.session_state:
        with st.spinner("Training predictor..."):
            predictor = LearnedDealQualityPredictor()
            predictor.train(outcomes, target='avg_cards', verbose=False)
            st.session_state.predictor = predictor

    predictor = st.session_state.predictor

    # Show comparison
    st.markdown("### 📊 Prediction Accuracy")

    # Extract data
    heuristic_qualities = np.array([o.heuristic_quality for o in outcomes])
    avg_cards = np.array([o.avg_cards for o in outcomes])
    empirical_predictions = predictor.predict_from_outcomes(outcomes)

    # Correlations
    corr_heuristic = np.corrcoef(heuristic_qualities, avg_cards)[0, 1]
    corr_empirical = np.corrcoef(empirical_predictions, avg_cards)[0, 1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Heuristic Correlation", f"{corr_heuristic:.3f}")

    with col2:
        st.metric("Empirical Correlation", f"{corr_empirical:.3f}")

    with col3:
        improvement = ((corr_empirical - corr_heuristic) / abs(corr_heuristic)) * 100
        st.metric("Improvement", f"{improvement:+.1f}%")

    # Scatter plots
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Heuristic Quality vs Actual")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(heuristic_qualities, avg_cards, alpha=0.5)
        ax.set_xlabel("Heuristic Quality")
        ax.set_ylabel("Average Cards Played")
        ax.set_title(f"Correlation: {corr_heuristic:.3f}")

        # Add trend line
        z = np.polyfit(heuristic_qualities, avg_cards, 1)
        p = np.poly1d(z)
        ax.plot(heuristic_qualities, p(heuristic_qualities), "r--", alpha=0.8)

        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("#### Empirical Prediction vs Actual")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(empirical_predictions, avg_cards, alpha=0.5)
        ax.set_xlabel("Empirical Prediction")
        ax.set_ylabel("Average Cards Played")
        ax.set_title(f"Correlation: {corr_empirical:.3f}")

        # Add trend line
        z = np.polyfit(empirical_predictions, avg_cards, 1)
        p = np.poly1d(z)
        ax.plot(empirical_predictions, p(empirical_predictions), "r--", alpha=0.8)

        st.pyplot(fig)
        plt.close()

    # Feature importance
    st.markdown("### 🎯 Feature Importance")

    importance = np.abs(predictor.weights)
    top_n = 15
    top_indices = np.argsort(importance)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), predictor.weights[top_indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"Feature {i}" for i in top_indices])
    ax.set_xlabel("Weight")
    ax.set_title(f"Top {top_n} Most Important Features")
    ax.grid(axis='x', alpha=0.3)

    st.pyplot(fig)
    plt.close()

    # Good vs bad deals
    st.markdown("### ✅❌ Good vs Bad Deals")

    good_threshold = np.percentile(avg_cards, 75)
    bad_threshold = np.percentile(avg_cards, 25)

    good_deals = [o for o in outcomes if o.avg_cards >= good_threshold]
    bad_deals = [o for o in outcomes if o.avg_cards <= bad_threshold]

    col_g, col_b = st.columns(2)

    with col_g:
        st.metric("Good Deals (Top 25%)", f"≥{good_threshold:.1f} cards")
        st.metric("Avg Heuristic Quality", f"{np.mean([o.heuristic_quality for o in good_deals]):.1f}")
        st.metric("Avg Empirical Prediction", f"{np.mean(predictor.predict_from_outcomes(good_deals)):.1f}")

    with col_b:
        st.metric("Bad Deals (Bottom 25%)", f"≤{bad_threshold:.1f} cards")
        st.metric("Avg Heuristic Quality", f"{np.mean([o.heuristic_quality for o in bad_deals]):.1f}")
        st.metric("Avg Empirical Prediction", f"{np.mean(predictor.predict_from_outcomes(bad_deals)):.1f}")


def display_outcomes_summary(outcomes):
    """Display summary of outcomes."""
    st.markdown("### 📈 Outcomes Summary")

    avg_cards = np.array([o.avg_cards for o in outcomes])
    avg_scores = np.array([o.avg_score for o in outcomes])
    win_rates = np.array([o.win_rate for o in outcomes])
    heuristic_qualities = np.array([o.heuristic_quality for o in outcomes])

    # Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Cards", f"{np.mean(avg_cards):.1f}/52")
        st.caption(f"Range: {np.min(avg_cards):.1f} - {np.max(avg_cards):.1f}")

    with col2:
        st.metric("Avg Score", f"${np.mean(avg_scores):.1f}")
        st.caption(f"Range: ${np.min(avg_scores):.0f} - ${np.max(avg_scores):.0f}")

    with col3:
        st.metric("Avg Win Rate", f"{np.mean(win_rates)*100:.1f}%")
        never_win = np.sum(win_rates == 0)
        st.caption(f"{never_win}/{len(outcomes)} never won")

    with col4:
        corr = np.corrcoef(heuristic_qualities, avg_cards)[0, 1]
        st.metric("Heuristic Correlation", f"{corr:.3f}")
        st.caption("with actual performance")

    # Distribution plot
    st.markdown("#### Cards Played Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(avg_cards, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(avg_cards), color='red', linestyle='--', label='Median')
    ax.set_xlabel("Average Cards Played")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Cards Played Across All Deals")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    render_empirical_analysis()
